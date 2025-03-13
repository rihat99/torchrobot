import torch

from .utils import *
from .model import RobotModel
from .robot_data import RobotData
from .kinematics import ForwardKinematics

def RNEA(
        model: RobotModel, 
        data:RobotData, 
        q:torch.Tensor, 
        v:torch.Tensor=None,
        a:torch.Tensor=None,
        f_ext:torch.Tensor=None):
    
    """
    Computes inverse dynamics using the Recursive Newton-Euler Algorithm.
    """

    
    base_transform = torch.eye(4, device=model.device)
    base_twist = torch.zeros(6, device=q.device)
    base_acceleration = -model.gravity
    if q.dim() > 1:
        base_transform = base_transform.unsqueeze(0).expand(q.size(0), -1, -1)
        base_twist = base_twist.unsqueeze(0).expand(q.size(0), -1)
        base_acceleration = base_acceleration.unsqueeze(0).expand(q.size(0), -1)
    
    joint_transforms = {}
    joint_velocities = {}
    joint_accelerations = {}
    joint_internal_forces = {}

    joint_config = data.split_config(q, v, a)

    # Forward Pass 
    def recurse(joint_id, parent_transform, parent_twist, parent_acceleration):

        # skip fixed joints or joints with no body
        if model.joints[joint_id].body is None or model.joints[joint_id].joint_type == 'fixed':
            return

        joint = model.joints[joint_id]
        config = joint_config[joint_id]

        joint_transform, local_twist, local_acceleration = joint.forward_kinematics(config, parent_transform, parent_twist, parent_acceleration)
        joint_transforms[joint_id] = joint_transform
        joint_velocities[joint_id] = local_twist
        joint_accelerations[joint_id] = local_acceleration

        inertia_matrix = joint.body.inertia_matrix
        joint_internal_force = (inertia_matrix.unsqueeze(0) @ local_acceleration.unsqueeze(-1)).squeeze(-1) + \
                                 (adjoint_bracket_operator_dual(local_twist) @ (inertia_matrix.unsqueeze(0) @ local_twist.unsqueeze(-1))).squeeze(-1)

        if f_ext is not None:
            joint_internal_forces[joint_id] = joint_internal_force - f_ext[: , joint_id, :]
        else:
            joint_internal_forces[joint_id] = joint_internal_force


        # Recurse for child joints.
        for child_id in model.tree.get(joint_id, []):
            recurse(child_id, joint_transform, local_twist, local_acceleration)
    
    # Start recursion from each root joint.
    recurse(0, base_transform, base_twist, base_acceleration)

    # Backward Pass
    def recurse_backward(joint_id, child_id, child_force):

        joint = model.joints[child_id]

        # A = adjoint_transform(joint.offset @ joint.motion)
        A_star = adjoint_transform(inverse_homogeneous_transform(joint.offset @ joint.motion)).transpose(-2, -1)

        joint_internal_forces[joint_id] += (A_star @ child_force.unsqueeze(-1)).squeeze(-1)

        if joint_id not in model.connections_ids:
            parent_id = model.joints[joint_id].parent_id
            
            recurse_backward(parent_id, joint_id, joint_internal_forces[joint_id])


    for joint_id in model.end_effectors_ids:
        # if model.joints[joint_id].parent_id is not None:
        parent_id = model.joints[joint_id].parent_id
        recurse_backward(parent_id, joint_id, joint_internal_forces[joint_id])

    tau = []
    for joint_id, joint in enumerate(model.joints):
        if joint.nv > 0:
            tau.append(joint.get_torque(joint_internal_forces[joint_id]))

    tau = torch.concatenate(tau, dim=-1)

    return tau


def CRBA(
        model: RobotModel,
        data: RobotData,
        q: torch.Tensor,

):
    """
    Computes the Composite Rigid Body Algorithm to compute the mass matrix.
    
    Args:
        model: RobotModel instance.
        data: RobotData instance.
        q: Tensor of shape (batch, nq) representing joint configuration.
        
        Returns:
        M: Tensor of shape (batch, nq, nq) representing the mass matrix.
        
    """


    # forward pass (forward kinematics)
    ForwardKinematics(model, data, q)

    joint_composite_inertia = {}
    for joint_id, joint in enumerate(model.joints):
        if joint.nv > 0:
            joint_composite_inertia[joint_id] = joint.body.inertia_matrix.repeat(q.size(0), 1, 1)

    
    M = torch.zeros(q.shape[0], model.nv, model.nv, device=model.device)

    # Backward Pass
    def recurse_backward(joint_id, child_id, child_composite_inertia):

        i = child_id
        ni = model.joints[i].nv
        ii = 6 + (i-1) * 3 if i > 0 else 0
        Si = model.joints[i].get_motion_subspace().unsqueeze(0)
        F = child_composite_inertia @ Si
        M[:, ii:ii+ni, ii:ii+ni] = Si.transpose(-2, -1) @ F


        joint = model.joints[child_id]

        # A = adjoint_transform(joint.offset @ joint.motion)
        A = adjoint_transform(inverse_homogeneous_transform(joint.offset @ joint.motion))
        # A_star = A.transpose(-2, -1)
        # A_star = adjoint_transform(joint.offset @ joint.motion)

        I_c = child_composite_inertia
  
        joint_composite_inertia[joint_id] += A.transpose(-2, -1) @ I_c @ A

    
        j = joint_id
        c = child_id
        while j is not None:
            nj = model.joints[j].nv
            jj = 6 + (j-1) * 3 if j > 0 else 0
            Sj = model.joints[j].get_motion_subspace().unsqueeze(0)
            
            A = adjoint_transform(inverse_homogeneous_transform(model.joints[c].offset @ model.joints[c].motion))
            F = A.transpose(-2, -1) @ F
            M[:, jj:jj+nj, ii:ii+ni] = Sj.transpose(-2, -1) @ F
            M[:, ii:ii+ni, jj:jj+nj] = M[:, jj:jj+nj, ii:ii+ni].transpose(-2, -1)
        
            c = j
            j = model.joints[j].parent_id

        if joint_id not in model.connections_ids:
            parent_id = model.joints[joint_id].parent_id
            
            recurse_backward(parent_id, joint_id, joint_composite_inertia[joint_id])


    for joint_id in model.end_effectors_ids:
        # if model.joints[joint_id].parent_id is not None:
        parent_id = model.joints[joint_id].parent_id
        recurse_backward(parent_id, joint_id, joint_composite_inertia[joint_id])


    M[:, 0:6, 0:6] = joint_composite_inertia[0]

    # ii = 0
    # for i in range(len(model.joints)):
    #     if model.joints[i].nv > 0:
    #         nv_i = model.joints[i].nv
    #         S_i = model.joints[i].get_motion_subspace().unsqueeze(0).transpose(-2, -1)
    #         jj = 0
    #         for j in range(len(model.joints)):
    #             if model.joints[j].nv > 0:
    #                 nv_j = model.joints[j].nv

    #                 if i == j:
    #                     S_j = model.joints[j].get_motion_subspace().unsqueeze(0)
    #                     M[:, ii:ii+nv_i, jj:jj+nv_j] = S_i @ joint_composite_inertia[i] @ S_j
    #                 else:
    #                     joint = model.joints[j]
    #                     A_star = adjoint_transform(joint.offset @ joint.motion)

    #                     S_j = model.joints[j].get_motion_subspace().unsqueeze(0)
    #                     M_ij = S_i @ A_star @ joint_composite_inertia[i] @ S_j

    #                     M[:, ii:ii+nv_i, jj:jj+nv_j] = M_ij

    #                 jj += nv_j

    #         ii += nv_i

    return M, joint_composite_inertia

