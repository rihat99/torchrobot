import torch

from .utils import *
from .model import RobotModel
from .robot_data import RobotData


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
        A = adjoint_transform(inverse_homogeneous_transform(joint.offset @ joint.motion)).transpose(-2, -1)

        joint_internal_forces[joint_id] += \
            (A @ child_force.unsqueeze(-1)).squeeze(-1)

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