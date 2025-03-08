import torch

from .utils import *
from .model import RobotModel
from .robot_data import RobotData

def ForwardKinematics(
        model: RobotModel, 
        data:RobotData, 
        q:torch.Tensor, 
        v:torch.Tensor=None,
        a:torch.Tensor=None,):
    """
    Computes forward kinematics for the robot model given a joint configuration.
    
    Parameters:
      - model: an instance of RobotModel containing the static kinematic structure.
      - joint_config: a dictionary mapping joint id to its configuration.
          For spherical joints: {'q': tensor(4)}
          For freeflyer joints: {'translation': tensor(3), 'q': tensor(4)}
      - base_transform: 4x4 tensor representing the base frame transform (default identity).
      
    Returns:
      - joint_transforms: dict mapping joint id to its 4x4 world transform.
      - body_transforms: dict mapping body name to its 4x4 world transform.
    """
    
    base_transform = torch.eye(4, device=model.device)
    base_twist = torch.zeros(6, device=q.device)
    base_acceleration = torch.zeros(6, device=q.device)
    if q.dim() > 1:
        base_transform = base_transform.unsqueeze(0).expand(q.size(0), -1, -1)
        base_twist = base_twist.unsqueeze(0).expand(q.size(0), -1)
        base_acceleration = base_acceleration.unsqueeze(0).expand(q.size(0), -1)
    
    joint_transforms = {i: None for i in range(model.njoints)}
    joint_velocities = {i: None for i in range(model.njoints)}
    joint_accelerations = {i: None for i in range(model.njoints)}
    body_transforms = {}

    joint_config = data.split_config(q, v, a)

    
    def recurse(joint_id, parent_transform, parent_twist, parent_acceleration):

        joint = model.joints[joint_id]
        config = joint_config[joint_id]

        joint_transform, local_twist, local_acceleration = joint.forward_kinematics(config, parent_transform, parent_twist, parent_acceleration)
        joint_transforms[joint_id] = joint_transform
        joint_velocities[joint_id] = local_twist
        joint_accelerations[joint_id] = local_acceleration

        # If a body is attached, compute its transform.
        # if joint.body is not None:
        #     body_motion = homogeneous_transform(joint.body.default_translation, joint.body.default_orientation)
        #     T_body = T @ body_motion
        #     body_transforms[joint.body.name] = T_body
        
        # Recurse for child joints.
        for child_id in model.tree.get(joint_id, []):
            recurse(child_id, joint_transform, local_twist, local_acceleration)
    
    # Start recursion from each root joint.
    recurse(0, base_transform, base_twist, base_acceleration)
    
    data.joint_transforms = joint_transforms
    data.joint_velocities = joint_velocities
    data.joint_accelerations = joint_accelerations
    data.body_transforms = body_transforms