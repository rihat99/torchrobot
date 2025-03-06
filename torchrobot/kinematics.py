import torch
from .utils import homogeneous_transform
from .model import RobotModel

def ForwardKinematics(model: RobotModel, q:torch.Tensor, base_transform=None):
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
    if base_transform is None:
        base_transform = torch.eye(4, device=model.device)
    
    joint_transforms = {i: None for i in range(model.njoints)}
    body_transforms = {}

    joint_config = model.split_config(q)
    
    def recurse(joint_id, parent_transform):

        joint = model.joints[joint_id]
        config = joint_config[joint_id]
        motion = joint.forward_kinematics(config)
        
        # Total transform: parent's transform * fixed offset * joint motion.
        if joint_id == 0:
            joint_transforms[joint_id] = parent_transform @ joint.offset @ motion
        else:
            joint_transforms[joint_id] = parent_transform @ joint.offset
        T = parent_transform @ joint.offset @ motion
        
        # If a body is attached, compute its transform.
        # if joint.body is not None:
        #     body_motion = homogeneous_transform(joint.body.default_translation, joint.body.default_orientation)
        #     T_body = T @ body_motion
        #     body_transforms[joint.body.name] = T_body
        
        # Recurse for child joints.
        for child_id in model.tree.get(joint_id, []):
            recurse(child_id, T)
    
    # Start recursion from each root joint.
    recurse(0, base_transform)
    
    return joint_transforms, body_transforms