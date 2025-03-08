import torch
from .utils import *

class Joint:
    """
    Base class for a joint.
    
    Attributes:
      - name: joint name.
      - offset: fixed 4x4 transformation from the parent frame to the joint frame.
      - parent: the parent joint id (None if root).
      - body: attached body (if any).
      - joint_type: type of the joint (e.g., 'spherical' or 'freeflyer').
    """
    def __init__(self, name, offset, parent_id=None, device='cpu'):
        self.name = name
        self.offset = offset  # fixed 4x4 transform
        self.parent_id = parent_id
        self.bodies = []
        self.gemetry_objects = []
        self.joint_type = None
        self.device = device

    def process_config(self, config):
        raise NotImplementedError

    def forward_kinematics(self, config, parent_transform, parent_twist, parent_acceleration):
        
        motion, twist, acceleration = self.process_config(config)

        # Position of the joint in the world frame.
        joint_transform = parent_transform @ self.offset @ motion

        # Compute the spatial velocity of the joint in local frame.
        A = adjoint_transform(inverse_homogeneous_transform(self.offset @ motion))
        local_twist = (A @ parent_twist.unsqueeze(-1)).squeeze(-1) + twist

        # Compute the spatial acceleration of the joint in local frame.
        local_acceleration = \
            (A @ parent_acceleration.unsqueeze(-1)).squeeze(-1) + \
            acceleration + \
            (adjoint_bracket_operator(local_twist) @ twist.unsqueeze(-1)).squeeze(-1)

        return joint_transform, local_twist, local_acceleration

class SphericalJoint(Joint):
    """
    Spherical joint: 3-DOF rotation represented by a quaternion.
    """
    def __init__(self, name, offset, parent_id=None, device='cpu'):
        super().__init__(name, offset, parent_id, device)

        self.joint_type = 'spherical'
        self.nq = 4
        self.nv = 3
        self.default_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.default_v = torch.zeros(3, device=self.device)

    def process_config(self, config):
        """
        Compute the forward kinematics of the spherical joint.
        q: quaternion tensor.
        Returns: 4x4 transformation matrix.
        """
        shape = config['q'].shape[:-1]
        translation = torch.zeros(shape + (3,), device=self.device)
        motion = homogeneous_transform(translation, config['q'])

        if config['v'] is not None:
            twist = torch.cat([config['v'], torch.zeros(shape + (3,), device=self.device)], dim=-1)
            # twist = torch.cat([torch.zeros(shape + (3,), device=self.device), config['v']], dim=-1)
        else:
            twist = torch.zeros(shape + (6,), device=self.device)

        if config['a'] is not None:
            acceleration = torch.cat([config['a'], torch.zeros(shape + (3,), device=self.device)], dim=-1)
            # acceleration = torch.cat([torch.zeros(shape + (3,), device=self.device), config['a']], dim=-1)
        else:
            acceleration = torch.zeros(shape + (6,), device=self.device)

        return motion, twist, acceleration

class FreeFlyerJoint(Joint):
    """
    FreeFlyer joint: 6-DOF (translation and rotation).
    Renamed from SE3Joint.
    """
    def __init__(self, name, offset, parent_id=None, device='cpu'):
        super().__init__(name, offset, parent_id, device)

        self.joint_type = 'freeflyer'
        self.nq = 7
        self.nv = 6
        self.default_q = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.default_v = torch.zeros(6, device=self.device)

    def process_config(self, config):
        """
        Compute the forward kinematics of the freeflyer joint.
        q: quaternion tensor.
        Returns: 4x4 transformation matrix.
        """
        shape = config['q'].shape[:-1]

        translation = config['q'][..., 0:3]
        orientation = config['q'][..., 3:7]
        motion = homogeneous_transform(translation, orientation)

        if config['v'] is not None:
            twist = torch.cat([config['v'][..., 3:6], config['v'][..., 0:3]], dim=-1)
            # twist = torch.cat([config['v'][..., 0:3], config['v'][..., 3:6]], dim=-1)
        else:
            twist = torch.zeros(shape + (6,), device=self.device)

        if config['a'] is not None:
            acceleration = torch.cat([config['a'][..., 3:6], config['a'][..., 0:3]], dim=-1)
            # acceleration = torch.cat([config['a'][..., 0:3], config['a'][..., 3:6]], dim=-1)
        else:
            acceleration = torch.zeros(shape + (6,), device=self.device)

        return motion, twist, acceleration

    
class FixedJoint(Joint):
    """
    Fixed joint: 0-DOF (fixed transformation).
    """
    def __init__(self, name, offset, parent_id=None, device='cpu'):
        super().__init__(name, offset, parent_id, device)

        self.joint_type = 'fixed'
        self.nq = 0
        self.nv = 0
        self.default_q = torch.tensor([], device=self.device)
        self.default_v = torch.tensor([], device=self.device)

    def process_config(self, q):
        """
        Compute the forward kinematics of the fixed joint.
        q: empty tensor.
        Returns: 4x4 transformation matrix.
        """
        return torch.eye(4, device=self.device), torch.zeros(6, device=self.device), torch.zeros(6, device=self.device)