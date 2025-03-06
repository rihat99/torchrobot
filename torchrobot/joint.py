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

    def forward_kinematics(self, q):
        """
        Compute the forward kinematics of the joint.
        q: joint configuration tensor.
        Returns: 4x4 transformation matrix.
        """
        raise NotImplementedError

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

    def forward_kinematics(self, q):
        """
        Compute the forward kinematics of the spherical joint.
        q: quaternion tensor.
        Returns: 4x4 transformation matrix.
        """
        return homogeneous_transform(torch.zeros(3, device=self.device), q)

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

    def forward_kinematics(self, q):
        """
        Compute the forward kinematics of the freeflyer joint.
        q: quaternion tensor.
        Returns: 4x4 transformation matrix.
        """
        translation = q[0:3]
        q = q[3:7]
        return homogeneous_transform(translation, q)
    
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

    def forward_kinematics(self, q):
        """
        Compute the forward kinematics of the fixed joint.
        q: empty tensor.
        Returns: 4x4 transformation matrix.
        """
        return torch.eye(4, device=self.device)