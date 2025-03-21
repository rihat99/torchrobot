import torch

from .body import RigidBody
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
        self.body = None
        self.gemetry_objects = []
        self.joint_type = None
        self.device = device

        self.motion = None
        self.twist = None
        self.acceleration = None

    def compute_difference(self, q0, q1):
        raise NotImplementedError

    def process_config(self, config):
        raise NotImplementedError
    
    def get_motion_subspace(self):
        """
        Compute the motion subspace of the joint.
        Returns: 6xnv motion subspace matrix.
        """
        raise NotImplementedError
    
    def get_torque(self, force):
        """
        Compute the joint torque given the joint force.
        force: 6D force tensor.
        Returns: coresponging torque tensor.
        """

        raise NotImplementedError

    def forward_kinematics(self, config, parent_transform, parent_twist, parent_acceleration):
        
        motion, twist, acceleration = self.process_config(config)
        self.motion = motion
        self.twist = twist
        self.acceleration = acceleration

        # Position of the joint in the world frame.
        joint_transform = parent_transform @ self.offset @ motion

        # Compute the spatial velocity of the joint in local frame.
        A = adjoint_transform(inverse_homogeneous_transform(self.offset @ motion))
        # A = adjoint_transform(self.offset @ motion)
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

    def compute_difference(self, q1, q0):
        """
        Computes the batched logarithmic map (difference) between two batches of unit quaternions.
        
        Args:
            q0: Tensor of shape (B, 4) - base quaternions (w, x, y, z)
            q1: Tensor of shape (B, 4) - target quaternions (w, x, y, z)

        Returns:
            dq: Tensor of shape (B, 3) - rotation vector (axis-angle representation)
        """
        # Ensure inputs are normalized
        q0 = q0 / q0.norm(dim=1, keepdim=True)
        q1 = q1 / q1.norm(dim=1, keepdim=True)

        # Compute relative quaternion: q_rel = q0* (inverse) * q1
        w0, xyz0 = q0[:, :1], q0[:, 1:]
        w1, xyz1 = q1[:, :1], q1[:, 1:]

        # Quaternion conjugate (inverse for unit quats)
        q0_conj = torch.cat([w0, -xyz0], dim=1)

        # Hamilton product: q_rel = q0_conj * q1
        w0, x0, y0, z0 = q0_conj[:, 0], q0_conj[:, 1], q0_conj[:, 2], q0_conj[:, 3]
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]

        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        q_rel = torch.stack([w, x, y, z], dim=1)

        # Normalize again (for safety)
        q_rel = q_rel / q_rel.norm(dim=1, keepdim=True)

        # Extract axis and angle
        w = q_rel[:, 0].clamp(-1.0, 1.0)  # for numerical safety
        v = q_rel[:, 1:]

        theta = 2.0 * torch.acos(w)  # angle
        sin_theta_half = torch.sqrt(1.0 - w**2 + 1e-8)  # avoid divide-by-zero

        # Avoid division by zero by using Taylor expansion for small angles
        small_angle = sin_theta_half < 1e-4
        scale = theta / sin_theta_half

        # For small angles, use linear approximation: scale ≈ 2
        scale[small_angle] = 2.0

        # Final rotation vector
        dq = v * scale.unsqueeze(1)

        return dq

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
            twist = torch.cat([torch.zeros(shape + (3,), device=self.device), config['v']], dim=-1)
        else:
            twist = torch.zeros(shape + (6,), device=self.device)

        if config['a'] is not None:
            acceleration = torch.cat([torch.zeros(shape + (3,), device=self.device), config['a']], dim=-1)
        else:
            acceleration = torch.zeros(shape + (6,), device=self.device)

        return motion, twist, acceleration
    
    def get_torque(self, force):
        """
        Compute the joint torque given the joint force.
        force: 6D force tensor.
        Returns: coresponging torque tensor.
        """

        return force[..., 3:6]
    
    def get_motion_subspace(self):
        S = torch.zeros(6, 3, device=self.device)
        S[3:6, :] = torch.eye(3, device=self.device)
        return S

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

    def compute_difference(self, s1, s0):
        """
        Batched SE(3) log map (difference) between poses (p0, q0) and (p1, q1)

        Args:
            p0: Tensor (B, 3) - initial translation
            q0: Tensor (B, 4) - initial quaternion (w, x, y, z)
            p1: Tensor (B, 3) - target translation
            q1: Tensor (B, 4) - target quaternion

        Returns:
            delta_q: Tensor (B, 6) - [angular (3), linear (3)] velocity vector in base frame
        """
        p0, q0 = s0[..., :3], s0[..., 3:]
        p1, q1 = s1[..., :3], s1[..., 3:]

        # --- Normalize quaternions ---
        q0 = q0 / q0.norm(dim=1, keepdim=True)
        q1 = q1 / q1.norm(dim=1, keepdim=True)

        # --- Rotation part (log of relative quaternion) ---
        w0, xyz0 = q0[:, :1], q0[:, 1:]
        w1, xyz1 = q1[:, :1], q1[:, 1:]

        q0_conj = torch.cat([w0, -xyz0], dim=1)

        # Hamilton product: q_rel = q0_conj * q1
        def quat_mul(q, r):
            w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], dim=1)

        q_rel = quat_mul(q0_conj, q1)
        q_rel = q_rel / q_rel.norm(dim=1, keepdim=True)

        w = q_rel[:, 0].clamp(-1.0, 1.0)
        v = q_rel[:, 1:]
        theta = 2.0 * torch.acos(w)
        sin_theta_half = torch.sqrt(1.0 - w**2 + 1e-8)

        small = sin_theta_half < 1e-4
        scale = theta / sin_theta_half
        scale[small] = 2.0
        omega = v * scale.unsqueeze(1)  # (B, 3)

        # --- Translation part ---
        # Compute relative translation in world frame
        dp = p1 - p0  # (B, 3)

        # Rotate into base frame using q0⁻¹
        # Equivalent to applying q0_conj to dp
        def quat_rotate(q, v):
            # Rotate vector v by quaternion q (batch)
            qvec = q[:, 1:]
            uv = torch.cross(qvec, v, dim=1)
            uuv = torch.cross(qvec, uv, dim=1)
            return v + 2 * (q[:, :1] * uv + uuv)

        v_linear = quat_rotate(q0_conj, dp)  # (B, 3)

        # --- Combine into motion vector ---
        delta_q = torch.cat([v_linear, omega], dim=1)  # (B, 6)
        return delta_q

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
            twist = torch.cat([config['v'][..., 0:3], config['v'][..., 3:6]], dim=-1)
        else:
            twist = torch.zeros(shape + (6,), device=self.device)

        if config['a'] is not None:
            acceleration = torch.cat([config['a'][..., 0:3], config['a'][..., 3:6]], dim=-1)
        else:
            acceleration = torch.zeros(shape + (6,), device=self.device)

        return motion, twist, acceleration
    
    
    def get_torque(self, force):
        """
        Compute the joint torque given the joint force.
        force: 6D force tensor.
        Returns: coresponging torque tensor.
        """
        return force
    
    def get_motion_subspace(self):
        S = torch.eye(6, device=self.device)
        return S

    
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