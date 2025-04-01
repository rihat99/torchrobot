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

    @staticmethod
    def differentiate(q0, q1):
        raise NotImplementedError
    
    def integrate(self, q, v, a, dt):
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

    @staticmethod
    def differentiate(q0, q1):
        """
        Computes the batched logarithmic map (difference) between two batches of unit quaternions.
        
        Args:
            q0: Tensor of shape (B, 4) - base quaternions (w, x, y, z)
            q1: Tensor of shape (B, 4) - target quaternions (w, x, y, z)

        Returns:
            dq: Tensor of shape (B, 3) - rotation vector (axis-angle representation)
        """
        # Ensure inputs are normalized

        q0 = q0 / (q0.norm(dim=-1, keepdim=True) + 1e-8)
        q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute relative quaternion: q_rel = q0* (inverse) * q1
        w0, xyz0 = q0[..., :1], q0[..., 1:]
        w1, xyz1 = q1[..., :1], q1[..., 1:]

        # Quaternion conjugate (inverse for unit quats)
        q0_conj = torch.cat([w0, -xyz0], dim=-1)

        # Hamilton product: q_rel = q0_conj * q1
        w0, x0, y0, z0 = q0_conj[..., 0], q0_conj[..., 1], q0_conj[..., 2], q0_conj[..., 3]
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]

        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        q_rel = torch.stack([w, x, y, z], dim=-1) + 1e-8

        # Normalize again (for safety)
        q_rel = q_rel / (q_rel.norm(dim=-1, keepdim=True) + 1e-8)

        # Extract axis and angle
        w = q_rel[..., 0].clamp(-1.0, 1.0)  # for numerical safety
        v = q_rel[..., 1:]

        # Use the atan2 formulation: note that ||v|| is sin(theta/2)
        r = v.norm(dim=-1, keepdim=True)  # equals sin(theta/2)
        # theta = 2 * atan2(||v||, w)
        theta = 2.0 * torch.atan2(r, w.unsqueeze(-1))

        # Define a differentiable sinc function: sinc(x) = sin(x)/x with Taylor expansion near zero
        def sinc(x):
            small = x.abs() < 1e-4
            return torch.where(small, 1 - x**2/6.0, torch.sin(x)/x)

        half_theta = theta / 2.0
        # Compute the scale factor in a fully differentiable manner.
        # Note that as theta -> 0, sin(theta/2)/(theta/2) -> 1, so scale -> 2.
        scale = 2.0 / (sinc(half_theta) + 1e-8)

        # Final rotation vector
        dq = v * scale

        return dq

    def integrate(self, q, v, a, dt):
        """
        Integrate spherical joint using Lie group integration (SO(3))

        Args:
            q: (..., 4) quaternion [w, x, y, z]
            v: (..., 3) angular velocity
            a: (..., 3) angular acceleration
            dt: float

        Returns:
            q_next: (..., 4)
            w_next: (..., 3)
        """
        v_next = v + a * dt
        delta_rotvec = v_next * dt  # (rotation vector)
        # delta_rotvec = v * dt  # (rotation vector) (semi-implicit)
        delta_q = exp_map_so3(delta_rotvec)  # (quaternion)

        # Quaternion multiplication: q_next = q * delta_q

        q_next = quat_mul(q, delta_q)
        q_next = q_next / q_next.norm(dim=-1, keepdim=True)  # Normalize for safety

        return q_next, v_next

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

    @staticmethod
    def differentiate(q1, q2, eps=1e-8):
        # Split into translation and rotation parts:
        p1, p2 = q1[:, :3], q2[:, :3]
        quat1, quat2 = q1[:, 3:], q2[:, 3:]
        
        # Compute rotation matrix from the first quaternion:
        R1 = quat_to_rot(quat1)  # shape (B, 3, 3)
        
        # Relative translation in the frame of the first configuration:
        p_diff = (p2 - p1).unsqueeze(2)            # shape (B, 3, 1)
        p_rel = torch.bmm(R1.transpose(1, 2), p_diff).squeeze(2)  # shape (B, 3)
        
        # Compute relative rotation: q_rel = conj(quat1) * quat2
        q1_conj = quat_conjugate(quat1)
        q_rel = quat_mul(q1_conj, quat2)
        # Normalize to avoid numerical drift:
        q_rel = q_rel / (q_rel.norm(dim=1, keepdim=True) + eps)
        
        # Compute the rotational error (log map of SO(3)):
        phi = so3_log(q_rel)  # shape (B, 3)
        
        # Compute theta as the norm of the rotation vector (without adding eps)
        theta = torch.norm(phi, dim=1, keepdim=True)  # shape (B, 1)
        
        # Precompute some matrices for the left Jacobian inverse:
        B = q1.shape[0]
        I = torch.eye(3, device=q1.device, dtype=q1.dtype).unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
        hat_phi = skew_symmetric(phi)  # (B, 3, 3)
        hat_phi2 = torch.bmm(hat_phi, hat_phi)  # (B, 3, 3)
        
        # Define a helper function to compute the factor in a differentiable manner.
        def safe_factor(theta, eps=1e-8, small_thresh=1e-3):
            # Compute sin and cos of theta safely within this function
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            # Protect denominators by adding eps to avoid division by zero.
            # Note: Even though these values aren't used when theta is small,
            # they must not produce NaNs.
            factor_large = 1.0 / (theta**2 + eps) - (1 + cos_theta) / (2 * theta * sin_theta + eps)
            # Use Taylor expansion for small theta: factor ≈ 1/12 - theta^2/720
            factor_small = 1.0/12.0 - theta**2/720.0
            # Use torch.where to choose the appropriate factor.
            return torch.where(theta < small_thresh, factor_small, factor_large)
        
        # Compute the safe factor:
        factor = safe_factor(theta)
        
        # Compute the inverse left Jacobian:
        # J_inv = I - 0.5 * hat_phi + factor * hat_phi^2
        # (unsqueeze factor to match dimensions)
        J_inv = I - 0.5 * hat_phi + factor.unsqueeze(-1) * hat_phi2  # shape (B, 3, 3)
        
        # Compute translational error corrected by the inverse left Jacobian:
        rho = torch.bmm(J_inv, p_rel.unsqueeze(2)).squeeze(2)  # shape (B, 3)
        
        # Concatenate: (translation error, rotation error)
        diff = torch.cat([rho, phi], dim=1)  # shape (B, 6)
        return diff

    def integrate(self, s, v, a, dt):
        """
        Integrate free-flyer joint (SE(3)) using Lie group exponential map.

        Args:
            p: (B, 3) current position
            q: (B, 4) current orientation (unit quaternion)
            v: (B, 6) current spatial velocity [omega, v_lin] in base frame
            a: (B, 6) current spatial acceleration [alpha, a_lin] in base frame
            dt: float time step

        Returns:
            p_next: (B, 3) next position
            q_next: (B, 4) next orientation
            v_next: (B, 6) next spatial velocity
        """
        p, q = s[..., :3], s[..., 3:]

        # 1. Integrate acceleration
        v_next = v + a * dt

        # 2. Integrate motion using exponential map
        delta_q, delta_p = exp_map_se3(v_next * dt)  # ∆pose in local frame
        # delta_q, delta_p = exp_map_se3(v * dt)  # ∆pose in local frame (semi-implicit)

        # Rotate delta_p into world frame
        delta_p_world = quat_rotate(q, delta_p)

        # Update position and orientation
        p_next = p + delta_p_world
        q_next = quat_mul(q, delta_q)
        q_next = q_next / q_next.norm(dim=-1, keepdim=True)

        s_next = torch.cat([p_next, q_next], dim=-1)

        return s_next, v_next

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