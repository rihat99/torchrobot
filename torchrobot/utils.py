import torch
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation


def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # if not fast:
    #     return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )


def matrix_to_axis_angle(matrix: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    if not fast:
        return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = (
        0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :]
        + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
    )
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

    return axis_angles

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    # normalize the input quaternion
    quaternions = quaternions / quaternions.norm(p=2, dim=-1, keepdim=True)

    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)
    # angles/2 are between [-pi/2, pi/2], thus sin_half_angles_over_angles
    # can't be zero
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def homogeneous_transform(translation, q):
    """
    Create a 4x4 homogeneous transformation matrix from a translation vector (3,)
    and a quaternion (4,).
    """
    # R = quaternion_to_rotation_matrix(q)
    # T = torch.eye(4, dtype=translation.dtype, device=translation.device)
    # T[:3, :3] = R
    # T[:3, 3] = translation
    # return T

    # make code work with any input shape
    R = quaternion_to_rotation_matrix(q)
    T = torch.eye(4, dtype=translation.dtype, device=translation.device).repeat(
        *translation.shape[:-1], 1, 1
    )
    T[..., :3, :3] = R
    T[..., :3, 3] = translation

    return T


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def rgb_to_hex(color, prefix="0x"):
    '''
    Converting a RGB color tuple to a six digit code
    '''
    color = [int(c) for c in color]
    r = color[0]
    g = color[1]
    b = color[2]
    return prefix+"{0:02x}{1:02x}{2:02x}".format(r, g, b)

def normalize(v):
        """ Normalize a vector """
        return v / torch.norm(v, p=2)


def align_box_to_vector(vector, box_center, target_center):
    """
    Computes the rotation matrix and translation vector to align a box with a given 3D vector.

    Parameters:
        vector (array-like): The target direction vector (3D).
        box_center (array-like): The initial center of the box (3D).
        target_center (array-like): The desired center of the box after alignment (3D).

    Returns:
        R (numpy.ndarray): The 3x3 rotation matrix.
        T (numpy.ndarray): The 3D translation vector.
    """
    vector = normalize(vector)  # Ensure the target vector is normalized
    reference_axis = torch.tensor([0, 1, 0], device=vector.device, dtype=torch.float32)  # Assume the box is originally aligned with the Z-axis

    # Compute the rotation matrix to align Z-axis with the given vector
    rotvec = torch.cross(reference_axis, vector)  # Rotation axis
    angle = torch.arccos(torch.dot(reference_axis, vector))  # Rotation angle

    if torch.norm(rotvec, p=2) == 0:  # If already aligned, return identity matrix
        R_matrix = torch.eye(3, device=vector.device, dtype=torch.float32)
    else:
        rotvec = normalize(rotvec) * angle  # Convert axis to rotation vector
        R_matrix = Rotation.from_rotvec(rotvec.detach().cpu().numpy()).as_matrix()  # Compute rotation matrix using SciPy
        R_matrix = torch.tensor(R_matrix, device=vector.device, dtype=torch.float32)
    # Compute translation to align centers
    T_vector = target_center - R_matrix @ box_center

    return R_matrix, T_vector

def skew_symmetric(p):
    """
    Compute the skew-symmetric matrix of a 3D vector p.
    """
    p0, p1, p2 = torch.unbind(p, dim=-1)
    zero = torch.zeros_like(p0)
    return torch.stack([
        torch.stack([zero, -p2, p1], dim=-1),
        torch.stack([p2, zero, -p0], dim=-1),
        torch.stack([-p1, p0, zero], dim=-1)
    ], dim=-2)



def adjoint_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the 6x6 adjoint transformation of a 4x4 homogeneous transform T.
    T is assumed to have shape (4,4). Returns a 6x6 matrix A such that,
    for a twist ξ (6-vector), the transformed twist is A @ ξ.
    
    Note: The first three components of ξ represent the angular velocity
    in axis–angle format.
    """
    R = T[..., :3, :3]  # shape: (..., 3, 3)
    p = T[..., :3, 3]   # shape: (..., 3)

    # Construct each row of the skew symmetric matrix
    p_hat = skew_symmetric(p)

    A_upper = torch.cat([R, p_hat @ R], dim=-1)
    A_lower = torch.cat([torch.zeros_like(R), R], dim=-1)
    A = torch.cat([A_upper, A_lower], dim=-2)
    return A

def adjoint_bracket_operator(V):
    """
    Compute the Adjoint Bracket Operator (ad_V) for batched spatial velocities.
    
    Args:
        V: Tensor of shape (..., 6), where the last dimension contains (omega, v).

    Returns:
        Tensor of shape (..., 6, 6) representing the adjoint bracket operator.
    """

    omega = V[..., 3:]  # Angular velocity
    v = V[..., :3]      # Linear velocity
    
    # Compute skew-symmetric matrices
    omega_hat = skew_symmetric(omega)  # (..., 3, 3)
    v_hat = skew_symmetric(v)          # (..., 3, 3)
    
    # Construct the adjoint bracket matrix
    batch_shape = omega.shape[:-1]
    zero_block = torch.zeros(*batch_shape, 3, 3, device=V.device, dtype=V.dtype)
    

    ad_V = torch.cat([
        torch.cat([omega_hat, v_hat], dim=-1),
        torch.cat([zero_block, omega_hat], dim=-1)
    ], dim=-2)  # Concatenate to form (..., 6, 6)
    
    return ad_V

def adjoint_bracket_operator_dual(V):
    """
    Compute the Adjoint Bracket Operator (ad_V) for batched spatial velocities.
    
    Args:
        V: Tensor of shape (..., 6), where the last dimension contains (omega, v).

    Returns:
        Tensor of shape (..., 6, 6) representing the adjoint bracket operator.
    """

    omega = V[..., 3:]  # Angular velocity
    v = V[..., :3]      # Linear velocity
    
    # Compute skew-symmetric matrices
    omega_hat = skew_symmetric(omega)  # (..., 3, 3)
    v_hat = skew_symmetric(v)          # (..., 3, 3)
    
    # Construct the adjoint bracket matrix
    batch_shape = omega.shape[:-1]
    zero_block = torch.zeros(*batch_shape, 3, 3, device=V.device, dtype=V.dtype)
    

    ad_V = torch.cat([
        torch.cat([omega_hat, zero_block], dim=-1),
        torch.cat([v_hat, omega_hat], dim=-1)
    ], dim=-2)  # Concatenate to form (..., 6, 6)
    
    return ad_V



def inverse_homogeneous_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a 4x4 homogeneous transform T.
    T is assumed to have shape (4,4). Returns a 4x4 matrix T_inv.
    """
    R = T[..., :3, :3]  # shape: (..., 3, 3)
    p = T[..., :3, 3]   # shape: (..., 3)

    R_T = R.transpose(-2, -1)
    p_inv = -R_T @ p.unsqueeze(-1)

    T_inv = torch.cat([
        torch.cat([R_T, p_inv], dim=-1),
        torch.tensor([0, 0, 0, 1], device=T.device).repeat(T.shape[:-2] + (1, 1))
        ],
        dim=-2)
    return T_inv


def exp_map_so3(rotvec):
    """
    Exponential map from so(3) to SO(3) (rotation vector to quaternion)
    rotvec: (..., 3)
    Returns: (..., 4) quaternion [w, x, y, z]
    """
    theta = torch.norm(rotvec, dim=-1, keepdim=True) + 1e-8
    axis = rotvec / theta
    half_theta = 0.5 * theta

    w = torch.cos(half_theta)
    xyz = axis * torch.sin(half_theta)

    return torch.cat([w, xyz], dim=-1)

def exp_map_se3(spatial_velocity):
    """
    Exponential map from se(3) to SE(3)
    Input: (..., 6) spatial velocity = [v_lin, omega]
    Output: (..., 4), (..., 3) -> quaternion, translation
    """
    omega = spatial_velocity[..., 3:]
    v = spatial_velocity[..., :3]

    theta = torch.norm(omega, dim=-1, keepdim=True) + 1e-8
    axis = omega / theta
    half_theta = 0.5 * theta

    # Rotation part (quaternion)
    w = torch.cos(half_theta)
    xyz = axis * torch.sin(half_theta)
    delta_q = torch.cat([w, xyz], dim=-1)  # (..., 4)

    # Translation part: simplified for small rotation
    # Use first-order approximation: Δp ≈ v * dt
    # (This ignores curvature of SE(3), which is often fine for small dt)
    delta_p = v  # will be multiplied by dt outside

    return delta_q, delta_p

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_rotate(q, v):
    """ Rotate vector v by unit quaternion q (B, 4), (B, 3) """
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def quat_to_rot(q):
    """
    Convert a quaternion in w,x,y,z format to a rotation matrix.
    Input:
        q: Tensor of shape (B, 4) in (w, x, y, z) format.
    Output:
        Tensor of shape (B, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    
    R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    return R

def quat_conjugate(q):
    """
    Compute the conjugate of a quaternion (w, x, y, z).
    Input:
        q: Tensor of shape (B, 4)
    Output:
        Tensor of shape (B, 4)
    """
    w = q[:, 0:1]
    return torch.cat([w, -q[:, 1:]], dim=1)


def so3_log(q, eps=1e-8):
    """
    Computes the logarithm map of a unit quaternion in a robust, differentiable way.
    
    Args:
        q: Tensor of shape (B, 4) with q = [w, x, y, z] assumed normalized.
        eps: Small constant for numerical safety.
    
    Returns:
        phi: Tensor of shape (B, 3) representing the rotation vector.
    """
    # Extract scalar and vector parts
    w = q[:, 0:1]       # shape (B, 1)
    v = q[:, 1:]        # shape (B, 3)is 
    norm_v = v.norm(dim=1, keepdim=True)  # (B, 1)

    # Compute half the rotation angle using atan2 for numerical robustness:
    half_theta = torch.atan2(norm_v, w)
    theta = 2 * half_theta  # full rotation angle

    # When the rotation is very small, norm_v is nearly zero.
    # In that case, the ratio theta/norm_v -> 2, so we use a branch.
    ratio = torch.where(norm_v < 1e-3, torch.full_like(norm_v, 2.0), theta/(norm_v + eps))
    phi = ratio * v
    return phi