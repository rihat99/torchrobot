import torch
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation

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

    A_upper = torch.cat([R, torch.zeros_like(R)], dim=-1)
    A_lower = torch.cat([p_hat @ R, R], dim=-1)
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
    omega = V[..., :3]  # Angular velocity
    v = V[..., 3:]      # Linear velocity
    
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