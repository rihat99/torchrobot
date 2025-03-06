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

def axis_angle_to_quaternion(axis_angle):

    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(axis_angle)  # Compute the rotation angle

    if theta < 1e-10:  # Handle near-zero rotation (return identity quaternion)
        return np.array([0.0, 0.0, 0.0, 1.0])

    axis = axis_angle / theta  # Normalize to get the unit axis
    quaternion = Rotation.from_rotvec(axis_angle).as_quat()

    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

    return quaternion  # [x, y, z, w]


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