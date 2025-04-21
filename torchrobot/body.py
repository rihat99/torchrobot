import torch
import numpy as np
from .utils import rgb_to_hex, skew_symmetric

import meshcat.geometry as g

class RigidBody:
    """
    Rigid body defined by its name, mass, inertia, and a pose relative to its joint.

    """
    def __init__(self, name, parent_joint_id, mass, com, inertia, offset=None, device='cpu'):
        self.device = device

        self.name = name
        self.parent_joint_id = parent_joint_id
        self.com = com
        self.mass = mass
        self.inertia = inertia  # 3x3 tensor
        self.offset = offset

        self.compute_inertia_matrix()


    def compute_inertia_matrix(self):
        """
        Computes the inertia matrix of the rigid body.
        """
        skew_com = skew_symmetric(self.com)
        self.inertia_matrix = torch.zeros(6, 6, device=self.device)

        self.inertia_matrix[:3, :3] = self.mass * torch.eye(3, device=self.device)
        self.inertia_matrix[:3, 3:] = -self.mass * skew_com
        self.inertia_matrix[3:, :3] = self.mass * skew_com
        self.inertia_matrix[3:, 3:] = self.inertia - self.mass * skew_com @ skew_com
        # self.inertia_matrix[3:, 3:] = self.inertia



class GeometryObject:
    """
    Geometry object defined by its name, type, and pose relative to its body.
    The pose is represented by:

    """
    def __init__(self, name, shape, offset, parent_id=None, color=None, device='cpu'):
        self.name = name
        self.offset = offset
        self.parent_id = parent_id
        self.shape = shape

        if color is None:
            color = g.MeshLambertMaterial(
                color=rgb_to_hex(np.array([0.5, 0.5, 0.5])),
                reflectivity=1.0,
            )
        self.color = g.MeshLambertMaterial(
            color=rgb_to_hex(np.array(color[:3])),
            reflectivity=float(color[3]),
        )

        self.device = device  