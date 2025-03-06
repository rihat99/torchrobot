import torch
import numpy as np
from .utils import rgb_to_hex

import meshcat.geometry as g

class RigidBody:
    """
    Rigid body defined by its name, mass, inertia, and a pose relative to its joint.
    The pose is represented by:
      - default_translation: a 3-vector (default zero)
      - default_orientation: a quaternion (default identity)
    """
    def __init__(self, name, mass, inertia, default_translation=None, default_orientation=None):
        self.name = name
        self.mass = mass
        self.inertia = inertia  # 3x3 tensor
        if default_translation is None:
            default_translation = torch.zeros(3)
        if default_orientation is None:
            default_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.default_translation = default_translation
        self.default_orientation = default_orientation


class GeometryObject:
    """
    Geometry object defined by its name, type, and pose relative to its body.
    The pose is represented by:
      - default_translation: a 3-vector (default zero)
      - default_orientation: a quaternion (default identity)
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