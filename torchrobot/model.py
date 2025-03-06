import torch
from .joint import SphericalJoint, FreeFlyerJoint, FixedJoint
from .body import RigidBody, GeometryObject
from .robot_data import RobotData
from .utils import homogeneous_transform

class RobotModel:
    """
    Robot model that holds the static structure (joints, bodies, tree).
    """
    def __init__(self, device='cpu'):
        self.device = device

        self.joints = []  # list of Joint instances
        self.bodies = []  # list of RigidBody instances
        self.geometry_objects = []  # list of GeometryObject instances
        self.tree = {}    # dictionary: key = joint id, value = list of child joint ids

        self.njoints = 0
        self.nq = 0
        self.nv = 0

    def addJoint(self, name, parent_id=None, joint_type='spherical', joint_offset=None):
        """
        Adds a joint to the robot model.
        
        Parameters:
          - name: joint identifier.
          - parent: parent joint id (None if root).
          - joint_type: 'spherical' or 'freeflyer'.
          - joint_offset: fixed 4x4 transform from parent to joint frame (default identity).
          - default_translation: default joint translation (only for freeflyer).
          - default_q: default joint orientation (quaternion).
          
        Returns:
          - joint_id: index of the newly added joint.
        """
        if joint_offset is None:
            joint_offset = torch.eye(4)
        
        if joint_type == 'spherical':
            joint = SphericalJoint(name, joint_offset, parent_id, device=self.device)
        elif joint_type == 'freeflyer':
            joint = FreeFlyerJoint(name, joint_offset, parent_id, device=self.device)
        elif joint_type == 'fixed':
            joint = FixedJoint(name, joint_offset, parent_id, device=self.device)
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")
        
        joint_id = len(self.joints)
        self.joints.append(joint)
        
        # Update the kinematic tree.
        if parent_id is None:
            self.tree[joint_id] = []
        else:
            if parent_id not in self.tree:
                self.tree[parent_id] = []
            self.tree[parent_id].append(joint_id)
            self.tree[joint_id] = []

        self.njoints += 1
        self.nq += joint.nq
        self.nv += joint.nv
        
        return joint_id

    # def add_body(self, joint_id, mass, inertia, name=None, default_translation=None, default_orientation=None):
    #     """
    #     Adds a body to the model and attaches it to the joint with index joint_id.
        
    #     Parameters:
    #       - joint_id: index of the joint to attach the body.
    #       - mass: scalar mass.
    #       - inertia: 3x3 inertia tensor.
    #       - name: optional body identifier.
    #       - default_translation: body pose translation relative to the joint.
    #       - default_orientation: body pose orientation (quaternion) relative to the joint.
          
    #     Returns:
    #       - body_id: index of the newly added body.
    #     """
    #     if name is None:
    #         name = f"body_{len(self.bodies)}"
    #     body = RigidBody(name, mass, inertia,
    #                      default_translation=default_translation,
    #                      default_orientation=default_orientation)
    #     self.bodies.append(body)
    #     # Attach the body to the specified joint.
    #     self.joints[joint_id].body = body
    #     return len(self.bodies) - 1


    def addGeometryObject(self, name, parent_id, shape, body_offset, color=None):
        """
        Adds a geometry object to the body with index body_id.
        
        Parameters:
          - body_id: index of the body to attach the geometry object.
          - shape: geometry shape (e.g., 'box', 'sphere', 'cylinder').
          - offset: geometry pose relative to the body.
          - name: optional geometry identifier.
          
        Returns:
          - geometry_id: index of the newly added geometry object.
        """

        geometry = GeometryObject(name, shape, body_offset, parent_id, color, device=self.device)
        self.geometry_objects.append(geometry)
        return len(self.geometry_objects) - 1


    def split_config(self, q):
        """
        Splits a joint configuration tensor into individual joint configurations.
        
        Parameters:
          - q: tensor of shape (nq,) or (batch, nq).
          
        Returns:
          - joint_configs: list of joint configurations.
        """
        
        joint_config = []
        idx = 0
        for joint in self.joints:
            if joint.nq == 0:
                joint_config.append(None)
            else:
                joint_config.append(q[..., idx:idx + joint.nq])
                idx += joint.nq
        return joint_config
    
    def neeutal_pose(self):
        """
        Returns the neutral pose of the robot model.
        """

        joint_config = []

        for joint in self.joints:
                joint_config.append(joint.default_q)

        q = torch.cat(joint_config, dim=0)
        return q

    # def create_data(self):
    #     """
    #     Creates a RobotData instance with default joint configurations.
    #     """
    #     data = RobotData()
    #     for idx, joint in enumerate(self.joints):
    #         if joint.joint_type == 'spherical':
    #             # Clone default configuration and enable gradients.
    #             data.joint_values[idx] = {'q': joint.default_q.clone().detach().requires_grad_()}
    #         elif joint.joint_type == 'freeflyer':
    #             data.joint_values[idx] = {
    #                 'translation': joint.default_translation.clone().detach().requires_grad_(),
    #                 'q': joint.default_q.clone().detach().requires_grad_()
    #             }
    #     return data