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
        # self.bodies = []  # list of RigidBody instances
        self.geometry_objects = []  # list of GeometryObject instances
        self.tree = {}    # dictionary: key = joint id, value = list of child joint ids

        self.njoints = 0
        self.nq = 0
        self.nv = 0

        self.joint_info = []

        self.gravity = torch.zeros(6, device=device)
        # Set gravity along the z-axis.
        self.gravity[5] = -9.81

        self.end_effectors_ids = []
        self.connections_ids = []

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

        self.joint_info.append({
            'name': name,
            'nq': joint.nq,
            'nv': joint.nv,
            'joint_type': joint_type
        })
        
        return joint_id

    def addBody(self, name, parent_joint_id, mass, com, inertia, offset=None):
        """
        Adds a body to the model and attaches it to the joint with index joint_id.
        
        Parameters:
            - name: body identifier.
            - joint_id: index of the joint to attach the body.
            - mass: body mass.
            - com: body center of mass.
            - inertia: body inertia tensor.
            - offset: body pose relative to the joint.
          
        Returns:
          - body_id: index of the newly added body.
        """

        if self.joints[parent_joint_id].body is not None:
            raise ValueError("Each joint can only have one body.")

        body = RigidBody(name, parent_joint_id, mass, com, inertia, offset, device=self.device)
        self.joints[parent_joint_id].body = body
        # self.bodies.append(body)
        # return len(self.bodies) - 1
        

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

    def neutral_pose(self):
        """
        Returns the neutral pose of the robot model.
        """

        joint_config = []

        for joint in self.joints:
                joint_config.append(joint.default_q)

        q = torch.cat(joint_config, dim=0)
        return q

    def create_data(self):
        """
        Creates a RobotData instance with default joint configurations.
        """
        data = RobotData(self.nq, self.nv, self.joint_info, self.device)
        
        return data