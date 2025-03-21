import torch

from .utils import *


class RobotData:
    """
    Data class for storing the dynamic state of the robot.
    
    Attributes:
      - joint_values: dictionary mapping joint id to its configuration.
          For spherical joints: {'q': tensor(4)}
          For SE3 joints: {'translation': tensor(3), 'q': tensor(4)}
      - joint_transforms: dictionary mapping joint id to computed 4x4 transforms.
      - body_transforms: dictionary mapping body name to computed 4x4 transforms.
    """
    def __init__(self, nq: int, nv: int, joint_info: list, device='cpu'):
        self.device = device

        self.q_ = None
        self.v_ = None
        self.a_ = None

        self.nq = nq
        self.nv = nv
        self.joints_info = joint_info

        self.joint_transforms = {i: None for i in range(len(joint_info))}
        self.joint_velocities = {i: None for i in range(len(joint_info))}
        self.joint_accelerations = {i: None for i in range(len(joint_info))}
        self.body_transforms = {}

        self.com = torch.zeros(3, device=self.device)
        self.vcom = torch.zeros(6, device=self.device)
        self.acom = torch.zeros(6, device=self.device)
        self.body_com = {}
        self.total_mass = 0.0

      
    def split_config(self, q: torch.Tensor, v: torch.Tensor=None, a: torch.Tensor=None):
        """
        Splits a joint configuration tensor into individual joint configurations.
        
        Parameters:
          - q: tensor of shape (nq,) or (batch, nq).
          
        Returns:
          - joint_configs: list of joint configurations.
        """
        
        joint_config = []
        idx_q = 0
        idx_v = 0
        idx_a = 0
        for joint in self.joints_info:
            joint_config.append({'q': None, 'v': None, 'a': None})
            if joint['nq'] == 0:
                continue
                
            joint_config[-1]['q'] = q[..., idx_q: idx_q + joint['nq']]
            idx_q += joint['nq']

            if v is not None:
                joint_config[-1]['v'] = v[..., idx_v: idx_v + joint['nv']]
                idx_v += joint['nv']

            if a is not None:
                joint_config[-1]['a'] = a[..., idx_a: idx_a + joint['nv']]
                idx_a += joint['nv']
    
                
        return joint_config
    
    
    def load_from_axis_angle(self, config: torch.Tensor, dt: float=1/30):
        """
        Loads joint configuration from axis-angle representation.
        
        Parameters:
          - config: tensor of shape (batch, nq).
        """

        self.q_ = torch.zeros((config.shape[0], self.nq), device=self.device)
        idx_input = 0
        idx_q = 0
        for joint in self.joints_info:
            if joint['nq'] == 0:
                continue
            if joint['joint_type'] == 'freeflyer':
                # copy translation part
                self.q_[:, idx_q:idx_q + 3] = config[:, idx_input:idx_input + 3]
                idx_input += 3
                idx_q += 3

                # convert axis-angle to quaternion
                axis_angle = config[:, idx_input:idx_input + 3]
                q = axis_angle_to_quaternion(axis_angle)
                self.q_[:, idx_q:idx_q + 4] = q
                idx_input += 3
                idx_q += 4

            elif joint['joint_type'] == 'spherical':
                # convert axis-angle to quaternion
                axis_angle = config[:, idx_input:idx_input + 3]
                q = axis_angle_to_quaternion(axis_angle)
                self.q_[:, idx_q:idx_q + 4] = q
                idx_input += 3
                idx_q += 4

            else:
                raise ValueError(f"Unsupported joint type: {joint['type']}")
            
        # v_ = self.interpolate_data(config, dt)
        # a_ = self.interpolate_data(v_, dt)

        # base_rotation = self.q_[:, 3:7]
        # base_rotation = quaternion_to_rotation_matrix(base_rotation).transpose(-1, -2)
        
        # self.v_[:, 0:3] = RobotData.rotate_axis_angle(self.v_[:, 0:3], base_rotation)
        # self.v_[:, 3:6] = RobotData.rotate_axis_angle(self.v_[:, 3:6], base_rotation)
        # self.a_[:, 0:3] = RobotData.rotate_axis_angle(self.a_[:, 0:3], base_rotation)
        # self.a_[:, 3:6] = RobotData.rotate_axis_angle(self.a_[:, 3:6], base_rotation)

        # self.v_[:, :3] *= -1
        # self.a_[:, :3] *= -1

        # offset = model.joints[0].offset
        # base_transform = homogeneous_transform(self.q_[:, :3], self.q_[:, 3:7])
        # A = adjoint_transform(offset @ base_transform)
        # base_v = (A @ v_[:, :6].unsqueeze(-1)).squeeze(-1)
        # base_a = (A @ a_[:, :6].unsqueeze(-1)).squeeze(-1)

        # self.v_ = torch.cat([base_v, v_[:, 6:]], dim=-1)
        # self.a_ = torch.cat([base_a, a_[:, 6:]], dim=-1)

        # self.v_ = v_
        # self.a_ = a_
        

        # self.q_ = self.q_[2:]
        # self.v_ = self.v_[1:]
            
        return self.q_

    
    @staticmethod
    def interpolate_data(data, dt):
        begin = (data[1] - data[0]) / dt
        end = (data[-1] - data[-2]) / dt
        mid = (data[2:] - data[:-2]) / (2 * dt)

        return torch.cat([begin.unsqueeze(0), mid, end.unsqueeze(0)], dim=0)

        # return (data[1:] - data[:-1]) / dt


    @staticmethod
    def rotate_axis_angle(axis_angle, rotation):
        """
        Rotates an axis-angle vector by a given rotation matrix.
        
        Parameters:
          - axis_angle: tensor of shape (3,) representing the axis-angle vector.
          - rotation: tensor of shape (3, 3) representing the rotation matrix.
          
        Returns:
          - rotated_axis_angle: tensor of shape (3,) representing the rotated axis-angle vector.
        """
        
        rot_mat = axis_angle_to_matrix(axis_angle)

        new_rot = rotation @ rot_mat
        return matrix_to_axis_angle(new_rot)

        
        

            
