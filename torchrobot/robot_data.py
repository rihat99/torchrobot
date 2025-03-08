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
            
        self.v_ = self.interpolate_data(config, dt)
        self.a_ = self.interpolate_data(self.v_, dt)
            
        return self.q_, self.v_, self.a_
    
    @staticmethod
    def interpolate_data(data, dt):
        begin = (data[1] - data[0]) / dt
        end = (data[-1] - data[-2]) / dt
        mid = (data[2:] - data[:-2]) / (2 * dt)

        return torch.cat([begin.unsqueeze(0), mid, end.unsqueeze(0)], dim=0)

        
        

            
