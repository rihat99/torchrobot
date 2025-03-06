import torch
import numpy as np

from torchrobot.model import RobotModel
from torchrobot.kinematics import ForwardKinematics

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


class Visualizer:
    def __init__(self, robot_model:RobotModel, meshcat_vis=None,):
        if meshcat_vis is None:
            self.vis = meshcat.Visualizer()
            self.vis.url()
        else:
            self.vis = meshcat_vis

        self.robot_model = robot_model
        self.vis.delete()


        for i, geometry_object in enumerate(self.robot_model.geometry_objects):
            self.vis['robot'][geometry_object.name].set_object(
                geometry_object.shape, 
                geometry_object.color,

            )

    def set_configuration(self, q):

        joint_trans, _ = ForwardKinematics(self.robot_model, q)

        joint_config = self.robot_model.split_config(q)

        for i, geometry_object in enumerate(self.robot_model.geometry_objects):
            if geometry_object.parent_id is not None:
                parent_transform = joint_trans[geometry_object.parent_id]
                
                if self.robot_model.joints[geometry_object.parent_id].joint_type == 'freeflyer':
                    joint_motion = torch.eye(4, device=q.device)
                else:
                    joint_motion = self.robot_model.joints[geometry_object.parent_id].forward_kinematics(joint_config[geometry_object.parent_id])
                # joint_motion[:3, 3] = 0.0
                T = parent_transform @ joint_motion @ geometry_object.offset
                T = T.detach().cpu().numpy().T.reshape(-1)
                T = T.astype(np.float64)

                self.vis['robot'][geometry_object.name].set_transform(T)
    


    