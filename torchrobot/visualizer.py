import torch
import numpy as np

from .model import RobotModel
from .robot_data import RobotData
from .kinematics import ForwardKinematics

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


class Visualizer:
    def __init__(self, robot_model:RobotModel, data:RobotData, meshcat_vis=None,):
        if meshcat_vis is None:
            self.vis = meshcat.Visualizer()
            self.vis.url()
        else:
            self.vis = meshcat_vis

        self.robot_model = robot_model
        self.data = data
        self.vis.delete()


        for i, geometry_object in enumerate(self.robot_model.geometry_objects):
            self.vis['robot'][geometry_object.name].set_object(
                geometry_object.shape, 
                geometry_object.color,

            )

    def set_configuration(self, q):

        # q[2] -= 5.0

        ForwardKinematics(self.robot_model, self.data, q)
        joint_trans = self.data.joint_transforms

        joint_config = self.data.split_config(q)

        for i, geometry_object in enumerate(self.robot_model.geometry_objects):
            if geometry_object.parent_id is not None:
                parent_transform = joint_trans[geometry_object.parent_id]
                
                T = parent_transform @ geometry_object.offset
                T = T.detach().cpu().numpy().T.reshape(-1)
                T = T.astype(np.float64)

                self.vis['robot'][geometry_object.name].set_transform(T)
    


    