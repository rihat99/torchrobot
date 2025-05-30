import os
import numpy as np
import pickle

import torch
from .torchrobot.model import RobotModel
from .torchrobot.utils import homogeneous_transform, align_box_to_vector

import meshcat.geometry as g

from smplx import SMPL


joint_info = [ 
    {'name':      'pelvis', 'parent':         None, 'type': 0},     #  0
    {'name':       'l_hip', 'parent':     'pelvis', 'type': 1},     #  1
    {'name':       'r_hip', 'parent':     'pelvis', 'type': 1},     #  2
    {'name':     'spine_0', 'parent':     'pelvis', 'type': 1},     #  3
    {'name':      'l_knee', 'parent':      'l_hip', 'type': 1},     #  4
    {'name':      'r_knee', 'parent':      'r_hip', 'type': 1},     #  5
    {'name':     'spine_1', 'parent':    'spine_0', 'type': 1},     #  6
    {'name':     'l_ankle', 'parent':     'l_knee', 'type': 1},     #  7
    {'name':     'r_ankle', 'parent':     'r_knee', 'type': 1},     #  8
    {'name':     'spine_2', 'parent':    'spine_1', 'type': 1},     #  9
    {'name':      'l_toes', 'parent':    'l_ankle', 'type': 1},     # 10
    {'name':      'r_toes', 'parent':    'r_ankle', 'type': 1},     # 11
    {'name':     'spine_3', 'parent':    'spine_2', 'type': 1},     # 12
    {'name':   'l_scapula', 'parent':    'spine_2', 'type': 1},     # 13
    {'name':   'r_scapula', 'parent':    'spine_2', 'type': 1},     # 14
    {'name':     'spine_4', 'parent':    'spine_3', 'type': 1},     # 15
    {'name':  'l_shoulder', 'parent':  'l_scapula', 'type': 1},     # 16
    {'name':  'r_shoulder', 'parent':  'r_scapula', 'type': 1},     # 17
    {'name':     'l_elbow', 'parent': 'l_shoulder', 'type': 1},     # 18
    {'name':     'r_elbow', 'parent': 'r_shoulder', 'type': 1},     # 19
    {'name':     'l_wrist', 'parent':    'l_elbow', 'type': 1},     # 20
    {'name':     'r_wrist', 'parent':    'r_elbow', 'type': 1},     # 21
    {'name':   'l_fingers', 'parent':    'l_wrist', 'type': 1},     # 22
    {'name':   'r_fingers', 'parent':    'r_wrist', 'type': 1},     # 23

    # {'name':        'nose', 'parent':    'spine_4', 'type': 2},     # 24
    # {'name':       'r_eye', 'parent':       'nose', 'type': 2},     # 25
    # {'name':       'l_eye', 'parent':       'nose', 'type': 2},     # 26
    # {'name':       'r_ear', 'parent':      'r_eye', 'type': 2},     # 27
    # {'name':       'l_ear', 'parent':      'l_eye', 'type': 2},     # 28
    # {'name':   'l_big_toe', 'parent':     'l_toes', 'type': 2},     # 29
    # {'name': 'l_small_toe', 'parent':     'l_toes', 'type': 2},     # 30
    # {'name':      'l_heel', 'parent':    'l_ankle', 'type': 2},     # 31
    # {'name':   'r_big_toe', 'parent':     'r_toes', 'type': 2},     # 32
    # {'name': 'r_small_toe', 'parent':     'r_toes', 'type': 2},     # 33
    # {'name':      'r_heel', 'parent':    'r_ankle', 'type': 2},     # 34
    # {'name':  'l_finder_1', 'parent':    'l_wrist', 'type': 2},     # 35
    # {'name':  'l_finder_2', 'parent':  'l_fingers', 'type': 2},     # 36
    # {'name':  'l_finder_3', 'parent':  'l_fingers', 'type': 2},     # 37
    # {'name':  'l_finder_4', 'parent':  'l_fingers', 'type': 2},     # 38
    # {'name':  'l_finder_5', 'parent':  'l_fingers', 'type': 2},     # 39
    # {'name':  'r_finder_1', 'parent':    'r_wrist', 'type': 2},     # 40
    # {'name':  'r_finder_2', 'parent':  'r_fingers', 'type': 2},     # 41
    # {'name':  'r_finder_3', 'parent':  'r_fingers', 'type': 2},     # 42
    # {'name':  'r_finder_4', 'parent':  'r_fingers', 'type': 2},     # 43
    # {'name':  'r_finder_5', 'parent':  'r_fingers', 'type': 2},     # 44
]

def loadInertias(pathToInertia, mass_scale=1.):
    with open(pathToInertia, 'rb') as inputf:
        inertias_temp = pickle.load(inputf, encoding='latin-1')
        inertias = {}
        for k in inertias_temp.keys():
            # inertias[self.name+'_'+k] = inertias_temp[k]
            mass = inertias_temp[k][0] * mass_scale
            I = inertias_temp[k][1] * mass_scale
            inertias[k] = (mass, I)

        # print("inertias loaded from "+pathToInertia)
        return inertias

def create_robot(smpl, shape=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # smpl = SMPL(model_path="../data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl").to(device)

    pose = torch.zeros(1, 72, requires_grad=False).to(device) # Pose parameters
    if shape is None:
        shape = torch.zeros(1, 10, requires_grad=False).to(device)  # Shape parameters
    else:
        shape = torch.tensor(shape, requires_grad=False).to(device).unsqueeze(0)
    translation = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=False).to(device) # Translation parameters

    output = smpl.forward(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=shape, transl=translation)

    neutral_pose = output.joints.squeeze()

    # if inertias is None:
    #     if inertia_path is None:
    #         raise ValueError("Either inertias or inertia_path must be provided.")
    inertias = loadInertias("data/smpl/full_body_inertia.pkl")

    joint_name_to_id = {joint["name"]: i for i, joint in enumerate(joint_info)}
    joint_children = {joint["name"]: [] for joint in joint_info}
    for joint in joint_info:
        if joint["parent"] is not None:
            joint_children[joint["parent"]].append(joint["name"])

    joint_types = {
        0: "freeflyer",
        1: "spherical",
        2: "fixed",
    }
    joint_colors = {
        0: np.array([250,   0,   0, 1]),
        1: np.array([  0, 250,   0, 1]),
        2: np.array([250,   0,   0, 1]),
    }
    joint_ids = {}

    body_radius = 0.01

    model = RobotModel(device=device)
    model.gravity[:3] = torch.tensor([0., 9.81, 0.], device=device)
    model.end_effectors_ids = [10, 11, 15, 22, 23, 9]
    model.connections_ids = [9, 0]

    # Create joints
    for i, joint in enumerate(joint_info):

        joint_name = joint["name"]
        joint_id = joint_name_to_id[joint_name]
        parent_name = joint["parent"]
        parent_id = joint_name_to_id[parent_name] if parent_name is not None else None

        joint_type = joint['type']
        # if joint_type == 2:
        #     continue

        if joint_name == "pelvis":
            base_trans = neutral_pose[joint_id]
            joint_offset = homogeneous_transform(base_trans, torch.tensor([1., 0., 0., 0.], device=device))
            # joint_offset = torch.eye(4, device=device)
        else:
            joint_trans = neutral_pose[joint_id] - neutral_pose[parent_id]
            joint_offset = homogeneous_transform(joint_trans, torch.tensor([1., 0., 0., 0.], device=device))

        # Add joint to the model
        joint_model = joint_types[joint['type']]
        joint_id = model.addJoint(joint_name, parent_id, joint_model, joint_offset)

        joint_ids[joint_name] = joint_id

    # Add bodies to the joints
    for joint in joint_info:
        joint_name = joint["name"]
        joint_id = joint_name_to_id[joint_name]
        parent_name = joint["parent"]
        parent_id = joint_name_to_id[parent_name] if parent_name is not None else None

        if joint_id > 23:
            continue

        if joint_name == "pelvis":
            com = torch.tensor([0., 0., 0.], device=device)
        elif len(joint_children[joint_name]) > 0 and joint_ids[joint_children[joint_name][0]] < 24:
            child_name = joint_children[joint_name][0]
            child_id = joint_name_to_id[child_name]
            com = (neutral_pose[child_id] - neutral_pose[joint_id]) / 2
        else:
            com = (neutral_pose[joint_id] - neutral_pose[parent_id]) / 2.4


        mass = torch.tensor(inertias[joint_name + '_link'][0], device=device)
        inertia_matrix = torch.tensor(inertias[joint_name + '_link'][1], device=device)

        # mass = torch.tensor(1.0, device=device, dtype=torch.float32)
        # inertia_matrix = torch.eye(3, device=device, dtype=torch.float32)
        # com = torch.tensor([0., 0., 0.], device=device, dtype=torch.float32)

        model.addBody(joint_name, joint_id, mass, com, inertia_matrix)


    # Add geometry objects to joints
    for joint in joint_info:
        joint_name = joint["name"]
        joint_type = joint['type']
        joint_id = joint_name_to_id[joint_name]

        shape = g.Sphere(body_radius * 1.5)
        shape_placement = torch.eye(4, device=device)

        model.addGeometryObject(
            name=f"{joint_name}_sphere", 
            parent_id=joint_id,
            shape=shape,
            body_offset=shape_placement,
            color=joint_colors[joint_type],
        )

        for child_name in joint_children[joint_name]:
            child_id = joint_name_to_id[child_name]

            joint_trans = neutral_pose[child_id] - neutral_pose[joint_id]
            target_center = joint_trans / 2
        
            distance = float(torch.norm(joint_trans, p=2).detach().cpu())
            shape = g.Cylinder(distance, body_radius)

            # Align the box with the joint direction
            shape_center = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
            
            R_matrix, T_vector = align_box_to_vector(joint_trans, shape_center, target_center)
            shape_placement = torch.eye(4, device=device)
            shape_placement[:3, :3] = R_matrix
            shape_placement[:3, 3] = T_vector

            model.addGeometryObject(
                name=f"{joint_name}_{child_name}_link", 
                parent_id=joint_id,
                shape=shape,
                body_offset=shape_placement,
                color=np.array([250, 250, 250, 1]),
            )

    return model