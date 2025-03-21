import torch

from .dynamics import RNEA, CRBA
from .utils import *


def ForwardDynamics(model, data, q, v, tau, f_ext=None):
    """
    Computes forward dynamics using the Composite Rigid Body Algorithm.
    """
    # if f_ext is not None:
    #     assert f_ext.size(1) == model.n_joints, "External forces must have the same number of joints as the model"
    
    # Compute the inverse dynamics
    tau_0 = RNEA(model, data, q, v, None, f_ext)
    
    # Compute the composite rigid body algorithm
    M = CRBA(model, data, q)
    
    # Compute the acceleration
    a = torch.linalg.solve(M, tau - tau_0)
    
    return a

def SimulationStep(model, data, q, v, tau, dt, f_ext=None):
    """
    Simulates a single time step using the forward dynamics.
    """
    a = ForwardDynamics(model, data, q, v, tau, f_ext)
    
    # Integrate the acceleration to get the velocity
    new_v = v + a * dt

    delta_q = v * dt + 0.5 * a * dt**2

    q_axis_angle = [q[:, :3]]
    for i in range(24):
        q_axis_angle.append(quaternion_to_axis_angle(q[:, 3 + i*4: 3 + (i+1)*4]))

    q_axis_angle = torch.cat(q_axis_angle, dim=-1)

    q_axis_angle = q_axis_angle + delta_q

    new_q = [q_axis_angle[:, :3]]
    for i in range(24):
        new_q.append(axis_angle_to_quaternion(q_axis_angle[:, 3 + i*3: 3 + (i+1)*3]))

    new_q = torch.cat(new_q, dim=-1)

    return new_q, new_v, a

    