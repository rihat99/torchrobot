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
    next_a = ForwardDynamics(model, data, q, v, tau, f_ext)

    next_q = torch.zeros_like(q, device=q.device)   
    next_v = torch.zeros_like(v, device=v.device)
    
    config = data.split_config(q, v, next_a)

    idx_v = 0
    idx_q = 0

    for joint_id in range(model.njoints):
        joint = model.joints[joint_id]
        config_joint = config[joint_id]

        q_next_joint, v_next_joint = joint.integrate(
            config_joint['q'], config_joint['v'], config_joint['a'], dt)
        
        next_v[:, idx_v:idx_v + joint.nv] = v_next_joint
        next_q[:, idx_q:idx_q + joint.nq] = q_next_joint

        idx_v += joint.nv
        idx_q += joint.nq

    return next_q, next_v, next_a

    