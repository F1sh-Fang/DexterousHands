from matplotlib.pyplot import axis
import numpy as np
import os
import random
from dexhand.utils.torch_jit_utils import *
import torch

def control_ik(self,action):
    '''
        input desired ee pose, calculate Joint action
        args: (6DoF action (num_envs,6))action
    '''
    max_linear_velocity = 0.2   # 米/秒
    max_angular_velocity = np.pi / 4 # 弧度/秒
    desired_lin_vel = action[:, 0:3] * max_linear_velocity
    desired_ang_vel = action[:, 3:6] * max_angular_velocity
    self.ee_target_pose[:,:3] += desired_lin_vel * self.dt

    delta_rot_vector = desired_ang_vel * self.dt
    print("delta_rot_vector",delta_rot_vector[0,:])
    delta_quat = quat_from_rotvec(delta_rot_vector)
    print("delta_quat",delta_quat[0,:])
    self.ee_target_pose[:,3:7] = quat_mul(delta_quat, self.ee_target_pose[:,3:7])
    self.ee_target_pose[:,3:7] = torch.nn.functional.normalize(self.ee_target_pose[:,3:7], p=2, dim=-1)
    print("ee_target_pose",self.ee_target_pose[0,:])
    print("ee_pose",self.ee_pose[0,:])

    pos_err = self.ee_target_pose[:,:3] - self.ee_pos
    orn_err = orientation_error(self.ee_target_pose[:,3:7], self.ee_rot)

    dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)
    dtheta = self.cal_ik(dpose)
    

    return dtheta

@torch.jit.script
def quat_from_rotvec(rot_vec):
    # type: (torch.Tensor) -> torch.Tensor
    """
    将一个旋转向量 (角速度 * dt) 转换为一个增量四元数。
    这个函数利用了已有的 quat_from_angle_axis。

    Args:
        rot_vec (Tensor): 形状为 (N, 3) 的旋转向量张量。
    
    Returns:
        Tensor: 形状为 (N, 4) 的四元数张量。
    """
    # 旋转向量的模长(norm)就是旋转角度
    angle = torch.norm(rot_vec, p=2, dim=-1)
    
    # 旋转向量的方向就是旋转轴
    # 我们需要处理角度为0的特殊情况，以避免除以零
    axis = rot_vec / (angle.unsqueeze(-1) + 1e-8)
    
    # 调用已有的函数来完成转换
    return quat_from_angle_axis(angle, axis)

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)