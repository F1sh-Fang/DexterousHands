# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import os
import random

from dexhand.utils.torch_jit_utils import *
from dexhand.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import open3d as o3d
import trimesh
import gc

class BotyardHandPick(BaseTask):
    @torch.no_grad()
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]

        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 1#0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.botyard_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)
        
        # num of obs 
        self.num_robot_obs = 181
        # action = arm 7 + hand 23 
        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": self.num_robot_obs + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": self.num_robot_obs + self.num_point_cloud_feature_dim * 3,
            "full_state": self.num_robot_obs
        }
        # self.num_hand_obs = 72 + 95 + 22
        self.num_hand_obs = 118
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        ## action space
        ## 6 dof ee ; FAJ3 JAJ1 THJ4321 FFJ432 MFJ432 RFJ432 LFJ432 ; 24
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 22
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 24

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        #     dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        #     self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_allegro_hand_dofs * 2)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.botyard_hand_default_dof_pos = torch.zeros(self.num_botyard_hand_dofs, dtype=torch.float, device=self.device)
        self.botyard_hand_default_dof_pos[:7] = torch.tensor([0, -1.3, 0, -2.4, 0, 2.66, 0], dtype=torch.float, device=self.device)
        self.ee_default_target_pose = torch.zeros(7, dtype=torch.float, device=self.device)
        self.ee_default_target_pose = torch.tensor([0.365, 0, 0.809, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.ee_pos_lower_limits = torch.tensor([0.1, -0.7, 0.62], dtype=torch.float, device=self.device)
        self.ee_pos_upper_limits = torch.tensor([0.85, 0.7, 1.2], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.botyard_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_botyard_hand_dofs]
        self.botyard_hand_dof_pos = self.botyard_hand_dof_state[..., 0]
        self.botyard_hand_dof_vel = self.botyard_hand_dof_state[..., 1]
        print("dof_state",self.dof_state.shape)
        print("botyard_hand_dof_state",self.botyard_hand_dof_state.shape)
        print("botyard_hand_dof_pos",self.botyard_hand_dof_pos.shape)
        
        self.rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor)
        self.rigid_body_states = self.rigid_body_tensor.view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        print("rigid_body_states",self.rigid_body_states.shape,self.num_bodies)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        
        ########################### observation ####################################
        self.botyard_right_hand_pos = self.rigid_body_states[:, 7, 0:3]
        self.botyard_right_hand_rot = self.rigid_body_states[:, 7, 3:7]

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pos = self.object_state[:, 0:3]
        self.object_rot =  self.object_state[:, 3:7]
        self.object_linvel = self.object_state[:, 7:10]
        self.object_angvel = self.object_state[:, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles, 0:13]
        # self.fingertip_pos   = self.rigid_body_states[:, self.fingertip_handles, 0: 3]
        # print(self.fingertip_pos)
        self.fingertip_state = self.rigid_body_tensor[self.fingertip_indices, 0:13]
        self.fingertip_pos   = self.rigid_body_tensor[self.fingertip_indices, 0: 3]
        # print(self.fingertip_pos)
    
        self.ee_pose   = self.rigid_body_states[:, self.ee_handle, 0:7]
        self.ee_pos    = self.rigid_body_states[:, self.ee_handle, 0:3]
        self.ee_rot    = self.rigid_body_states[:, self.ee_handle, 3:7]
        self.ee_linvel = self.rigid_body_states[:, self.ee_handle, 7:10]
        self.ee_angvel = self.rigid_body_states[:, self.ee_handle, 10:13]
        
        self.ee_target_pose = torch.zeros((self.num_envs, 7),dtype=torch.float32, device=self.device)

        self.postive_distance = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self.negative_distance = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self.finger_mid_pos = (self.fingertip_pos[:, -1, 0:3] + self.fingertip_pos[:, -2, 0:3]) / 2.0
        self.finger_mid_vec = self.finger_mid_pos - self.object_pos
        self.finger_mid_dis = torch.norm(self.finger_mid_vec, dim=-1, keepdim=True)
        self.table_distance = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self.postive_distance_mod = torch.mean(self.postive_distance, dim=-1, keepdim=True) 

        self.fingertip_distance = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.fingertip_distance[:,0] = self.get_surface_distance(self.fingertips[-1],self.fingertips[-2])
        self.fingertip_distance[:,1] = self.get_surface_distance(self.fingertips[-1],self.fingertips[-3])
        self.fingertip_distance = torch.mean(self.fingertip_distance, dim=-1, keepdim=True)
        ########################### observation ####################################

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.free_cuda_cache_threshold = 100 * 4
        self.free_cuda_cache_count = 0
        self.total_successes = 0
        self.total_resets = 0
        self.point_cloud_debug = True

    @torch.no_grad()
    def print_dof_info(self, asset):
        """
        打印给定asset的所有DOF的名字和索引。

        Args:
            asset (gymapi.Asset): 从中获取信息的Isaac Gym asset。
        """
        # 获取DOF的总数
        num_dofs = self.gym.get_asset_dof_count(asset)
        print(f"--- DOF Information for Asset ---")
        print(f"Total number of DOFs: {num_dofs}")

        # 获取DOF名字的列表
        dof_names = self.gym.get_asset_dof_names(asset)
        
        # 获取DOF属性的字典
        dof_dict = self.gym.get_asset_dof_dict(asset)

        print("\n--- Mapping by Index ---> Name ---")
        # 遍历索引，打印名字
        for i in range(num_dofs):
            print(f"  Index {i}: {dof_names[i]}")

        print("\n--- Mapping by Name ---> Index ---")
        # 遍历字典，打印名字和索引
        for name, idx in dof_dict.items():
            print(f"  Name '{name}': Index {idx}")
        
        print("----------------------------------\n")

    @torch.no_grad()
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    @torch.no_grad()
    def get_surface_distance(self, name_a: str, name_b: str):

        """
        计算两个已注册的单个刚体之间的近似表面距离。

        Args:
            name_a (str): 数据库中第一个物体的名字 (例如, "if5")。
            name_b (str): 数据库中第二个物体的名字 (例如, "object")。

        Returns:
            一个形状为 (num_envs,) 的张量，包含了所有环境中A和B之间的最短表面距离。
        """
        if "table" in name_b:
            indices_a = self.body_indices[name_a]    # 形状: (num_envs,)
            vertices_a = self.body_vertices[name_a]  # 形状: (num_samples, 3)
            poses_a = self.rigid_body_states[:, self.body_handles[name_a], :13]# 形状: (num_envs, 13)
            min_distances = compute_table_distance_jit(poses_a,vertices_a,self.table_height)
        else:
            # 1. 从数据库中查找输入数据
            indices_a = self.body_indices[name_a]    # 形状: (num_envs,)
            vertices_a = self.body_vertices[name_a]  # 形状: (num_samples, 3)
            indices_b = self.body_indices[name_b]    # 形状: (num_envs,)
            vertices_b = self.body_vertices[name_b]  # 形状: (num_samples, 3)

            # 2. 获取实时位姿
            if "object" in name_a:
                poses_a = self.root_state_tensor[indices_a, :13] # 形状: (num_envs, 13)
            else:
                poses_a = self.rigid_body_states[:, self.body_handles[name_a], :13] # 形状: (num_envs, 13)

            if "object" in name_b:
                poses_b = self.root_state_tensor[indices_b, :13]# 形状: (num_envs, 13)
            else:
                poses_b = self.rigid_body_states[:, self.body_handles[name_b], :13] # 形状: (num_envs, 13)
            min_distances = compute_surface_distance_jit(poses_a,poses_b,vertices_a,vertices_b)

        return min_distances
    
    @torch.no_grad()
    def set_hand_rigid_body_props(self, env, actor_handle, botyard_hand_asset):
        if True:#self.hand_shape_name_id_map is None:
            num_shapes     = self.gym.get_asset_rigid_shape_count(botyard_hand_asset)
            num_bodies     = self.gym.get_asset_rigid_body_count(botyard_hand_asset)
            body_names     = self.gym.get_asset_rigid_body_names(botyard_hand_asset)
            body_shape_map = self.gym.get_asset_rigid_body_shape_indices(botyard_hand_asset)
            self.hand_shape_name_id_map = {}
            for i in range(num_bodies):
                name = body_names[i]
                shape_idx_range = body_shape_map[i]
                if shape_idx_range.count > 0:
                    shape_idx = shape_idx_range.start
                    self.hand_shape_name_id_map[name] = shape_idx
                if shape_idx_range.count > 1:
                    shape_idx = range(shape_idx_range.start, shape_idx_range.start + shape_idx_range.count)
                    self.hand_shape_name_id_map[name] = shape_idx
        ## (env, index of the asset) as input
        if True:#self.hand_rigid_body_props is None:
            self.hand_rigid_body_props = self.gym.get_actor_rigid_shape_properties(env, actor_handle)
            finger_name = ['lfdistal', 'rfdistal', 'mfdistal', 'ffdistal', 'thdistal',
                        'lfmiddle', 'rfmiddle', 'mfmiddle', 'ffmiddle', 'thmiddle',
                        'lfproximal', 'rfproximal', 'mfproximal', 'ffproximal', 'thproximal']
            finger_id = [self.hand_shape_name_id_map[name] for name in finger_name]
            base_name = ['pmbase', 'palm', 'fabase']
            base_id = [self.hand_shape_name_id_map[name] for name in base_name]
            th_name = ["thbase", "thproximal", "thmiddle", "thdistal"]
            th_id = [self.hand_shape_name_id_map[name] for name in th_name]
            ff_name = ["ffbase", "ffproximal", "ffmiddle", "ffdistal"]
            ff_id = [self.hand_shape_name_id_map[name] for name in ff_name]
            mf_name = ["mfbase", "mfproximal", "mfmiddle", "mfdistal"]
            mf_id = [self.hand_shape_name_id_map[name] for name in mf_name]
            rf_name = ["rfbase", "rfproximal", "rfmiddle", "rfdistal"]
            rf_id = [self.hand_shape_name_id_map[name] for name in rf_name]
            lf_name = ["lfbase", "lfproximal", "lfmiddle", "lfdistal"]
            lf_id = [self.hand_shape_name_id_map[name] for name in lf_name]
            tip_id = [self.hand_shape_name_id_map[name] for name in self.fingertips]
            for i in range(len(self.hand_rigid_body_props)):
                if i in [base_id[0], base_id[1]]:
                    self.hand_rigid_body_props[i].filter = 0b11111
                elif i in th_id:
                    self.hand_rigid_body_props[i].filter = (1 << 0)
                elif i in ff_id:
                    self.hand_rigid_body_props[i].filter = (1 << 1)
                elif i in mf_id:
                    self.hand_rigid_body_props[i].filter = (1 << 2)
                elif i in rf_id:
                    self.hand_rigid_body_props[i].filter = (1 << 3)
                elif i in lf_id:
                    self.hand_rigid_body_props[i].filter = (1 << 4)
                else:
                    self.hand_rigid_body_props[i].filter = 0
                if i in finger_id:
                    self.hand_rigid_body_props[i].contact_offset = 0.005
                    self.hand_rigid_body_props[i].rest_offset = 0.00
                if i in tip_id:
                    self.hand_rigid_body_props[i].friction = 1.0
            # props[shape_name_id_map['lfdistal']].filter = (1 << 1)
            # props[shape_name_id_map['rfdistal']].filter = (1 << 1)

        self.gym.set_actor_rigid_shape_properties(env, actor_handle, self.hand_rigid_body_props) ### (env, index number, properties List)
    
    @torch.no_grad()
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments, spacing %f, num per row %d" % (num_envs, spacing, num_per_row))
        asset_root = "../assets"
        # allegro_hand_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"
        # allegro_hand_another_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"

        botyard_hand_asset_file = "botyard/panda_by_description/urdf/panda_by.urdf"
        # table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        # table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        object_asset_file = self.asset_files_dict[self.object_type]
        object_asset_file = 'botyard/panda_by_description/meshes/object/box_50mm.urdf'

        # load shadow hand_ asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.0001
        hand_asset_options.armature = 0.001
        # asset_options.angular_damping = 0.1
        # asset_options.linear_damping = 0.1
        

        # if self.physics_engine == gymapi.SIM_PHYSX:
        #     asset_options.use_physx_armature = True
        hand_asset_options.use_physx_armature = True
        
        print("hand")
        botyard_hand_asset = self.gym.load_asset(self.sim, asset_root, botyard_hand_asset_file, hand_asset_options)
        print("hand loaded")
        self.num_botyard_hand_bodies = self.gym.get_asset_rigid_body_count(botyard_hand_asset)
        self.num_botyard_hand_shapes = self.gym.get_asset_rigid_shape_count(botyard_hand_asset)
        self.num_botyard_hand_dofs = self.gym.get_asset_dof_count(botyard_hand_asset)
        self.num_botyard_hand_actuators = self.gym.get_asset_dof_count(botyard_hand_asset)
        self.num_botyard_hand_tendons = self.gym.get_asset_tendon_count(botyard_hand_asset)

        print("self.num_botyard_hand_bodies: ",    self.num_botyard_hand_bodies)
        print("self.num_botyard_hand_shapes: ",    self.num_botyard_hand_shapes)
        print("self.num_botyard_hand_dofs: ",      self.num_botyard_hand_dofs)
        print("self.num_botyard_hand_actuators: ", self.num_botyard_hand_actuators)
        print("self.num_botyard_hand_tendons: ",   self.num_botyard_hand_tendons)

        print("hand done")

        # tendon set up
        limit_stiffness = 3
        t_damping = 0.1
        # relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        # a_relevant_tendons = ["robot1:T_FFJ1c", "robot1:T_MFJ1c", "robot1:T_RFJ1c", "robot1:T_LFJ1c"]
        # tendon_props = self.gym.get_asset_tendon_properties(botyard_hand_asset)

        # for i in range(self.num_botyard_hand_tendons):
        #     tendon_props[i].limit_stiffness = limit_stiffness
        #     tendon_props[i].damping = t_damping

        # self.gym.set_asset_tendon_properties(botyard_hand_asset, tendon_props)
        
        self.actuated_dof_indices = [i for i in range(self.num_botyard_hand_dofs)]

        # set allegro_hand dof properties
        botyard_hand_dof_props = self.gym.get_asset_dof_properties(botyard_hand_asset)

        self.botyard_hand_dof_lower_limits = []
        self.botyard_hand_dof_upper_limits = []
        self.botyard_hand_dof_default_pos = []
        self.botyard_hand_dof_default_vel = []
        self.botyard_hand_dof_stiffness = []
        self.botyard_hand_dof_damping = []
        self.botyard_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_botyard_hand_dofs):
            self.botyard_hand_dof_lower_limits.append(botyard_hand_dof_props['lower'][i])
            self.botyard_hand_dof_upper_limits.append(botyard_hand_dof_props['upper'][i])
            self.botyard_hand_dof_default_pos.append(0.0)
            self.botyard_hand_dof_default_vel.append(0.0)

        x_arm_dof_effort = to_torch([87, 87, 87, 87, 12, 12, 12], dtype=torch.float, device=self.device)

        for i in range(0, 7):
            botyard_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            botyard_hand_dof_props['stiffness'][i] = 10000
            botyard_hand_dof_props['damping'][i] = 200
            # botyard_hand_dof_props['effort'][i] = x_arm_dof_effort[i]
            # botyard_hand_dof_props['armature'][i] = 0.01

        for i in range(7, self.num_botyard_hand_dofs):
            botyard_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            botyard_hand_dof_props['stiffness'][i] = 100
            botyard_hand_dof_props['damping'][i] = 20
            botyard_hand_dof_props['effort'][i] = 0.5
            # botyard_hand_dof_props['armature'][i] = 0.002

        # botyard_hand_dof_props["stiffness"].fill(625.0)
        # botyard_hand_dof_props["damping"].fill(50.0)
        
        self.hand_rigid_body_props = None
        self.hand_shape_name_id_map = None

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.botyard_hand_dof_lower_limits = to_torch(self.botyard_hand_dof_lower_limits, device=self.device)
        self.botyard_hand_dof_upper_limits = to_torch(self.botyard_hand_dof_upper_limits, device=self.device)
        self.botyard_hand_dof_default_pos  = to_torch(self.botyard_hand_dof_default_pos, device=self.device)
        self.botyard_hand_dof_default_vel  = to_torch(self.botyard_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 100
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        # create table asset
        table_dims = gymapi.Vec3(0.65, 1.5, 0.6)
        self.table_height = table_dims.z
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

        botyard_hand_start_pose = gymapi.Transform()
        botyard_hand_start_pose.p = gymapi.Vec3(0, 0, 0)
        botyard_hand_start_pose.p.z = 0 #table_dims.z
        botyard_hand_start_pose.p.x = -0.05
        botyard_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.65, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = table_pose.p.x
        pose_dx, pose_dy, pose_dz = 0., 0., 0.05

        object_start_pose.p.x = table_pose.p.x + pose_dx
        object_start_pose.p.y = table_pose.p.y + pose_dy
        object_start_pose.p.z = table_dims.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = botyard_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(0., 0.05, 0.3)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
    

        # compute aggregate size
        max_agg_bodies = self.num_botyard_hand_bodies * 2 + 2
        max_agg_shapes = self.num_botyard_hand_shapes * 2 + 2

        self.botyard_hands = []
        self.envs = []

        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.fingertip_indices = []

        self.fingertips = ['lfdistal', 'rfdistal', 'mfdistal', 'ffdistal', 'thdistal'] 
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(botyard_hand_asset, name) for name in self.fingertips]
        self.ee_handle = self.gym.find_asset_rigid_body_index(botyard_hand_asset, "central")
        self.dof_J1_name = ["FFJ1", "MFJ1", "RFJ1", "LFJ1"]
        self.dof_J2_name = ["FFJ2", "MFJ2", "RFJ2", "LFJ2"]
        self.dof_J1_index = [self.gym.find_asset_dof_index(botyard_hand_asset, name) for name in self.dof_J1_name]
        self.dof_J2_index = [self.gym.find_asset_dof_index(botyard_hand_asset, name) for name in self.dof_J2_name]
        self.print_dof_info(botyard_hand_asset)
        print("dof_J1_index",self.dof_J1_index,"dof_J2_index",self.dof_J2_index)

        #############################################################################################
        self.body_vertices = {}
        self.body_indices = {}
        self.body_handles = {}

        self.num_surface_samples = 512 
        fingertip_mesh_path_list = ["../assets/botyard/panda_by_description/meshes/botyard/" + name + ".STL" for name in self.fingertips]
        # object_mesh_path = "../assets/botyard/panda_by_description/meshes/object/009_gelatin_box/google_16k/nontextured.stl" 
        object_mesh_path = "../assets/botyard/panda_by_description/meshes/object/box_50mm/box.stl" 
        all_fingertips_vertices_local_list = []
        
        for i in range(len(fingertip_mesh_path_list)):
            path = fingertip_mesh_path_list[i]
            name = self.fingertips[i]
            self.body_handles[name] = self.fingertip_handles[i]
            try:
                mesh = trimesh.load(path)
                samples_np, _ = trimesh.sample.sample_surface(mesh, self.num_surface_samples)
                vertice = to_torch(samples_np, device=self.device, dtype=torch.float) # (num_samples, 3)
                self.body_vertices[name] = vertice.unsqueeze(0).expand(num_envs, -1, -1) # (num_envs, num_samples, 3)
                # print(f"  - 成功为 '{name}' ({path}) 采样 {samples_torch.shape[0]} 个点。")
            except Exception as e:
                print(f"  - 警告: 加载或采样 ({path}) 失败: {e}")
                self.body_vertices[name] = torch.zeros((self.num_surface_samples, self.num_surface_samples, 3), device=self.device)

        try:
            mesh = trimesh.load(object_mesh_path)
            samples_np, _ = trimesh.sample.sample_surface(mesh, self.num_surface_samples)
            vertice = to_torch(samples_np, device=self.device, dtype=torch.float)
            self.body_vertices["object"] = vertice.unsqueeze(0).expand(num_envs, -1, -1)
        except Exception as e:
            print(f"  - 警告: 加载或采样 ({object_mesh_path}) 失败: {e}")
            self.body_vertices["object"] = torch.zeros((self.num_surface_samples, self.num_surface_samples, 3), device=self.device)
        
        #######################################################################################################
        fingertip_local_indices = self.fingertip_handles
        object_local_idx = 0
        
        all_fingertip_indices_across_envs = []
        all_object_indices = []

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(botyard_hand_asset, ft_handle, sensor_pose)



        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            botyard_hand_actor = self.gym.create_actor(env_ptr, botyard_hand_asset, botyard_hand_start_pose, "hand", i, 0, 0)
                       
            self.hand_start_states.append([botyard_hand_start_pose.p.x, botyard_hand_start_pose.p.y, botyard_hand_start_pose.p.z,
                                           botyard_hand_start_pose.r.x, botyard_hand_start_pose.r.y, botyard_hand_start_pose.r.z, botyard_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.set_hand_rigid_body_props(env_ptr, botyard_hand_actor, botyard_hand_asset)
            self.gym.set_actor_dof_properties(env_ptr, botyard_hand_actor, botyard_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, botyard_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            fingertip_global_indices_in_env = [self.gym.get_actor_rigid_body_index(env_ptr, botyard_hand_actor, local_idx, gymapi.DOMAIN_SIM) for local_idx in fingertip_local_indices]
            self.fingertip_indices.append(fingertip_global_indices_in_env)

            for j in range(len(self.fingertips)):
                name = self.fingertips[j]
                local_idx = self.fingertip_handles[j]
                global_idx = self.gym.get_actor_rigid_body_index(env_ptr, botyard_hand_actor, local_idx, gymapi.DOMAIN_SIM)
                # 存入数据库，键是每个指尖的名字，值是形状为 (num_envs,) 的张量
                self.body_indices[name] = to_torch(global_idx, dtype=torch.long, device=self.device)
            
            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, botyard_hand_actor)
            
            # add object
            self.object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, self.object_handle, gymapi.DOMAIN_SIM)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.object_handle)
            for object_shape_prop in object_shape_props:
                object_shape_prop.friction = 0.8
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.object_handle, object_shape_props)

            self.object_indices.append(object_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            for object_shape_prop in table_shape_props:
                object_shape_prop.friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                           goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z, goal_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, self.object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.botyard_hands.append(botyard_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        print("fingertip_handles " + str(self.fingertip_handles))
        self.ee_handle = to_torch(self.ee_handle, dtype=torch.long, device=self.device)
        print("ee_handle " + str(self.ee_handle))
        self.fingertip_indices = to_torch(self.fingertip_indices, dtype=torch.long, device=self.device)
        print("fingertip_indices:" + str(self.fingertip_indices))
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.body_indices["object"] = self.object_indices
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        # print("goal_object_indices:" + str(self.goal_object_indices))
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

        print(self.body_indices.keys(), self.body_indices["object"].shape)
        print(self.body_vertices.keys())

        ############ ik #####################
        self.num_ik_arm_dof = 9
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "hand")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = self.jacobian[:, self.ee_handle - 1, :, :self.num_ik_arm_dof]
        self.hand_ik_damping = 0.1

    @torch.no_grad()
    def compute_reward(self, actions, visdebug = False):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.botyard_right_hand_pos, self.ee_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.finger_mid_dis, self.postive_distance_mod, self.fingertip_distance,
            self.ee_obj_rot_cos
        )
        
        if visdebug:
            goal_dist = torch.norm(self.goal_pos - self.object_pos, p=2, dim=-1)
            # Orientation alignment for the cube in hand and goal cube
            quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
            dist_rew = goal_dist
            a = dist_rew * self.dist_reward_scale
            reward1 = torch.exp(-(0.2*(dist_rew * self.dist_reward_scale + rot_dist)))
            reward2 = torch.exp(-10 * self.finger_mid_dis.t())
            reward3 = torch.exp(-10 * self.postive_distance_mod.t())
            reward4 = torch.exp(-10 * self.fingertip_distance.t())
            reward =  reward1 + 0.4 * reward2 + 0.5 * reward3 + 0.02 * reward4
            print("reward before:", reward)

            # Find out which envs hit the goal and update successes count
            goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(self.reset_goal_buf), self.reset_goal_buf)

            # Success bonus: orientation is within `success_tolerance` of goal orientation
            reward = torch.where(goal_resets == 1, reward + self.reach_goal_bonus, reward)

            # Fall penalty: distance to the goal is larger than a threashold
            reward = torch.where(self.object_pos[:, 2] <= 0.2, reward + self.fall_penalty, reward)
            print("reward after:", reward)
            print("dist" + str(dist_rew))
            print("dist_rew" + str(a))
            print("rot_dist" + str(rot_dist))
            print("finger_mid_dis" + str(self.finger_mid_dis))
            print("postive_distance_mod" + str(self.postive_distance_mod))
            print("fingertip_distance" + str(self.fingertip_distance))

        # print(self.rew_buf)
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    @torch.no_grad()
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.fingertip_state[:, :] = self.rigid_body_tensor[self.fingertip_indices, 0:13]
        self.fingertip_pos[:, :]   = self.rigid_body_tensor[self.fingertip_indices, 0: 3]
        self.object_state[:, :] = self.root_state_tensor[self.object_indices, 0:13]

        with torch.no_grad():
            self.finger_mid_pos[:,:] = (self.fingertip_pos[:, -1, 0:3] + self.fingertip_pos[:, -2, 0:3]) / 2.0
            self.finger_mid_vec[:,:] = self.finger_mid_pos - self.object_pos
            self.finger_mid_dis[:] = torch.norm(self.finger_mid_vec, dim=-1, keepdim=True)

            # self.postive_distance = torch.zeros((self.num_envs, 5), dtype=torch.float32, device="cuda")
            for i in range(len(self.fingertips)):
                name = self.fingertips[i]
                self.postive_distance[:, i] = self.get_surface_distance(name,"object")
            self.postive_distance_mod[:] = torch.mean(self.postive_distance, dim=-1, keepdim=True) 
            # print(self.postive_distance)
            
            # self.negative_distance = torch.zeros((self.num_envs, 5), dtype=torch.float32, device="cuda")
            # TODO: no negtive now

            # self.table_distance = torch.zeros((self.num_envs, 5), dtype=torch.float32, device="cuda")
            for i in range(len(self.fingertips)):
                name = self.fingertips[i]
                self.table_distance[:, i] = self.get_surface_distance(name,"table")
            
            ################################################## ee y ########
            ee_forward_world = quat_apply(self.ee_rot, - self.z_unit_tensor)
            hand_to_object_world = self.object_pos - self.ee_pos
            self.ee_obj_rot_cos = torch.sum(torch.nn.functional.normalize(ee_forward_world, dim=1) * 
                                    torch.nn.functional.normalize(hand_to_object_world, dim=1), dim=1)

            # self.fingertip_distance = torch.zeros((self.num_envs, 2), dtype=torch.float32, device="cuda")
            # self.fingertip_distance[:,0] = self.get_surface_distance(self.fingertips[-1],self.fingertips[-2])
            # self.fingertip_distance[:,1] = self.get_surface_distance(self.fingertips[-1],self.fingertips[-3])
            # self.fingertip_distance = torch.mean(self.fingertip_distance, dim=-1, keepdim=True)
            self.fingertip_distance = (self.get_surface_distance(self.fingertips[-1],self.fingertips[-3]) + self.get_surface_distance(self.fingertips[-1],self.fingertips[-2])) / 2.0
            self.compute_full_state()
            
            if self.asymmetric_obs:
                self.compute_full_state(True)

    @torch.no_grad()
    def compute_full_state(self, asymm_obs=False, visdebug = False):
        # num of obs 
        # arm 7 pos, 7 vel
        # hand 22 pos, 22 vel
        # hand fingertip pose, linear velocity, angle velocity 13 * 5
        # hand fingertip force-torque sensors 6 * 5
        # ee 7 pose, 3 vel, 3 ang vel
        # target 7 pose, 3 vel, 3 ang vel
        # goal 7 pose 4 relative quat
        # middle point between thumb and index finger 3 pos 3 vector
        # distance between figertips and the target-object 5 
        # distance between body and the non-target-object 5 
        # distance between figertips and the table 5
        # add up to 111

        # fingertip observations, state(pose and vel) + force-torque sensors
        num_ft_states = 13 * 5  # 65

        self.obs_buf[:, 0:self.num_botyard_hand_dofs] = unscale(self.botyard_hand_dof_pos,
                                                            self.botyard_hand_dof_lower_limits, self.botyard_hand_dof_upper_limits)
        # print("1",self.botyard_hand_dof_pos)
        # print("2",self.obs_buf[:, 0:self.num_botyard_hand_dofs])
        self.obs_buf[:, self.num_botyard_hand_dofs:2*self.num_botyard_hand_dofs] = self.vel_obs_scale * self.botyard_hand_dof_vel
        
        # self.obs_buf[:, 2*self.num_allegro_hand_dofs:3*self.num_allegro_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]
        
        fingertip_obs_start = 2*self.num_botyard_hand_dofs
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        
        ee_state_start = fingertip_obs_start + 65
        self.obs_buf[:, ee_state_start:ee_state_start + 7] = self.ee_pose
        self.obs_buf[:, ee_state_start + 7:ee_state_start + 10] = self.ee_linvel
        self.obs_buf[:, ee_state_start + 10:ee_state_start + 13] = self.vel_obs_scale * self.ee_angvel
        
        obj_obs_start = ee_state_start + 13
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.object_pos
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.object_rot
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
        

        goal_obs_start = obj_obs_start + 13  
        self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        

        mid_obs_start = goal_obs_start + 11
        self.obs_buf[:, mid_obs_start:mid_obs_start + 3] = self.finger_mid_pos
        self.obs_buf[:, mid_obs_start+3:mid_obs_start + 6] = self.finger_mid_vec
        
        obj_distance_obs_start = mid_obs_start + 6
        self.obs_buf[:, obj_distance_obs_start:obj_distance_obs_start + 5] = self.postive_distance
        self.obs_buf[:, obj_distance_obs_start + 5:obj_distance_obs_start + 10] = self.negative_distance
        
        table_distance_obs_start = obj_distance_obs_start + 10
        self.obs_buf[:, table_distance_obs_start:table_distance_obs_start + 5] = self.table_distance
        
        if visdebug:
            print("robotstate: " + str(self.obs_buf[:, 0:2*self.num_botyard_hand_dofs]))
            print("fingertip state: " + str(self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states]))
            print("ee: " + str(self.obs_buf[:, ee_state_start:ee_state_start + 13]))
            print("obj: " + str(self.obs_buf[:, obj_obs_start:obj_obs_start + 13]))
            print("goal: " + str(self.obs_buf[:, goal_obs_start:goal_obs_start + 11]))
            print("mid_obs: " + str(self.obs_buf[:, mid_obs_start:mid_obs_start + 6]))
            print("obj_distance: " + str(self.obs_buf[:, obj_distance_obs_start:obj_distance_obs_start + 10]))
            print("table_distance_obs_start: " + str(self.obs_buf[:, table_distance_obs_start:table_distance_obs_start + 5]))
        #add up to 181

    @torch.no_grad()
    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    @torch.no_grad()
    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_botyard_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + 0.02 #+ \
            # self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        # new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        new_object_rot = quat_from_angle_axis(rand_floats[:, 3], self.z_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset botyard hand
        delta_max = self.botyard_hand_dof_upper_limits - self.botyard_hand_dof_default_pos
        delta_min = self.botyard_hand_dof_lower_limits - self.botyard_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_botyard_hand_dofs]

        # pos = self.allegro_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        pos = self.botyard_hand_default_dof_pos
        self.botyard_hand_dof_pos[env_ids, :] = pos
        self.botyard_hand_dof_vel[env_ids, :] = self.botyard_hand_dof_default_vel
        
        self.prev_targets[env_ids, :self.num_botyard_hand_dofs] = pos
        self.cur_targets[ env_ids, :self.num_botyard_hand_dofs] = pos
        self.ee_target_pose[env_ids, :] = self.ee_default_target_pose.clone()

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        all_indices = torch.unique(torch.cat([all_hand_indices, self.table_indices[env_ids],
                                                 object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self.ee_target_pose[env_ids,:] = self.rigid_body_states[env_ids, self.ee_handle, 0:7].clone()
        # print("After reset - dof_state pos[0, :7]:", self.dof_state.view(self.num_envs, -1, 2)[0, :7, 0].cpu().numpy())
        # print("botyard_hand_dof_pos[0, :7]:", self.botyard_hand_dof_pos[0, :7].cpu().numpy())
        # print("ee_pose[0]:", self.ee_pose[0].cpu().numpy())
        # for i in range(5):  # 沉降 5 步
        #     self.gym.simulate(self.sim)
        #     self.gym.fetch_results(self.sim, True)
        #     self.gym.refresh_dof_state_tensor(self.sim)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)
        #     print(f"After settle step {i} - ee_pose[0]:", self.ee_pose[0].cpu().numpy())

    @torch.no_grad()
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
        
        self.ee_target_pose[:,:3] = torch.clamp(self.ee_target_pose[:,:3],self.ee_pos_lower_limits,self.ee_pos_upper_limits)
        # delta_rot_vector = desired_ang_vel * self.dt
        # # print("delta_rot_vector",delta_rot_vector[0,:])
        # delta_quat = quat_from_rotvec(delta_rot_vector)
        # # print("delta_quat",delta_quat[0,:])
        # self.ee_target_pose[:,3:7] = quat_mul(delta_quat, self.ee_target_pose[:,3:7])
        # self.ee_target_pose[:,3:7] = torch.nn.functional.normalize(self.ee_target_pose[:,3:7], p=2, dim=-1)
        # print("ee_target_pose",self.ee_target_pose[0,:])
        # print("ee_pose",self.ee_pose[0,:])

        pos_err = self.ee_target_pose[:,:3] - self.ee_pos
        # print("pos_err",pos_err[0,:])
        orn_err = orientation_error(self.ee_target_pose[:,3:7], self.ee_rot)
        # print("orn_err",orn_err[0,:])

        dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)
        dtheta = self.cal_ik(dpose)
        
        return dtheta

    @torch.no_grad()
    def cal_ik(self, dpose):
        # j_eef 就是雅可比矩阵 J
        # j_eef_T 是 J 的转置 Jᵀ
        j_eef_T = torch.transpose(self.j_eef, 1, 2)

        # lmbda 就是阻尼项 λ² * I
        lmbda = torch.eye(6, device=self.device) * (self.hand_ik_damping ** 2)

        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, self.num_ik_arm_dof)
        return u

    @torch.no_grad()
    def input_action_to_action(self):
        ## action space
        ## 6 dof ee ; FAJ3 JAJ1 FFJ432 LFJ432 MFJ432 RFJ432 THJ4321; 24

        ################### ik ################################
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        targets = self.prev_targets[:, self.actuated_dof_indices].clone()
        targets[:,0:self.num_ik_arm_dof] = self.botyard_hand_dof_pos[:,0:self.num_ik_arm_dof] + self.control_ik(self.actions[:,:6])

        def map_finger_action_to_action():
            tarfin = torch.zeros_like(targets)
            ## FAJ31
            if self.num_ik_arm_dof == 7:
                tarfin[:,7:9]  = self.actions[:,6:8]
            ## FFJ 432  FFJ1
            tarfin[:,9:12] = self.actions[:,8:11]
            tarfin[:,12]   = self.actions[:,10]
            ## LFJ 432  LFJ1
            tarfin[:,13:16] = self.actions[:,11:14]
            tarfin[:,16]    = self.actions[:,13]
            ## MFJ 432  MFJ1
            tarfin[:,17:20] = self.actions[:,14:17]
            tarfin[:,20]    = self.actions[:,16]
            ## RFJ 432  RFJ1
            tarfin[:,21:24] = self.actions[:,17:20]
            tarfin[:,24]    = self.actions[:,19]
            ## THJ 4321
            tarfin[:,25:29] = self.actions[:,20:24]
            return tarfin

        if self.use_relative_control:
            targets[:, :] = targets[:, :] + self.botyard_hand_dof_speed_scale * self.dt * map_finger_action_to_action()
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.botyard_hand_dof_lower_limits[self.actuated_dof_indices], self.botyard_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            tarfin_norm = map_finger_action_to_action()
            targets[:, self.num_ik_arm_dof:] = scale(tarfin_norm[:, self.num_ik_arm_dof:], self.botyard_hand_dof_lower_limits[self.num_ik_arm_dof:], self.botyard_hand_dof_upper_limits[self.num_ik_arm_dof:])
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.botyard_hand_dof_lower_limits[self.actuated_dof_indices], self.botyard_hand_dof_upper_limits[self.actuated_dof_indices])
        # print("target",self.prev_targets[0, :7])
        # print("actual",self.botyard_hand_dof_pos[0,:7])
        # print("err", self.cur_targets[0, :7] - self.botyard_hand_dof_pos[0,:7])
        # print("final",self.cur_targets[0,:])
        # print(self.cur_targets[0,:])
        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    @torch.no_grad()
    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        # print(self.actions[0,:])
        self.input_action_to_action()
        
    @torch.no_grad()
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.free_cuda_cache_count >= self.free_cuda_cache_threshold:
            self.free_cuda_cache_count = 0
            print(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.free_cuda_cache_count += 1

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.botyard_right_hand_pos[i], self.botyard_right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.botyard_left_hand_pos[i], self.allegro_left_hand_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    # (将这个函数作为你的 Task 类的一个方法，或替换旧的同名函数)

    

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_pos, ee_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool,
    finger_mid_dis, postive_distance_mod, fingertip_distance, ee_obj_rot_cos
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # print("dist_rew:", dist_rew)
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward1 = torch.exp(-(0.2*(dist_rew * dist_reward_scale + rot_dist)))
    reward2 = torch.exp(-10 * finger_mid_dis.t())
    reward3 = torch.exp(-10 * postive_distance_mod.t())
    # reward4 = torch.exp(-10 * fingertip_distance.t())
    reward5 = ee_obj_rot_cos * 0.2
    reward =  reward1 + 0.4 * reward2 + 0.5 * reward3 + reward5 #+ 0.02 * reward4
    # print("reward before:", reward)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0.02, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(ee_pos[:, 2] <= 0.35, torch.ones_like(resets), resets)
    resets = torch.where(ee_pos[:, 0] <= -0.1, torch.ones_like(resets), resets)
    resets = torch.where(ee_pos[:, 0] >= 0.85, torch.ones_like(resets), resets)
    resets = torch.where(ee_pos[:, 1] <= -0.7, torch.ones_like(resets), resets)
    resets = torch.where(ee_pos[:, 1] >= 0.7, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)
    # print("rew shape:", reward.shape)
    # print("resets shape:", resets.shape)
    # print("goal_resets shape:", goal_resets.shape)
    # print("progress_buf shape:", progress_buf.shape)
    # print("successes shape:", successes.shape)
    # print("cons_successes shape:", cons_successes.shape)
    return reward, resets, goal_resets, progress_buf, successes, cons_successes

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

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

@torch.jit.script
def compute_table_distance_jit(poses_a, vertices_a, table_height: float):
    # type: (torch.Tensor, torch.Tensor, float) -> torch.Tensor
    """
    一个JIT编译的函数，用于计算点云到水平桌面的最小垂直距离。

    Args:
        poses_a (Tensor): 物体的位姿, 形状 (num_envs, 13)
        vertices_a (Tensor): 物体表面的采样点 (局部坐标), 形状 (num_envsnum_samples, 3)
        table_height (float): 桌面的Z坐标高度

    Returns:
        Tensor: 每个环境中物体到桌面的最小距离, 形状 (num_envs,)
    """
    num_envs = poses_a.shape[0]
    num_samples = vertices_a.shape[1]

    pos_a, rot_a = poses_a[:, 0:3], poses_a[:, 3:7]

    rot_a_expanded = rot_a.unsqueeze(1).expand(-1, num_samples, -1)
    pos_a_expanded = pos_a.unsqueeze(1)

    pcd_a_world = quat_apply(rot_a_expanded, vertices_a) + pos_a_expanded
    
    # 只计算Z轴方向的距离
    pcd_a_world_z = pcd_a_world[..., 2]
    dist_to_table = pcd_a_world_z - table_height
    
    # 找到每个环境中的最小距离
    min_distances, _ = torch.min(dist_to_table, dim=1)
    
    return min_distances

@torch.jit.script
def compute_surface_distance_jit(poses_a, poses_b, vertices_a, vertices_b):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    一个JIT编译的函数，用于计算两批点云之间的最小表面距离。

    Args:
        poses_a (Tensor): A批物体的位姿, 形状 (num_envs, 13)
        poses_b (Tensor): B批物体的位姿, 形状 (num_envs, 13)
        vertices_a (Tensor): A物体表面的采样点 (局部坐标), 形状 (num_envs, num_samples, 3)
        vertices_b (Tensor): B物体表面的采样点 (局部坐标), 形状 (num_envs, num_samples, 3)

    Returns:
        Tensor: 每个环境中A和B之间的最小距离, 形状 (num_envs,)
    """
    num_envs = poses_a.shape[0]
    num_samples = vertices_a.shape[1]

    # 1. 获取实时位姿
    pos_a, rot_a = poses_a[:, 0:3], poses_a[:, 3:7]
    pos_b, rot_b = poses_b[:, 0:3], poses_b[:, 3:7]

    # 2. 变换点云到世界坐标系 (向量化操作)
    # 扩展旋转和平移张量以进行广播
    rot_a_expanded = rot_a.unsqueeze(1).expand(-1, num_samples, -1) # (num_envs, num_samples, 4)
    rot_b_expanded = rot_b.unsqueeze(1).expand(-1, num_samples, -1) # (num_envs, num_samples, 4)
    pos_a_expanded = pos_a.unsqueeze(1) # (num_envs, 1, 3)
    pos_b_expanded = pos_b.unsqueeze(1) # (num_envs, 1, 3)

    # 计算世界坐标系中的点云
    pcd_a_world = quat_apply(rot_a_expanded, vertices_a) + pos_a_expanded
    pcd_b_world = quat_apply(rot_b_expanded, vertices_b) + pos_b_expanded
    # 两个点云的形状都是: (num_envs, num_samples, 3)

    # 3. 高效计算距离矩阵
    dist_matrix = torch.cdist(pcd_a_world, pcd_b_world)
    # dist_matrix 的形状为: (num_envs, num_samples, num_samples)

    # 4. 找到每个环境的最小值
    # 将最后两个维度展平以找到全局最小值
    dist_flat = dist_matrix.view(num_envs, -1)
    min_distances, _ = torch.min(dist_flat, dim=1)

    return min_distances

if __name__ ==  "__main__":
    import dexhand as bi
    env_name = 'BotyardHandPick'
    algo = "ppo"
    env = bi.make(env_name, algo)

    obs = env.reset()
    terminated = False
    cnt = 0
    while not terminated:
        act = torch.tensor(env.action_space.sample())
        act = torch.zeros(29)
        # a = [10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 27, 28]
        # j4 = [9, 13, 17, 21, 25]
        # act[a] = -1.0
        if cnt > 90:
            #act[j4] = -1
            # act[0] = -0.9
            act[1] = -0.9
            # act[2] = -0.9
            act[3:6] = 0
        else:
            # act[j4]= 1
            # act[0] = 0.9
            act[1] = 0.9
            # act[2] = 0.9
            act[3:6] = 0
        act = act.repeat((env.num_envs, 1))
        # print(env.task.ee_target_pose[0,:])
        # print("action: " + str(act[1,:]))
        obs, reward, done, info = env.step(act)
        cnt += 1
        if cnt > 100:
            cnt = 0
# 9, 13, 17, 21, 25
#8 9 10 14 18 21 26
#   Name 'FAJ1': Index 8
#   Name 'FAJ3': Index 7
#   Name 'FFJ1': Index 12
#   Name 'FFJ2': Index 11
#   Name 'FFJ3': Index 10
#   Name 'FFJ4': Index 9
#   Name 'LFJ1': Index 16
#   Name 'LFJ2': Index 15
#   Name 'LFJ3': Index 14
#   Name 'LFJ4': Index 13
#   Name 'MFJ1': Index 20
#   Name 'MFJ2': Index 19
#   Name 'MFJ3': Index 18
#   Name 'MFJ4': Index 17
#   Name 'RFJ1': Index 24
#   Name 'RFJ2': Index 23
#   Name 'RFJ3': Index 22
#   Name 'RFJ4': Index 21
#   Name 'THJ1': Index 28
#   Name 'THJ2': Index 27
#   Name 'THJ3': Index 26
#   Name 'THJ4': Index 25
#   Name 'joint1': Index 0
#   Name 'joint2': Index 1
#   Name 'joint3': Index 2
#   Name 'joint4': Index 3
#   Name 'joint5': Index 4
#   Name 'joint6': Index 5
#   Name 'joint7': Index 6