
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 9)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)

def get_shape_map(gym, asset):
    num_shapes    = gym.get_asset_rigid_shape_count(asset)
    num_bodies    = gym.get_asset_rigid_body_count(asset)
    body_names   = gym.get_asset_rigid_body_names(asset)
    body_shape_map = gym.get_asset_rigid_body_shape_indices(asset)
    _map = {}
    for i in range(num_bodies):
        name = body_names[i]
        shape_idx_range = body_shape_map[i]
        if shape_idx_range.count > 0:
            shape_idx = shape_idx_range.start
            _map[name] = shape_idx
        if shape_idx_range.count > 1:
            shape_idx = range(shape_idx_range.start, shape_idx_range.start + shape_idx_range.count)
            _map[name] = shape_idx
    print(_map)
    return _map 

def print_asset_info(gym, asset):
    num_bodies    = gym.get_asset_rigid_body_count(asset)
    num_shapes    = gym.get_asset_rigid_shape_count(asset)
    num_dofs      = gym.get_asset_dof_count(asset)
    num_actuators = gym.get_asset_dof_count(asset)
    num_tendons   = gym.get_asset_tendon_count(asset)

    print("num_bodies: ",    num_bodies)
    print("num_shapes: ",    num_shapes)
    print("num_dofs: ",      num_dofs)
    print("num_actuators: ", num_actuators)
    print("num_tendons: ",   num_tendons)

def set_hand_rigid_body_props(env, actor_handle, botyard_hand_asset):
    num_shapes     = gym.get_asset_rigid_shape_count(botyard_hand_asset)
    num_bodies     = gym.get_asset_rigid_body_count(botyard_hand_asset)
    body_names     = gym.get_asset_rigid_body_names(botyard_hand_asset)
    body_shape_map = gym.get_asset_rigid_body_shape_indices(botyard_hand_asset)
    hand_shape_name_id_map = {}
    for i in range(num_bodies):
        name = body_names[i]
        shape_idx_range = body_shape_map[i]
        if shape_idx_range.count > 0:
            shape_idx = shape_idx_range.start
            hand_shape_name_id_map[name] = shape_idx
        if shape_idx_range.count > 1:
            shape_idx = range(shape_idx_range.start, shape_idx_range.start + shape_idx_range.count)
            hand_shape_name_id_map[name] = shape_idx

    hand_rigid_body_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    finger_name = ['lfdistal', 'rfdistal', 'mfdistal', 'ffdistal', 'thdistal',
                'lfmiddle', 'rfmiddle', 'mfmiddle', 'ffmiddle', 'thmiddle',
                'lfproximal', 'rfproximal', 'mfproximal', 'ffproximal', 'thproximal']
    finger_id = [hand_shape_name_id_map[name] for name in finger_name]
    base_name = ['pmbase', 'palm', 'fabase']
    base_id = [hand_shape_name_id_map[name] for name in base_name]
    th_name = ["thbase", "thproximal", "thmiddle", "thdistal"]
    th_id = [hand_shape_name_id_map[name] for name in th_name]
    ff_name = ["ffbase", "ffproximal", "ffmiddle", "ffdistal"]
    ff_id = [hand_shape_name_id_map[name] for name in ff_name]
    mf_name = ["mfbase", "mfproximal", "mfmiddle", "mfdistal"]
    mf_id = [hand_shape_name_id_map[name] for name in mf_name]
    rf_name = ["rfbase", "rfproximal", "rfmiddle", "rfdistal"]
    rf_id = [hand_shape_name_id_map[name] for name in rf_name]
    lf_name = ["lfbase", "lfproximal", "lfmiddle", "lfdistal"]
    lf_id = [hand_shape_name_id_map[name] for name in lf_name]
    for i in range(len(hand_rigid_body_props)):
        if i in [base_id[0], base_id[1]]:
            hand_rigid_body_props[i].filter = 0b11111
        elif i in th_id:
            hand_rigid_body_props[i].filter = (1 << 0)
        elif i in ff_id:
            hand_rigid_body_props[i].filter = (1 << 1)
        elif i in mf_id:
            hand_rigid_body_props[i].filter = (1 << 2)
        elif i in rf_id:
            hand_rigid_body_props[i].filter = (1 << 3)
        elif i in lf_id:
            hand_rigid_body_props[i].filter = (1 << 4)
        else:
            hand_rigid_body_props[i].filter = 0
        if i in finger_id:
            hand_rigid_body_props[i].contact_offset = 0.005
            hand_rigid_body_props[i].rest_offset = 0.00

    gym.set_actor_rigid_shape_properties(env, actor_handle, hand_rigid_body_props)
    # props[shape_name_id_map['lfdistal']].filter = (1 << 1)
    # props[shape_name_id_map['rfdistal']].filter = (1 << 1)



# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"
args.use_gpu_pipeline = True

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.1

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

ASSET_ROOT = "../assets"
URDF_FILE = "botyard/panda_by_description/urdf/panda_by.urdf"
DEVICE = 'cuda:0' if args.use_gpu_pipeline else 'cpu'

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.use_mesh_materials = True
asset_options.armature = 0.001
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.thickness = 0.0001
asset_options.use_physx_armature = True
asset_options.disable_gravity = False

asset = gym.load_asset(sim, ASSET_ROOT, URDF_FILE, asset_options)
if asset is None:
    print("!!! Failed to load asset")
    quit()

num_dofs = gym.get_asset_dof_count(asset)
dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)

for i in range(7, 29):
    # dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 150
    dof_props['damping'][i] = 20
    dof_props['effort'][i] = 0.5
    # dof_props['armature'][i] = 0.004
x_arm_dof_effort = to_torch([87, 87, 87, 87, 12, 12, 12], dtype=torch.float, device=DEVICE)

for i in range(0, 7):
    # dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 10000
    dof_props['damping'][i] = 200
    dof_props['effort'][i] = x_arm_dof_effort[i]
    # dof_props['armature'][i] = 0.01

# dof_props["stiffness"].fill(625.0)
# dof_props["damping"].fill(50.0)

dof_lower_limits = torch.tensor(dof_props['lower'], device=DEVICE)
dof_upper_limits = torch.tensor(dof_props['upper'], device=DEVICE)

print(f"\nLoaded asset with {num_dofs} DOFs:")
for i in range(num_dofs):
    print(f"  DOF {i}: {dof_names[i]} (Limits: [{dof_lower_limits[i]:.2f}, {dof_upper_limits[i]:.2f}])")
print_asset_info(gym, asset)

default_dof_pos = torch.zeros(num_dofs,dtype=torch.float32, device=DEVICE)
default_dof_pos[:7] = torch.tensor([0,-1.3,0,-2.4,0,2.66,0])
default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos.cpu()

shape_name_id_map = get_shape_map(gym, asset)

hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(-0.05, 0.0, 0)

table_dims = gymapi.Vec3(0.65, 1.5, 0.6)
table_asset_options = gymapi.AssetOptions()
table_asset_options.fix_base_link = True
table_asset_options.flip_visual_attachments = True
table_asset_options.collapse_fixed_joints = True
table_asset_options.disable_gravity = True
table_asset_options.thickness = 0.001
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.65, 0.0, 0.5 * table_dims.z)
table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

object_asset_file = 'botyard/panda_by_description/meshes/object/box_50mm.urdf'
object_asset_options = gymapi.AssetOptions()
object_asset_options.density = 100
object_asset = gym.load_asset(sim, ASSET_ROOT, object_asset_file, object_asset_options)

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

box_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
ee_idxs = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.7 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add hand
    hand_handle = gym.create_actor(env, asset, hand_pose, "hand", i, 0)
    hand_idx = gym.get_actor_index(env, hand_handle, gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    set_hand_rigid_body_props(env, hand_handle, asset)
    
    # set dof properties
    gym.set_actor_dof_properties(env, hand_handle, dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, hand_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, hand_handle, default_dof_pos.cpu())

    # get global index of hand in rigid body state tensor
    ee_idx = gym.find_actor_rigid_body_index(env, hand_handle, "central", gymapi.DOMAIN_SIM)
    ee_idxs.append(ee_idx)

box_idxs = to_torch(box_idxs, dtype=torch.long, device=DEVICE)
hand_idxs = to_torch(hand_idxs, dtype=torch.long, device=DEVICE)
ee_idxs = to_torch(ee_idxs, dtype=torch.long, device=DEVICE)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
ee_home_pose = torch.tensor([0.415, 0, 0.88, 0, 0, 0, 1], dtype=torch.float, device=DEVICE)
ee_target_pose = ee_home_pose.clone().repeat((num_envs,1))

# hand orientation for grasping
# down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
# box_half_size = 0.5 * box_size
# corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
# corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "hand")
jacobian = gymtorch.wrap_tensor(_jacobian)
ee_handle = gym.find_asset_rigid_body_index(asset, "central")
# jacobian entries corresponding to franka hand
j_eef = jacobian[:, ee_handle - 1, :, :9]

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, -1)
dof_vel = dof_states[:, 1].view(num_envs, -1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)
go_home = torch.full([num_envs], False, dtype=torch.bool).to(device)
close_grasp = torch.full([num_envs], False, dtype=torch.bool).to(device)
# Set action tensors
pos_action = default_dof_pos.repeat((num_envs, 1))
# grasp_finger_pos = torch.tensor([0, 0.38, 0.84, 0.84, 0, 0.8, 0.68, 0.68, 0, 0.45, 0.88, 0.88, 0, 0.59, 0.71, 0.71, 0, 0.59, 0.5, 0.61], dtype=torch.float, device=device)
grasp_finger_pos = torch.tensor([0, 0.4, 0.8, 0.6, 0, 0.6, 0.6, 0.6, 0, 0.2, 0.6, 0.6, 0, 0.4, 0.4, 0.4, 0, 0.2, 0.8, 0.0], dtype=torch.float, device=device)
grasp_finger_pos = grasp_finger_pos.repeat((num_envs, 1))
home_finger_pos = torch.zeros_like(grasp_finger_pos)
grasp_offset = torch.tensor([0.094, -0.041, -0.05], dtype=torch.float, device=device)
# simulation loop
init_pos = ee_home_pose[:3].repeat((num_envs,1))
init_rot = ee_home_pose[3:7].repeat((num_envs,1))

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    ee_pos = rb_states[ee_idxs, :3]
    ee_rot = rb_states[ee_idxs, 3:7]
    
    to_box = box_pos - ee_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # determine if we're holding the box (grippers are closed and box is near)
    grasp_dist = torch.norm(dof_pos[:, 9:] - grasp_finger_pos, dim=-1).unsqueeze(-1)   
    box_grasp_offset = torch.norm(to_box - grasp_offset, dim=-1).unsqueeze(-1)
    grasped = (grasp_dist < 0.41) & (box_grasp_offset < 0.1)
    # print(grasped.shape)
    # print("box_grasp_offset",box_grasp_offset.shape)
    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - ee_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | grasped.squeeze(-1)).unsqueeze(-1)
    go_home = go_home | (return_to_start.squeeze(-1)) 
    go_home = go_home & (init_dist > 0.02)
    # print(return_to_start.shape)

    # grasp manipulation
    close_grasp = (box_grasp_offset.squeeze(-1) < 0.01) | (grasped.squeeze(-1)) | ((box_grasp_offset.squeeze(-1) < 0.08) & close_grasp)
    # print(close_grasp.shape)
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 2] > 0.75)
    # print(hand_restart.shape)
    keep_going = torch.logical_not(hand_restart)
    # print(keep_going.shape)
    close_grasp = close_grasp & keep_going
    # print(close_grasp.shape)
    # print(grasp_finger_pos.shape)
    # print(home_finger_pos.shape)
    grasp_acts = torch.where(close_grasp.unsqueeze(-1), grasp_finger_pos, home_finger_pos)
    pos_action[:, 9:] = grasp_acts

    # if close to box and approaching from above, move down to grasp
    grasp_pos = box_pos - grasp_offset
    grasp_rot = torch.zeros_like(ee_rot)
    grasp_rot[:, 3] = 1.0

    # print(close_grasp.shape,ee_pos.shape,grasp_pos.shape,return_to_start.shape,init_pos.shape)
    goal_pos = torch.where(close_grasp.unsqueeze(-1), ee_pos, grasp_pos)
    goal_pos = torch.where(return_to_start, init_pos, goal_pos)
    goal_rot = torch.where(return_to_start, init_rot, grasp_rot)
    print(hand_restart[1],grasped[1],grasp_dist[1],(to_box - grasp_offset)[1,:])
    # print("goal_pos", goal_pos[0,:])

    pos_err = goal_pos - ee_pos
    # print("pos_err",pos_err[0,:])
    orn_err = orientation_error(goal_rot, ee_rot)
    # print("orn_err",orn_err[0,:])

    dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)
    limit = 0.01
    dpose = torch.clamp(dpose, -limit, limit)
    dtheta = control_ik(dpose)
    pos_action[:, :9] = tensor_clamp(dof_pos[:, :9] + dtheta, dof_lower_limits[:9], dof_upper_limits[:9])


    box_out = (box_pos[:, 2] < 0.4) & (box_pos[:, 0] > 0.2) & (box_pos[:, 0] < -0.4) & (box_pos[:, 1] < -0.5) & (box_pos[:, 1] > 0.5)
    reset_env_ids = torch.nonzero(box_out).squeeze(-1)
    box_random_pos = torch.zeros_like(box_pos)
    box_random_pos[reset_env_ids, 0] = table_pose.p.x + torch.rand_like(box_pos[reset_env_ids, 0]) * -0.2
    box_random_pos[reset_env_ids, 1] = table_pose.p.y + torch.rand_like(box_pos[reset_env_ids, 1]) * -0.2
    box_random_pos[reset_env_ids, 2] = table_dims.z + 0.7 * box_size
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(box_random_pos[reset_env_ids]), gymtorch.unwrap_tensor(reset_env_ids), len(reset_env_ids))
    # box_random_pos = torch.zeros_like(box_pos)
    # box_random_pos[:, 0] = table_pose.p.x + torch.rand_like(box_pos[:, 0]) * -0.2
    # box_random_pos[:, 1] = table_pose.p.y + torch.rand_like(box_pos[:, 1]) * -0.2
    # box_random_pos[:, 2] = table_dims.z + 0.7 * box_size
    # box_pos = torch.where(box_out.unsqueeze(-1), box_random_pos, box_pos)
    # box_rot = torch.where(box_out.unsqueeze(-1), torch.tensor([0, 0, 0, 1], device=device).view(1,4), box_rot)

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

