import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R
from bidexhands.utils.torch_jit_utils import *
import torch
import math

# =================================================================================
# --- 1. 配置 ---
# =================================================================================
# 先用Franka模型进行“黄金标准”测试
# ASSET_ROOT = "../assets"
# URDF_FILE = "urdf/cartpole.urdf"

# 确认Franka能运行后，再换成您的模型
ASSET_ROOT = "../assets"
URDF_FILE = "botyard/panda_by_description/urdf/panda_by.urdf"

# 强制所有操作都在GPU上，这是Isaac Gym的标准用法
DEVICE = 'cpu'
# =================================================================================

def main():

    gym = gymapi.acquire_gym()

    # --- 1. 配置仿真参数 ---
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # 这是控制设备的核心参数
    sim_params.use_gpu_pipeline = False
    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    
    compute_device_id = 0
    graphics_device_id = 0

    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("!!! Failed to create sim")
        quit()

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # --- 2. 加载资源 ---
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
        dof_props['stiffness'][i] = 100
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
    shape_name_id_map = get_shape_map(gym, asset)

    # --- 3. 创建环境和Actor ---
    env = gym.create_env(sim, gymapi.Vec3(-2,-2,0), gymapi.Vec3(2,2,2), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-0.05, 0.0, 0)
    actor_handle = gym.create_actor(env, asset, pose, "hand", 0, 0, 0)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

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
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0, 0)

    box_size = 0.05
    asset_options = gymapi.AssetOptions()
    asset_options.density = 1000.0
    asset_options.thickness = 0.001
    asset_options.disable_gravity = False
    asset_options.fix_base_link = False
    box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_pose = gymapi.Transform()
    box_pose.p.x = table_pose.p.x
    box_pose.p.y = table_pose.p.y
    box_pose.p.z = table_dims.z + 0.7 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", 0, 0, 0)

    # --- 4. 准备张量 ---
    gym.refresh_dof_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    dof_state[0:7,0] = torch.tensor([0,-1.3,0,-2.4,0,2.66,0])
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    initial_dof_pos = dof_state[:, 0].clone()
    dof_targets = initial_dof_pos.clone()

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("!!! Failed to create viewer")
        quit()
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2, 1, 1.5), gymapi.Vec3(0, 0, 0.5))
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A,"A")       
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D,"D")        
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W,"W")      
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S,"S")     
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,"R") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P,"P") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP,"up") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN,"down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT,"left") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT,"right")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_PAGE_UP,"pup") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_PAGE_DOWN,"pdown")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1,"1")      
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2,"2")     
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3,"3") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4,"4") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5,"5") 
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6,"6") 


    # {'link0': 0, 'link1': 1, 'link2': 2, 'link3': 3, 'link4': 4, 'link5': range(5, 8), 'link6': 8, 'link7': 9, 
    #  'fabase': 10, 'pmbase': 11, 'palm': 12, 'ffbase': 13, 'ffproximal': 14, 'ffmiddle': 15, 'ffdistal': 16, 
    #  'lfbase': 17, 'lfproximal': 18, 'lfmiddle': 19, 'lfdistal': 20, 'mfbase': 21, 'mfproximal': 22, 'mfmiddle': 23, 
    #  'mfdistal': 24, 'rfbase': 25, 'rfproximal': 26, 'rfmiddle': 27, 'rfdistal': 28, 'thbase': 29, 'thproximal': 30, 
    #  'thmiddle': 31, 'thdistal': 32}
    props = gym.get_actor_rigid_shape_properties(env, actor_handle) ## (env, index of the asset) as input
    finger_name = ['lfdistal', 'rfdistal', 'mfdistal', 'ffdistal', 'thdistal',
                   'lfmiddle', 'rfmiddle', 'mfmiddle', 'ffmiddle', 'thmiddle',
                   'lfproximal', 'rfproximal', 'mfproximal', 'ffproximal', 'thproximal']
    finger_id = [shape_name_id_map[name] for name in finger_name]
    base_name = ['pmbase', 'palm', 'fabase']
    base_id = [shape_name_id_map[name] for name in base_name]
    th_name = ["thbase", "thproximal", "thmiddle", "thdistal"]
    th_id = [shape_name_id_map[name] for name in th_name]
    ff_name = ["ffbase", "ffproximal", "ffmiddle", "ffdistal"]
    ff_id = [shape_name_id_map[name] for name in ff_name]
    mf_name = ["mfbase", "mfproximal", "mfmiddle", "mfdistal"]
    mf_id = [shape_name_id_map[name] for name in mf_name]
    rf_name = ["rfbase", "rfproximal", "rfmiddle", "rfdistal"]
    rf_id = [shape_name_id_map[name] for name in rf_name]
    lf_name = ["lfbase", "lfproximal", "lfmiddle", "lfdistal"]
    lf_id = [shape_name_id_map[name] for name in lf_name]
    for i in range(len(props)):
        if i in [base_id[0], base_id[1]]:
            props[i].filter = 0b11111
        elif i in th_id:
            props[i].filter = (1 << 0)
        elif i in ff_id:
            props[i].filter = (1 << 1)
        elif i in mf_id:
            props[i].filter = (1 << 2)
        elif i in rf_id:
            props[i].filter = (1 << 3)
        elif i in lf_id:
            props[i].filter = (1 << 4)
        else:
            props[i].filter = 0
        if i in finger_id:
            props[i].contact_offset = 0.005
            props[i].rest_offset = 0.00
    # props[shape_name_id_map['lfdistal']].filter = (1 << 1)
    # props[shape_name_id_map['rfdistal']].filter = (1 << 1)
    gym.set_actor_rigid_shape_properties(env, actor_handle, props) ### (env, index number, properties List)

    # --- 5. 主循环 ---
    current_dof_idx = 0
    angle_step = 0.1
    ############ ik #####################
    gym.prepare_sim(sim)
    gym.refresh_jacobian_tensors(sim)
    _jacobian = gym.acquire_jacobian_tensor(sim, "hand")
    jacobian = gymtorch.wrap_tensor(_jacobian)
    ee_handle = gym.find_asset_rigid_body_index(asset, "central")
    print("ee_handle",ee_handle)
    j_eef = jacobian[:, ee_handle - 1, :, :9]
    print("ee_jacobian shape: value",j_eef.shape, j_eef)

    print("\n--- INTERACTIVE PD CONTROL ---")
    print("  UP/DOWN: Select | LEFT/RIGHT: Change Target | R: Reset | P: Print | V: View | Q/ESC: Quit\n")
    last = False
    num_arm_dofs = 9
    num_hand_dofs = num_dofs - num_arm_dofs
    rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor)
    rigid_body_states = rigid_body_tensor.view(-1, 13)
    ee_pose   = rigid_body_states[ee_handle, 0:7]
    ee_pos    = rigid_body_states[ee_handle, 0:3]
    ee_quat   = rigid_body_states[ee_handle, 3:7]
    ee_home_pose = torch.tensor([0.415, 0, 0.81,0,0,0])
    ee_target_pose = ee_home_pose.clone()
    while not gym.query_viewer_has_closed(viewer):
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "QUIT":
                gym.destroy_viewer(viewer); gym.destroy_sim(sim); return
            # ... (键盘事件处理逻辑) ...
            if evt.action == "A":
                if not last:
                    current_dof_idx = (current_dof_idx - 1 + num_hand_dofs) % num_hand_dofs
                    last = True
                else:
                    last = False
            elif evt.action == "D":
                if not last:
                    current_dof_idx = (current_dof_idx + 1) % num_hand_dofs
                    last = True
                else:
                    last = False
            elif evt.action == "S":
                dof_targets[current_dof_idx + num_arm_dofs] -= angle_step
            elif evt.action == "W":
                dof_targets[current_dof_idx + num_arm_dofs] += angle_step
            elif evt.action == "R":
                dof_targets[current_dof_idx + num_arm_dofs] = initial_dof_pos[current_dof_idx + num_arm_dofs]
            elif evt.action == "P":
                gym.refresh_dof_state_tensor(sim)
                current_pos = dof_state[:, 0]
                print("\n--- Angles (Pos | Target) ---")
                for i in range(num_dofs):
                    print(f"  {dof_names[i]}: {current_pos[i]:.3f} | {dof_targets[i]:.3f}")
            elif evt.action == "up":
                ee_target_pose[0] += 0.01
            elif evt.action == "down":
                ee_target_pose[0] += -0.01
            elif evt.action == "left":
                ee_target_pose[1] += 0.01
            elif evt.action == "right":
                ee_target_pose[1] += -0.01
            elif evt.action == "pup":
                ee_target_pose[2] += 0.01
            elif evt.action == "pdown":
                ee_target_pose[2] += -0.01
            elif evt.action == "1":
                ee_target_pose[3] += 0.05
            elif evt.action == "2":
                ee_target_pose[3] += -0.05
            elif evt.action == "3":
                ee_target_pose[4] += 0.05
            elif evt.action == "4":
                ee_target_pose[4] += -0.05
            elif evt.action == "5":
                ee_target_pose[5] += 0.05
            elif evt.action == "6":
                ee_target_pose[5] += -0.05
        
        # target_euler_cpu = ee_target_pose[3:6].cpu().numpy()
        # target_quat_xyzw = R.from_euler('xyz', target_euler_cpu).as_quat()
        # goal_rot = torch.tensor(target_quat_xyzw, device=DEVICE)
        # goal_pos = ee_target_pose[0:3]
        # pos_err = goal_pos - ee_pos
        # orn_err = orientation_error(goal_rot.unsqueeze(0), ee_quat.unsqueeze(0)).squeeze(0)
        # dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)
        # dof_vel = control_ik(dpose, j_eef)
        # dof_targets[:7] = dof_state[:7] + tensor_clamp(dof_vel, -0.1, 0.1) # 限制单步最大增量
        ee_dpose = ee_target_pose.clone()
        ee_target_quat = torch.tensor(R.from_euler("xyz",ee_target_pose[3:6],degrees=False).as_quat())
        ee_dpose[0:3] -= ee_pos
        ee_dpose[3:6] = orientation_error(ee_target_quat.unsqueeze(0),ee_quat.unsqueeze(0))
        dof_targets[:num_arm_dofs] = dof_state[:num_arm_dofs, 0] + control_ik(ee_dpose,j_eef)
        dof_targets = tensor_clamp(dof_targets, dof_lower_limits, dof_upper_limits)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        gym.refresh_dof_state_tensor(sim)
        current_pos = dof_state[:, 0]
        
        target_angle = dof_targets[current_dof_idx + num_arm_dofs]
        current_angle = current_pos[current_dof_idx + num_arm_dofs]
        print(f"\rControlling: {dof_names[current_dof_idx  + num_arm_dofs]} | Target: {target_angle:.3f} | Current: {current_angle:.3f} | ee: [{', '.join([f'{x:.3f}' for x in ee_pose])}] | target: [{', '.join([f'{x:.3f}' for x in ee_target_pose])}]", end="")

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(dpose, j_eef, num_envs=1, damping=0.1, num_arm_dofs=9):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=DEVICE) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, num_arm_dofs)
    return u

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


if __name__ == "__main__":
    main()