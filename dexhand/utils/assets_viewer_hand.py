import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R
from dexhand.utils.torch_jit_utils import *
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
URDF_FILE = "botyard/ur_by_description/urdf/dexhand.urdf"

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
    asset_options.disable_gravity = True

    asset = gym.load_asset(sim, ASSET_ROOT, URDF_FILE, asset_options)
    if asset is None:
        print("!!! Failed to load asset")
        quit()

    num_dofs = gym.get_asset_dof_count(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    for i in range(num_dofs):
        # dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        dof_props['stiffness'][i] = 100
        dof_props['damping'][i] = 20
        dof_props['effort'][i] = 0.5
        # dof_props['armature'][i] = 0.004
    
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
    pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
    actor_handle = gym.create_actor(env, asset, pose, "MyRobot", 0, 0, 0)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # --- 4. 准备张量 ---
    gym.refresh_dof_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    
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
    ee_handle = gym.find_asset_rigid_body_index(asset, "central")
    rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor)
    rigid_body_states = rigid_body_tensor.view(-1, 13)
    ee_pose   = rigid_body_states[ee_handle, 0:7]
    ee_pos    = rigid_body_states[ee_handle, 0:3]
    ee_quat   = rigid_body_states[ee_handle, 3:7]
    ee_rot    = R.from_quat(ee_quat).as_euler('xyz', degrees=False)
    print("\n--- INTERACTIVE PD CONTROL ---")
    print("  UP/DOWN: Select | LEFT/RIGHT: Change Target | R: Reset | P: Print | V: View | Q/ESC: Quit\n")
    last = False
    while not gym.query_viewer_has_closed(viewer):
        
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "QUIT":
                gym.destroy_viewer(viewer); gym.destroy_sim(sim); return
            # ... (键盘事件处理逻辑) ...
            if evt.action == "A":
                if not last:
                    current_dof_idx = (current_dof_idx - 1 + num_dofs) % num_dofs
                    last = True
                else:
                    last = False
            elif evt.action == "D":
                if not last:
                    current_dof_idx = (current_dof_idx + 1) % num_dofs
                    last = True
                else:
                    last = False
            elif evt.action == "S":
                dof_targets[current_dof_idx] -= angle_step
            elif evt.action == "W":
                dof_targets[current_dof_idx] += angle_step
            elif evt.action == "R":
                dof_targets[current_dof_idx] = initial_dof_pos[current_dof_idx]
            elif evt.action == "P":
                gym.refresh_dof_state_tensor(sim)
                current_pos = dof_state[:, 0]
                print("\n--- Angles (Pos | Target) ---")
                for i in range(num_dofs):
                    print(f"  {dof_names[i]}: {current_pos[i]:.3f} | {dof_targets[i]:.3f}")
        
        dof_targets = tensor_clamp(dof_targets, dof_lower_limits, dof_upper_limits)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        current_pos = dof_state[:, 0]
        
        target_angle = dof_targets[current_dof_idx]
        current_angle = current_pos[current_dof_idx]
        ee_rot    = R.from_quat(ee_quat).as_euler('xyz', degrees=False)
        # print(f"\rControlling: {dof_names[current_dof_idx]} | Target: {target_angle:.3f} | Current: {current_angle:.3f}", end="")
        print(f"\rControlling: {dof_names[current_dof_idx]} | Target: {target_angle:.3f} | Current: {current_angle:.3f} | ee: {ee_pos},{ee_rot} |d", end="")

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

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