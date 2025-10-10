import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

def main():
    # 初始化Gym
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments()

    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim = gym.create_sim(
        args.compute_device_id, 
        args.graphics_device_id, 
        args.physics_engine, 
        sim_params
    )

    # 创建地面
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # 环境配置
    num_envs = 4
    env_spacing = 2.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    # 创建资产并记录几何尺寸（用于计算表面距离）
    cube_half_extent = 0.25  # 立方体边长0.5的一半
    cube_asset = gym.create_box(sim, 0.5, 0.5, 0.5)
    
    capsule_radius = 0.2
    capsule_half_height = 0.4  # 胶囊体总高度0.8的一半
    capsule_asset = gym.create_capsule(sim, capsule_radius, 0.8)

    # 存储环境和actor信息
    envs = []
    cube_actors = []
    capsule_actors = []

    # 创建多个环境
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, 4)
        envs.append(env)

        # 放置立方体
        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(0, 0, 1.0)
        cube_actor = gym.create_actor(env, cube_asset, cube_pose, f"cube_{i}", i, 1)
        cube_actors.append(cube_actor)

        # 放置胶囊体
        capsule_pose = gymapi.Transform()
        capsule_pose.p = gymapi.Vec3(0.8 + (i%4)*0.3, 0, 1.0)
        capsule_actor = gym.create_actor(env, capsule_asset, capsule_pose, f"capsule_{i}", i, 1)
        capsule_actors.append(capsule_actor)

    # 执行仿真步骤
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 获取刚体状态的缓冲区
    max_bodies = gym.get_env_rigid_body_count(envs[0])
    body_states = gym.get_actor_rigid_body_states(envs[0], cube_actors[0], gymapi.STATE_POS)

    # 计算所有环境的表面距离
    distances = np.zeros(num_envs, dtype=np.float32)
    for i in range(num_envs):
        env = envs[i]
        
        # 获取立方体的位置
        cube_body_states = gym.get_actor_rigid_body_states(env, cube_actors[i], gymapi.STATE_POS)
        cube_pos = np.array([
            cube_body_states['pose']['p']['x'][0],
            cube_body_states['pose']['p']['y'][0],
            cube_body_states['pose']['p']['z'][0]
        ])
        
        # 获取胶囊体的位置
        capsule_body_states = gym.get_actor_rigid_body_states(env, capsule_actors[i], gymapi.STATE_POS)
        capsule_pos = np.array([
            capsule_body_states['pose']['p']['x'][0],
            capsule_body_states['pose']['p']['y'][0],
            capsule_body_states['pose']['p']['z'][0]
        ])
        
        # 计算中心距离
        center_distance = np.linalg.norm(cube_pos - capsule_pos)
        
        # 减去两个物体的表面偏移量（得到表面距离）
        surface_distance = center_distance - cube_half_extent - capsule_radius
        
        # 确保距离不为负（接触或重叠时为0）
        distances[i] = max(0.0, surface_distance)

    # 打印结果
    print(f"手动计算表面距离结果:")
    for i in range(num_envs):
        print(f"环境 {i}: 表面距离 = {distances[i]:.4f}")

    # 清理资源
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
