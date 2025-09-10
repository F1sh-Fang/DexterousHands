import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 找到urdf文件的路径
    urdf_file_name = 'panda_by.urdf'
    urdf = os.path.join(
        get_package_share_directory('panda_by_description'),
        'urdf',
        urdf_file_name)
    
    rviz_config_file = os.path.join(get_package_share_directory('panda_by_description'),'rviz', 'display.rviz')
    
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # 启动robot_state_publisher节点
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
        ),

        # (可选) 启动一个GUI来控制关节
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
        ),

        # 启动RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
        )
    ])