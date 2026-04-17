import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('photogrammetry_nbv')
    init_pkg_share = FindPackageShare('photogrammetry_init')
    mission_yaml = PathJoinSubstitution([pkg_share, 'config', 'unified_mission.yaml'])
    bridge_yaml = PathJoinSubstitution([init_pkg_share, 'config', 'bridge.yaml'])

    px4_autopilot_path = LaunchConfiguration('px4_autopilot_path')
    px4_gz_world = LaunchConfiguration('px4_gz_world')
    px4_make_target = LaunchConfiguration('px4_make_target')
    xrce_udp_port = LaunchConfiguration('xrce_udp_port')
    start_px4 = LaunchConfiguration('start_px4')
    start_xrce_agent = LaunchConfiguration('start_xrce_agent')
    start_bridge = LaunchConfiguration('start_bridge')

    px4_cmd = ['bash', '-lc', [
        'cd ', px4_autopilot_path,
        ' && PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD=', px4_gz_world,
        ' make px4_sitl ', px4_make_target,
    ]]
    xrce_cmd = ['bash', '-lc', ['MicroXRCEAgent udp4 -p ', xrce_udp_port]]

    return LaunchDescription([
        DeclareLaunchArgument('px4_autopilot_path', default_value='/home/dreamslab/PX4-Autopilot'),
        DeclareLaunchArgument('px4_gz_world', default_value='sample_15016'),
        DeclareLaunchArgument('px4_make_target', default_value='gz_px4_gsplat'),
        DeclareLaunchArgument('xrce_udp_port', default_value='8888'),
        DeclareLaunchArgument('start_px4', default_value='true'),
        DeclareLaunchArgument('start_xrce_agent', default_value='true'),
        DeclareLaunchArgument('start_bridge', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='false',
                              description='Launch colmap_rviz_publisher + RViz2'),
        DeclareLaunchArgument('base_dir',
                              default_value=os.path.expanduser('~/photogrammetry_NBV/data/photogrammetry'),
                              description='Parent dir of unified_run_* folders; publisher auto-discovers the latest'),

        ExecuteProcess(cmd=xrce_cmd, output='screen', condition=IfCondition(start_xrce_agent)),
        ExecuteProcess(cmd=px4_cmd, output='screen', condition=IfCondition(start_px4)),

        Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='photogrammetry_bridge', output='screen',
            parameters=[{'config_file': bridge_yaml}],
            condition=IfCondition(start_bridge),
        ),
        Node(
            package='ros_gz_image', executable='image_bridge',
            name='photogrammetry_image_bridge', output='screen',
            arguments=['/rgbd/image'],
            condition=IfCondition(start_bridge),
        ),
        Node(
            package='photogrammetry_nbv', executable='unified_controller_node',
            name='unified_controller_node', output='screen',
            parameters=[mission_yaml],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([pkg_share, 'launch', 'visualization.launch.py'])
            ]),
            launch_arguments={'base_dir': LaunchConfiguration('base_dir')}.items(),
            condition=IfCondition(LaunchConfiguration('start_rviz')),
        ),
    ])
