from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('photogrammetry_init')
    mission_yaml = PathJoinSubstitution([package_share, 'config', 'mission.yaml'])
    bridge_yaml = PathJoinSubstitution([package_share, 'config', 'bridge.yaml'])

    px4_autopilot_path = LaunchConfiguration('px4_autopilot_path')
    px4_gz_world = LaunchConfiguration('px4_gz_world')
    px4_make_target = LaunchConfiguration('px4_make_target')
    xrce_udp_port = LaunchConfiguration('xrce_udp_port')
    start_px4 = LaunchConfiguration('start_px4')
    start_xrce_agent = LaunchConfiguration('start_xrce_agent')
    start_bridge = LaunchConfiguration('start_bridge')

    px4_cmd = [
        'bash', '-lc',
        ['cd ', px4_autopilot_path,
         ' && PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD=', px4_gz_world,
         ' make px4_sitl ', px4_make_target]
    ]

    xrce_cmd = [
    	'bash', '-lc',
    	['/snap/bin/micro-xrce-dds-agent udp4 -p ', xrce_udp_port]
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'px4_autopilot_path',
            default_value='~/PX4-Autopilot',
            description='Absolute path to the local PX4-Autopilot checkout.',
        ),
        DeclareLaunchArgument(
            'px4_gz_world',
            default_value='sample_15016',
            description='Gazebo world passed to PX4_GZ_WORLD.',
        ),
        DeclareLaunchArgument(
            'px4_make_target',
            default_value='gz_px4_gsplat',
            description='PX4 make target for the drone/world combination.',
        ),
        DeclareLaunchArgument(
            'xrce_udp_port',
            default_value='8888',
            description='Micro XRCE-DDS Agent UDP port.',
        ),
        DeclareLaunchArgument('start_px4', default_value='true'),
        DeclareLaunchArgument('start_xrce_agent', default_value='true'),
        DeclareLaunchArgument('start_bridge', default_value='true'),

        ExecuteProcess(
            cmd=px4_cmd,
            output='screen',
            condition=IfCondition(start_px4),
        ),

        ExecuteProcess(
            cmd=xrce_cmd,
            output='screen',
            condition=IfCondition(start_xrce_agent),
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='photogrammetry_bridge',
            output='screen',
            parameters=[{'config_file': bridge_yaml}],
            condition=IfCondition(start_bridge),
        ),
        
        Node(
    	    package='ros_gz_image',
            executable='image_bridge',
            name='photogrammetry_image_bridge',
            output='screen',
            arguments=['/rgbd/image'],
            condition=IfCondition(start_bridge),
        ),

        Node(
            package='photogrammetry_init',
            executable='four_view_init_node',
            name='four_view_init_node',
            output='screen',
            parameters=[mission_yaml],
        ),
    ])
