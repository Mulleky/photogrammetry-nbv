from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('photogrammetry_nbv')
    phase2_yaml = PathJoinSubstitution([pkg_share, 'config', 'phase2_mission.yaml'])
    scoring_yaml = PathJoinSubstitution([pkg_share, 'config', 'scoring.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('seed_run_dir'),
        DeclareLaunchArgument('metrics_json'),
        DeclareLaunchArgument('output_dir', default_value='~/photogrammetry_NBV/data/photogrammetry/offline_eval'),
        ExecuteProcess(
            cmd=[
                'offline_phase2_eval',
                '--seed-run-dir', LaunchConfiguration('seed_run_dir'),
                '--metrics-json', LaunchConfiguration('metrics_json'),
                '--phase2-config', phase2_yaml,
                '--scoring-config', scoring_yaml,
                '--output-dir', LaunchConfiguration('output_dir'),
            ],
            output='screen',
        ),
    ])
