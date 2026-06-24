from setuptools import setup
from glob import glob
import os

package_name = 'photogrammetry_nbv'

setup(
    name=package_name,
    version='0.1.0',
    packages=[
        package_name,
        package_name + '.scorers',
        package_name + '.gt_supervision',
        package_name + '.adaptive',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'colmap_scripts'), glob('colmap_scripts/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Phase 2 sparse-cloud-driven active shape estimation mission for photogrammetry_NBV.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'phase2_controller_node = photogrammetry_nbv.phase2_controller_node:main',
            'unified_controller_node = photogrammetry_nbv.unified_controller_node:main',
            'offline_phase2_eval = photogrammetry_nbv.offline_phase2_eval:main',
            'colmap_rviz_publisher = photogrammetry_nbv.colmap_rviz_publisher:main',
        ],
    },
)
