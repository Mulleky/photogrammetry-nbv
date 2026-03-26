from setuptools import setup
from glob import glob
import os

package_name = 'photogrammetry_init'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Phase 1 deterministic four-view initialization mission for photogrammetry_NBV.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'four_view_init_node = photogrammetry_init.four_view_init_node:main',
        ],
    },
)
