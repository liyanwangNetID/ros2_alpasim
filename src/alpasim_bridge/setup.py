from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'alpasim_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml"),),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lab',
    maintainer_email='liyang08.wang@polyu.edu.hk',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    "console_scripts": [
        (
            "ego_state_publisher = "
            "alpasim_bridge.ego_state_publisher:main"
        ),
        (
            "actor_state_publisher = "
            "alpasim_bridge.actor_state_publisher:main"
        ),
        (
            "actor_marker_publisher = "
            "alpasim_bridge.actor_marker_publisher:main"
        ),
        (
            "map_server = "
            "alpasim_bridge.map_server:main"
        ),
        (
            "map_marker_publisher = "
            "alpasim_bridge.map_marker_publisher:main"
        ),
        (
            "ground_truth_trajectory_server = "
            "alpasim_bridge.ground_truth_trajectory_server:main"
        ),
        (
            "navigation_state_publisher = "
            "alpasim_bridge.navigation_state_publisher:main"
        ),
        (
            "ground_truth_future_publisher = "
            "alpasim_bridge.ground_truth_future_publisher:main"
        ),
        (
            "executed_path_publisher = "
            "alpasim_bridge.executed_path_publisher:main"
        ),
        (
            "navigation_marker_publisher = "
            "alpasim_bridge.navigation_marker_publisher:main"
        ),
    ],
},

)
