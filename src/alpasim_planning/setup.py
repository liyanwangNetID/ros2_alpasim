from setuptools import find_packages, setup

package_name = 'alpasim_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
                "synthetic_trajectory_planner = "
                "alpasim_planning.synthetic_trajectory_planner:main"
            ),
        ],
    },
)
