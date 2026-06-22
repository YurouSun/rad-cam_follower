import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mmwave_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        # Add config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zhenyu98',
    maintainer_email='kmrsywc@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'tracker_node = mmwave_follower.tracker_node:main',
          'tracker_node2 = mmwave_follower.tracker_node2:main',
          'auto_motion = mmwave_follower.auto_motion:main',
          'information_tracker = mmwave_follower.information_tracker:main',
          'car2_formation_node = mmwave_follower.car2_formation_node:main',
          'simple_avoidance_from_tracking_node = mmwave_follower.simple_avoidance_from_tracking_node:main',
          'angle_formation_node = mmwave_follower.angle_formation_node:main',
          'arc_avoidance_observer_node = mmwave_follower.arc_avoidance_observer_node:main',
          'arc_avoidance_observer_sim = mmwave_follower.arc_avoidance_observer_sim:main',
          'plot_arc_avoidance_log = mmwave_follower.plot_arc_avoidance_log:main',
          'wheel_odom_from_joint_states = mmwave_follower.wheel_odom_from_joint_states:main',
          'car3_bearing_formation_node = mmwave_follower.car3_bearing_formation_node:main',
          'trajectory_visualizer_node = mmwave_follower.trajectory_visualizer_node:main',
        ],
    },
)
