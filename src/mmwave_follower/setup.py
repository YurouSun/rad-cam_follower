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
          'auto_motion = mmwave_follower.auto_motion:main',
          'angle_formation_node = mmwave_follower.angle_formation_node:main',
          'car3_bearing_formation_node = mmwave_follower.car3_bearing_formation_node:main',
        ],
    },
)
