import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mobile_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'cfg'), glob('cfg/*.cfg')),
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
          'mmwave_6843_driver = mobile_tracker.mmwave_6843_driver:main',
        ],
    },
)
