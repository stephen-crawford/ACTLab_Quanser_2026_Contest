from setuptools import find_packages, setup

package_name = 'acc_stage1_mission'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/mission.yaml',
            'config/road_boundaries.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/mission_launch.py',
            'launch/mission_with_detection_launch.py',
            'launch/full_mission_launch.py',
            'launch/mpcc_mission_launch.py',
        ]),
        ('share/' + package_name + '/scripts', [
            'scripts/run_stage1.sh',
            'scripts/run_stage1_dev.sh',
            'scripts/rebuild_docker.sh',
        ]),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='stephen',
    maintainer_email='stephen@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_manager = acc_stage1_mission.mission_manager:main',
            'pose_logger = acc_stage1_mission.pose_logger:main',
            'obstacle_detector = acc_stage1_mission.obstacle_detector:main',
            'yolo_bridge = acc_stage1_mission.yolo_bridge:main',
            'mpcc_controller = acc_stage1_mission.mpcc_controller:main',
            'odom_from_tf = acc_stage1_mission.odom_from_tf:main',
        ],
    },
)
