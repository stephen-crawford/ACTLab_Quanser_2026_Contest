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
            'config/modules.yaml',
        ]),
        ('share/' + package_name + '/config/presets', [
            'config/presets/default.yaml',
            'config/presets/legacy_2025.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/mpcc_mission_launch.py',
        ]),
        ('share/' + package_name + '/scripts', [
            'scripts/run_stage1.sh',
            'scripts/run_stage1_dev.sh',
            'scripts/rebuild_docker.sh',
        ]),
        ('share/' + package_name + '/models', [
            'models/.gitkeep',
        ]),
        ('share/' + package_name + '/training', [
            'training/data.yaml',
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
            'pose_logger = acc_stage1_mission.pose_logger:main',
            'yolo_bridge = acc_stage1_mission.yolo_bridge:main',
            'dashboard = acc_stage1_mission.dashboard:main',
            'path_overlay = acc_stage1_mission.path_overlay:main',
            'capture_data = training.capture_data:main',
        ],
    },
)
