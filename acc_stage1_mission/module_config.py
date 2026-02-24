"""
Module configuration loader for ACC self-driving stack.

Loads config/modules.yaml and provides typed access to module backend
selections. Falls back to sensible defaults if the file is missing.

Presets (config/presets/<name>.yaml) provide named configurations that
override modules.yaml defaults. Use load_preset() to load a preset.

Usage:
    from acc_stage1_mission.module_config import load_module_config, load_preset
    config = load_module_config()
    print(config['detection']['backend'])  # 'auto'

    # Load the 2025 legacy preset
    preset = load_preset('legacy_2025')
    print(preset['controller']['backend'])  # 'casadi'
"""

import os
import yaml

# Defaults mirror config/modules.yaml — used when file is missing
_DEFAULTS = {
    'detection': {
        'backend': 'auto',
    },
    'path_planning': {
        'backend': 'experience_astar',
        'weighted_epsilon': 1.5,
    },
    'controller': {
        'backend': 'cpp',
    },
    'pedestrian_tracking': {
        'backend': 'kalman',
    },
}

_VALID_BACKENDS = {
    'detection': {'auto', 'hsv', 'yolo_coco', 'custom', 'hybrid', 'hough_hsv'},
    'path_planning': {'astar', 'dijkstra', 'weighted_astar', 'experience_astar'},
    'controller': {'auto', 'casadi', 'cpp', 'pure_pursuit'},
    'pedestrian_tracking': {'kalman', 'simple'},
}


def load_module_config(config_path=None):
    """Load module configuration from YAML.

    Args:
        config_path: Path to modules.yaml. If None, searches:
            1. Package share directory (installed)
            2. Source tree config/ (development)

    Returns:
        dict with validated module configuration.
    """
    if config_path is None:
        config_path = _find_config_file('modules.yaml')

    config = dict(_DEFAULTS)

    if config_path is not None and os.path.isfile(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            # Merge file config over defaults (one level deep)
            for section in _DEFAULTS:
                if section in file_config and isinstance(file_config[section], dict):
                    merged = dict(_DEFAULTS[section])
                    merged.update(file_config[section])
                    config[section] = merged
        except Exception:
            pass  # Fall back to defaults on any parse error

    # Validate backends
    for section, valid in _VALID_BACKENDS.items():
        backend = config.get(section, {}).get('backend', '')
        if backend not in valid:
            config[section]['backend'] = _DEFAULTS[section]['backend']

    return config


def load_preset(preset_name):
    """Load a named preset from config/presets/<name>.yaml.

    Presets can contain both launch-level parameters (use_cpp_controller,
    reference_velocity, etc.) and module backend overrides (detection,
    path_planning, controller, pedestrian_tracking sections).

    Args:
        preset_name: Name of the preset (without .yaml extension).
            Available presets: 'default', 'legacy_2025'

    Returns:
        dict with all preset values, or None if preset not found.
    """
    preset_path = _find_preset_file(preset_name)
    if preset_path is None:
        return None

    try:
        with open(preset_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def list_presets():
    """List available preset names.

    Returns:
        list of preset name strings (without .yaml extension).
    """
    presets = []
    for search_dir in _preset_search_dirs():
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if fname.endswith('.yaml'):
                    presets.append(fname[:-5])
    return sorted(set(presets))


def _find_config_file(filename):
    """Search for a config file in standard locations."""
    # 1. Package share directory (installed ROS2 package)
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('acc_stage1_mission')
        candidate = os.path.join(pkg_share, 'config', filename)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass

    # 2. Source tree (development)
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'config', filename)
    if os.path.isfile(source_dir):
        return os.path.abspath(source_dir)

    return None


def _preset_search_dirs():
    """Return directories to search for presets."""
    dirs = []
    # 1. Package share directory (installed)
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('acc_stage1_mission')
        dirs.append(os.path.join(pkg_share, 'config', 'presets'))
    except Exception:
        pass
    # 2. Source tree (development)
    dirs.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'presets')
    ))
    return dirs


def _find_preset_file(preset_name):
    """Search for a preset YAML file by name."""
    filename = f'{preset_name}.yaml'
    for search_dir in _preset_search_dirs():
        candidate = os.path.join(search_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    return None
