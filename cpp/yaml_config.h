/**
 * yaml-cpp helper utilities.
 */

#pragma once

#include <yaml-cpp/yaml.h>
#include <string>

namespace acc {

inline double yaml_double(const YAML::Node& node, const std::string& key, double def) {
    if (node[key]) return node[key].as<double>(def);
    return def;
}

inline int yaml_int(const YAML::Node& node, const std::string& key, int def) {
    if (node[key]) return node[key].as<int>(def);
    return def;
}

inline std::string yaml_str(const YAML::Node& node, const std::string& key,
                            const std::string& def = "")
{
    if (node[key]) return node[key].as<std::string>(def);
    return def;
}

inline bool yaml_bool(const YAML::Node& node, const std::string& key, bool def) {
    if (node[key]) return node[key].as<bool>(def);
    return def;
}

}  // namespace acc
