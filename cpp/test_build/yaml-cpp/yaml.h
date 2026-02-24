/**
 * Minimal yaml-cpp stub for compilation testing.
 * Only provides the interfaces used by road_boundaries.cpp and yaml_config.h.
 */
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace YAML {

class Node {
public:
    Node() = default;

    // Subscript access (returns by value)
    Node operator[](const std::string& key) const { (void)key; return Node(); }
    Node operator[](const char* key) const { (void)key; return Node(); }
    Node operator[](int idx) const { (void)idx; return Node(); }

    // Type checking
    explicit operator bool() const { return false; }
    bool IsDefined() const { return false; }
    bool IsNull() const { return true; }
    bool IsScalar() const { return false; }
    bool IsSequence() const { return false; }
    bool IsMap() const { return false; }

    // Value extraction
    template<typename T>
    T as() const { return T{}; }

    template<typename T>
    T as(const T& fallback) const { return fallback; }

    // Iteration support (empty - never iterates)
    const Node* begin() const { return nullptr; }
    const Node* end() const { return nullptr; }

    size_t size() const { return 0; }
};

inline Node LoadFile(const std::string& filename) {
    (void)filename;
    throw std::runtime_error("yaml-cpp stub: LoadFile not implemented");
    return Node();
}

inline Node Load(const std::string& input) {
    (void)input;
    return Node();
}

}  // namespace YAML
