cmake_minimum_required(VERSION 3.8)
project(acc_mpcc_controller_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

set(CMAKE_CXX_STANDARD 17)

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(cv_bridge REQUIRED)

# Find Eigen3 and OpenCV
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# =============================================================================
# MPCC Controller Node
# =============================================================================
add_executable(mpcc_controller_node
  mpcc_controller_node.cpp
)

target_include_directories(mpcc_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(mpcc_controller_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  visualization_msgs
  tf2_ros
  tf2
)

# =============================================================================
# Sign Detector Node (HSV + contour, no ML)
# =============================================================================
add_executable(sign_detector_node
  sign_detector_node.cpp
)

target_include_directories(sign_detector_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sign_detector_node
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(sign_detector_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS
  mpcc_controller_node
  sign_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
