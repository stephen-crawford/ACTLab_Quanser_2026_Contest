cmake_minimum_required(VERSION 3.8)
project(acc_mpcc_controller_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

set(CMAKE_CXX_STANDARD 17)

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(qcar2_interfaces REQUIRED)

# Find Eigen3, OpenCV, yaml-cpp
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# =============================================================================
# acados MPCC solver
# =============================================================================
set(ACADOS_DIR "$ENV{ACADOS_INSTALL_DIR}" CACHE PATH "acados install directory")
if(NOT ACADOS_DIR)
  # Try common locations: host install, then Docker workspace
  if(EXISTS "/home/stephen/acados/include")
    set(ACADOS_DIR "/home/stephen/acados")
  elseif(EXISTS "/workspaces/isaac_ros-dev/acados/include")
    set(ACADOS_DIR "/workspaces/isaac_ros-dev/acados")
  else()
    message(FATAL_ERROR "acados not found. Set ACADOS_INSTALL_DIR or install to /home/stephen/acados")
  endif()
endif()
message(STATUS "Using acados from: ${ACADOS_DIR}")

# acados generated solver code
set(ACADOS_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/acados_ocp/c_generated_code")

# Collect all generated C source files
file(GLOB ACADOS_MODEL_SRCS "${ACADOS_GENERATED_DIR}/mpcc_qcar2_model/*.c")
file(GLOB ACADOS_COST_SRCS "${ACADOS_GENERATED_DIR}/mpcc_qcar2_cost/*.c")
file(GLOB ACADOS_CONSTR_SRCS "${ACADOS_GENERATED_DIR}/mpcc_qcar2_constraints/*.c")

add_library(acados_mpcc_generated STATIC
  ${ACADOS_GENERATED_DIR}/acados_solver_mpcc_qcar2.c
  ${ACADOS_MODEL_SRCS}
  ${ACADOS_COST_SRCS}
  ${ACADOS_CONSTR_SRCS}
)

target_include_directories(acados_mpcc_generated PUBLIC
  ${ACADOS_GENERATED_DIR}
  ${ACADOS_DIR}/include
  ${ACADOS_DIR}/include/acados
  ${ACADOS_DIR}/include/blasfeo/include
  ${ACADOS_DIR}/include/hpipm/include
)

target_compile_definitions(acados_mpcc_generated PUBLIC ACADOS_WITH_QPOASES)
target_link_directories(acados_mpcc_generated PUBLIC ${ACADOS_DIR}/lib)
target_link_libraries(acados_mpcc_generated acados hpipm blasfeo qpOASES_e m)

# Ensure all executables linking acados can find shared libs at runtime.
# Use -Wl,--disable-new-dtags to force RPATH (inherited by transitive deps)
# instead of RUNPATH (which only applies to direct deps).
set(CMAKE_BUILD_RPATH "${ACADOS_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${ACADOS_DIR}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
add_link_options("-Wl,--disable-new-dtags")

# =============================================================================
# acc_core - Static library for non-ROS computational code
# =============================================================================
add_library(acc_core STATIC
  road_graph.cpp
  road_boundaries.cpp
)

target_include_directories(acc_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(acc_core
  yaml-cpp
)

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

target_link_libraries(mpcc_controller_node
  acc_core
  acados_mpcc_generated
)

ament_target_dependencies(mpcc_controller_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  sensor_msgs
  visualization_msgs
  tf2_ros
  tf2
  ament_index_cpp
  qcar2_interfaces
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
  geometry_msgs
  cv_bridge
  tf2_ros
  tf2
  tf2_geometry_msgs
)

# =============================================================================
# Odom From TF Node
# =============================================================================
add_executable(odom_from_tf_node
  odom_from_tf_node.cpp
)

target_include_directories(odom_from_tf_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

ament_target_dependencies(odom_from_tf_node
  rclcpp
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2
)

# =============================================================================
# Mission Manager Node
# =============================================================================
add_executable(mission_manager_node
  mission_manager_node.cpp
)

target_include_directories(mission_manager_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(mission_manager_node
  acc_core
)

ament_target_dependencies(mission_manager_node
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  nav2_msgs
  std_msgs
  tf2_ros
  tf2
  rcl_interfaces
  action_msgs
  ament_index_cpp
)

# =============================================================================
# State Estimator Node (EKF)
# =============================================================================
add_executable(state_estimator_node
  state_estimator_node.cpp
)

target_include_directories(state_estimator_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(state_estimator_node
  rclcpp
  nav_msgs
  std_msgs
  sensor_msgs
  tf2_ros
  tf2
)

# =============================================================================
# Obstacle Tracker Node (Multi-class Kalman + Lidar)
# =============================================================================
add_executable(obstacle_tracker_node
  obstacle_tracker_node.cpp
)

target_include_directories(obstacle_tracker_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(obstacle_tracker_node
  rclcpp
  std_msgs
  sensor_msgs
  tf2_ros
  tf2
)

# =============================================================================
# Traffic Light Map Node
# =============================================================================
add_executable(traffic_light_map_node
  traffic_light_map_node.cpp
)

target_include_directories(traffic_light_map_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(traffic_light_map_node
  acc_core
)

ament_target_dependencies(traffic_light_map_node
  rclcpp
  std_msgs
  tf2_ros
  tf2
  ament_index_cpp
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS
  mpcc_controller_node
  sign_detector_node
  odom_from_tf_node
  mission_manager_node
  state_estimator_node
  obstacle_tracker_node
  traffic_light_map_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY ../config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
