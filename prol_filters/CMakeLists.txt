cmake_minimum_required(VERSION 3.8)
project(prol_filters)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(DEPS
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

include_directories(include ${EIGEN3_INCLUDE_DIR})

# ── Kalman Filter node ──────────────────────────────────────────────────────
add_executable(kalman_filter_node src/kalman_filter_node.cpp)
ament_target_dependencies(kalman_filter_node ${DEPS})

# ── Extended Kalman Filter node ─────────────────────────────────────────────
add_executable(ekf_node src/ekf_node.cpp)
ament_target_dependencies(ekf_node ${DEPS})

# ── Particle Filter node ────────────────────────────────────────────────────
add_executable(particle_filter_node src/particle_filter_node.cpp)
ament_target_dependencies(particle_filter_node ${DEPS})

# ── Map publisher (publishes our map.yaml as a /map OccupancyGrid backdrop) ──
add_executable(map_publisher_node src/map_publisher_node.cpp)
ament_target_dependencies(map_publisher_node ${DEPS})

install(TARGETS
  kalman_filter_node
  ekf_node
  particle_filter_node
  map_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config include map
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/plot_results.py
  scripts/evaluate_filters.py
  scripts/imu_simulator.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
