cmake_minimum_required(VERSION 3.8)
project(athome_navigation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY maps/  DESTINATION share/${PROJECT_NAME}/maps)
install(DIRECTORY rviz/  DESTINATION share/${PROJECT_NAME}/rviz)
ament_package()
