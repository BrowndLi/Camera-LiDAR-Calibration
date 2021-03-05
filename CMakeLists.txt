#指定CMake最小版本
cmake_minimum_required(VERSION 2.8.3)

#设置项目名称
project(lidarPoint_plane)

#设置包含库目录
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

##设置包含库目录
#set(OpenCV_DIR /home/lifei/lib/opencv-3.4.5/opencv-3.4.5/build)
#set(OpenCV_INCLUDE_DIRS /home/lifei/lib/opencv-3.4.5/opencv-3.4.5/include)
#set(OpenCV_LIBRARIES_DIRS /home/lifei/lib/opencv-3.4.5/opencv-3.4.5/build/lib)

#查找指定的库
find_package(OpenCV 3.4.5 REQUIRED)
find_package(PCL 1.8.0 REQUIRED)

#输出相关信息
message(STATUS " OpenCV library status:")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS " PCL library status:")
message(STATUS " version: ${PCL_VERSION}")
message(STATUS " libraries: ${PCL_LIBS}")
message(STATUS " include path: ${PCL_INCLUDE_DIRS}")


#find_package(GTSAM REQUIRED QUIET)
#find_package(PCL REQUIRED QUIET)
#find_package(OpenCV REQUIRED QUIET)

#设置包含的目录//指定头文件路径
include_directories(
	include
	#${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	#${GTSAM_INCLUDE_DIR}
)

#设置链接库搜索目录
link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	#${GTSAM_LIBRARY_DIRS}
)

#生成可执行文件
add_executable(lidarPoint_plane lidarPoint_plane.cpp)

#指定target需要链接的库
target_link_libraries(lidarPoint_plane ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})