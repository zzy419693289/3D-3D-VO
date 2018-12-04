cmake_minimum_required(VERSION 2.8)
project(VO_SVD)

SET(CMAKE_BUILD_TYPE Release)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ZED REQUIRED)
find_package(CUDA 8.0 REQUIRED)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${Eigen3_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

include_directories(
${PROJECT_SOURCE_DIR}/include
${OPENCV_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${CUDA_INCLUDE_DIRS}
${ZED_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED
src/vo_svd.cc
src/CPU_ORB.cc
src/GPU_ORB.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(testmain testmain.cc)

target_link_libraries(testmain 
${PROJECT_NAME}
${ZED_LIBRARIES}
${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_npp_LIBRARY} 
)


