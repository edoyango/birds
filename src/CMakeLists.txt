cmake_minimum_required(VERSION 3.10)

project(birds_detector)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fPIC")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fPIC")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fPIC")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

# expecting librknrrt and librga to be external
find_library(LIBRKNNRT
    NAMES rknnrt
)
find_library(LIBRGA
    NAMES rga
)

# YOLO26 (NMS-free, graph cut before box decoding). yolo26/test is a host-side test harness, not part of the library.
add_library(${PROJECT_NAME} SHARED
    yolo26/model.cc
    yolo26/yolo26_postprocess.cc
    yolo26/yolo26.cc
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    ${LIBRKNNRT}
    ${LIBRGA}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/yolo26
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../birdector/detector )
