cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_utils)

add_library(fileutils STATIC
    file_utils.c
)
target_include_directories(fileutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if (DISABLE_RGA)
    add_definitions(-DDISABLE_RGA)
endif()

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

add_library(imageutils STATIC
    image_utils.c
)
target_include_directories(imageutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(imageutils
    ${LIBRGA}
)

target_include_directories(imageutils PUBLIC
    ${LIBRGA_INCLUDES}
)