cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_3rdparty_libs)

# rknn runtime (rknpu2)
set(LIBRKNNRT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/rknpu2 PARENT_SCOPE)

# rga
set(LIBRGA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/librga PARENT_SCOPE)
