cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_3rdparty_libs)

# stb image
set(STB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/stb_image PARENT_SCOPE)

# rknn runtime (rknpu2)
set(LIBRKNNRT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/rknpu2 PARENT_SCOPE)

# rga
set(LIBRGA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/librga PARENT_SCOPE)

# libsndfile
set(LIBSNDFILE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/libsndfile PARENT_SCOPE)
set(LIBSNDFILE ${CMAKE_CURRENT_SOURCE_DIR}/lib/libsndfile/libsndfile.a PARENT_SCOPE)
