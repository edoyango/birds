#ifndef _RKNN_MODEL_ZOO_FILE_UTILS_H_
#define _RKNN_MODEL_ZOO_FILE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read data from file
 * 
 * @param path [in] File path
 * @param out_data [out] Read data
 * @return int -1: error; > 0: Read data size
 */
int read_data_from_file(const char *path, char **out_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //_RKNN_MODEL_ZOO_FILE_UTILS_H_