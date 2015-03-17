cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#SET(CMAKE_MODULE_PATH $ENV{LIB_SPIMAGE_CMAKE}; "$ENV{LIB_SPIMAGE_CMAKE}/cuda")
#"${CMAKE_SOURCE_DIR}/cmake/Modules/")

#SET(CMAKE_MODULE_PATH "/home/jing.liu/libspimage/libspimage/cmake/Modules")
message(${CMAKE_MODULE_PATH})

IF(CMAKE_COMPILER_IS_GNUCC)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

FIND_PACKAGE(SpImage REQUIRED)
FIND_PACKAGE (GSL REQUIRED)	
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(HDF5 REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_21")
cuda_add_executable(emc main.cpp rotations.c mpi_helper.cpp emc_helper.cpp timer_helper.cpp file_helper.cpp configs.cpp emc_cuda.cu emc_atomic.cu)
cuda_add_cufft_to_target(emc)
target_link_libraries(emc spimage mpi mpi_cxx config gsl gslcblas hdf5)


