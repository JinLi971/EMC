#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>
#include <cufft.h>
#include "emc_helper.h"
#include "emc_cuda_help.h"
#include <stdio.h>


__global__ void update_slices_kernel(real * images, real * slices, int * mask, real * respons,
                                     real * scaling, int * active_images, int N_images,
                                     int slice_start, int N_2d,
                                     real * slices_total_respons, real * rot,
                                     real * x_coord, real * y_coord, real * z_coord,
                                     real * model, int slice_rows, int slice_cols,
                                     int model_x, int model_y, int model_z);
__global__ void cuda_normalize_responsabilities_sum_kernel(real * respons,  real* d_sum,
                                                           real* max, int N_images, int allocate_slices);

__global__ void update_slices_final_kernel(real * images, real * slices, int * mask, real * respons,
                                           real * scaling, int * active_images, int N_images,
                                           int slice_start, int N_2d,
                                           real * slices_total_respons, real * rot,
                                           real * x_coord, real * y_coord, real * z_coord,
                                           real * model, real * weight,
                                           int slice_rows, int slice_cols,
                                           int model_x, int model_y, int model_z);

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
                                     real * scaling, int N_images, int N_2d,
                                     real * slices_total_respons, real * rot,
                                     real * x_coord, real * y_coord, real * z_coord,
                                     real * model, real * weight,
                                     int slice_rows, int slice_cols,
                                     int model_x, int model_y, int model_z);

__global__ void calculate_fit_kernel(real *slices, real *images, int *mask,
                                     real *respons, real *fit, real sigma,
                                     real *scaling, int N_2d, int slice_start);

__global__ void calculate_fit_best_rot_kernel(real *slices, real *images, int *mask,
                                              int *best_rot, real *fit,
                                              real *scaling, int N_2d, int slice_start);

__global__ void calculate_radial_fit_kernel(real *slices, real *images, int *mask,
                                            real *respons, real *scaling, real *radial_fit,
                                            real *radial_fit_weight, real *radius,
                                            int N_2d, int side, int slice_start);

/*sum up within a block along x*/
template<typename T>
__device__ void inblock_reduce(T * data){
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            data[threadIdx.x] += data[threadIdx.x + s];
        }
        __syncthreads();
    }
}
/*sum up within a block along y*/
template<typename T>
__device__ void inblock_reduce_y(T * data){
    __syncthreads();
    for(unsigned int s=blockDim.y/2; s>0; s>>=1){
        if (threadIdx.y < s){
            data[threadIdx.y] += data[threadIdx.y+s];
        }
        __syncthreads();
    }
}
/*find out the maximas in a block*/
template<typename T>
__device__ void inblock_maximum(T * data){
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            if(data[threadIdx.x] < data[threadIdx.x + s]){
                data[threadIdx.x] = data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
}
/*find out the maxima and its index within a block*/
template<typename T>
__device__ void inblock_maximum_index(T * data, int *index) {
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            if (data[threadIdx.x] < data[threadIdx.x + s]) {
                data[threadIdx.x] = data[threadIdx.x + s];
                index[threadIdx.x] = index[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
}

/*the small e step expansion from intensity model to tomographic model by using Quaternion*/
__device__ void cuda_get_slice(real *model, real *slice,
                               real *rot, real *x_coordinates,
                               real *y_coordinates, real *z_coordinates, int slice_rows,
                               int slice_cols, int model_x, int model_y, int model_z,
                               int tid, int step)
{
    const int x_max = slice_rows;
    const int y_max = slice_cols;
    //tabulate angle later
    real new_x, new_y, new_z;
    int round_x, round_y, round_z;
    real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
    real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
    real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
    real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
    real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
    real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
    real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
    real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
    real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
    for (int x = 0; x < x_max; x++) {
        for (int y = tid; y < y_max; y+=step) {
            /* This is just a matrix multiplication with rot */
            new_x = m00*x_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*z_coordinates[y*x_max+x];
            new_y = m10*x_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*z_coordinates[y*x_max+x];
            new_z = m20*x_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*z_coordinates[y*x_max+x];
            /* changed the next lines +0.5 -> -0.5 (11 dec 2012)*/
            round_x = lroundf(model_x/2.0f - 0.5f + new_x);
            round_y = lroundf(model_y/2.0f - 0.5f + new_y);
            round_z = lroundf(model_z/2.0f - 0.5f + new_z);
            if (round_x > 0 && round_x < model_x &&
                    round_y > 0 && round_y < model_y &&
                    round_z > 0 && round_z < model_z) {
                slice[y*x_max+x] = model[round_z*model_x*model_y + round_y*model_x + round_x];
            }else{
                slice[y*x_max+x] = -1.0f;
            }
        }
    }
    __syncthreads();
}

/* updated to use rotations with an offset start. */
__global__ void get_slices_kernel(real * model, real * slices, real *rot, real *x_coordinates,
                                  real *y_coordinates, real *z_coordinates, int slice_rows,
                                  int slice_cols, int model_x, int model_y, int model_z,
                                  int start_slice){
    int bid = blockIdx.x;
    int i_slice = bid;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int N_2d = slice_rows*slice_cols;
    cuda_get_slice(model,&slices[N_2d*i_slice],&rot[4*(start_slice+i_slice)],x_coordinates,
            y_coordinates,z_coordinates,slice_rows,slice_cols,model_x,model_y,
            model_z,tid,step);
}

/* This responsability does not yet take scaling of patterns into accoutnt. */
__device__ void cuda_calculate_responsability_absolute(real *slice, real *image, int *mask, real sigma, real scaling, int N_2d, int tid, int step, real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f) {
            sum += pow(slice[i] - image[i]/scaling,2);
            count++;
        }
    }
    __syncthreads();
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

/*calculate the raltive respons function called by calculate_responsability_kernel*/
__device__ void cuda_calculate_responsability_relative(real *slice, real *image, int *mask, real sigma, real scaling, int N_2d, int tid, int step, real *sum_cache, int *count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.f) {
            sum += pow((slice[i] - image[i]/scaling) / slice[i], 2);
            count++;
        }
    }
    __syncthreads();
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}


/* This responsability does not yet take scaling of patterns into accoutnt. */
__device__ void cuda_calculate_responsability_poisson(real *slice, real *image, int *mask, real sigma, real scaling, int N_2d, int tid, int step, real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f) {
            //sum += pow((slice[i] - image[i]/scaling) / (sqrt(image[i])+0.4), 2);
            //sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]+0.02), 2); // 0.2 worked. this was used latest
            sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+0.02), 2); // 0.2 worked. this was used latest
            //sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]/0.5+10.0), 2); // 0.2 worked
            //sum += pow((slice[i]*scaling - image[i])/8.0/ (sqrt(image[i]/8.0 + 1.0)), 2); // 0.2 worked
            count++;
        }
    }
    __syncthreads();

    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_responsability_true_poisson(float *slice, float *image,
                                                           int *mask, real sigma, real scaling,
                                                           int N_2d, int tid, int step,
                                                           real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f) {
            sum += pow((slice[i]*scaling - image[i]) / 8.0, 2) / (image[i]/8.0 + 0.1) / 2.0;
            //sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+1.0), 2);
            count++;
        }
    }
    __syncthreads();

    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

/* Now takes a starting slice. Otherwise unchanged */
/*only respons is changed, the respons is in log space, and waiting for normalisation*/
__global__ void calculate_responsabilities_kernel(real * slices, real * images, int * mask,
                                                  real sigma, real * scaling, real * respons, real *weights,
                                                  int N_2d, int slice_start, enum diff_type diff){
    __shared__ real sum_cache[TNUM];
    __shared__ int count_cache[TNUM];
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    if (diff == relative) {
        cuda_calculate_responsability_relative(&slices[i_slice*N_2d],
                &images[i_image*N_2d],mask,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid,step,
                sum_cache,count_cache);
    } else if (diff == poisson) {
        cuda_calculate_responsability_poisson(&slices[i_slice*N_2d],
                &images[i_image*N_2d],mask,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid,step,
                sum_cache,count_cache);
    } else if (diff == absolute) {
        /* This one was used for best result so far.*/
        cuda_calculate_responsability_absolute(&slices[i_slice*N_2d],
                &images[i_image*N_2d],mask,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid,step,
                sum_cache,count_cache);
    }
    inblock_reduce(sum_cache);
    inblock_reduce(count_cache);
    if(tid == 0 ){
        //respons[(slice_start+i_slice)*N_images+i_image] = -sum_cache[0]/2.0/(real)count_cache[0]/pow(sigma,2);
        respons[(slice_start+i_slice)*N_images+i_image] = log(weights[slice_start+i_slice]) - sum_cache[0]/2.0/(real)count_cache[0]/pow(sigma,2);
    }
}


/* Now takes start slice and slice chunk. Also removed memcopy, done separetely later. */
void cuda_calculate_responsabilities(real * d_slices, real * d_images, int * d_mask,
                                     real sigma, real * d_scaling, real * d_respons, real *d_weights,
                                     int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff){


    dim3 nblocks(N_images,slice_chunk);
    int nthreads = TNUM;
    calculate_responsabilities_kernel<<<nblocks,nthreads>>>(d_slices, d_images, d_mask,
                                                            sigma, d_scaling, d_respons, d_weights,
                                                            N_2d, slice_start, diff);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (calc resp): %s\n",cudaGetErrorString(status));
    }

}

/*the sum of respons, checksum*/
void cuda_calculate_responsabilities_sum(real * respons, real * d_respons, int N_slices,
                                         int N_images){
    cudaMemcpy(respons,d_respons,sizeof(real)*N_slices*N_images,cudaMemcpyDeviceToHost);
    real respons_sum = 0;
    for(int i = 0;i<N_slices*N_images;i++){
        respons_sum += respons[i];
    }
    printf("respons_sum = %f\n",respons_sum);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (resp sum): %s\n",cudaGetErrorString(status));
    }
}

__global__ void calculate_weighted_power_kernel(real * images, real * slices, int * mask,
                                                real *respons, real * weighted_power, int N_images,
                                                int slice_start, int slice_chunk, int N_2d) {
    __shared__ real correlation[TNUM];
    //__shared__ int count[TNUM];
    int step = blockDim.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int i_image = bid;
    for (int i_slice = 0; i_slice < slice_chunk; i_slice++) {
        correlation[tid] = 0.0;
        //count[tid] = 0;
        for (int i = tid; i < N_2d; i+=step) {
            if (mask[i] != 0 && slices[i_slice*N_2d+i] > 0.0f) {
                correlation[tid] += images[i_image*N_2d+i]*slices[i_slice*N_2d+i];
                //correlation[tid] += images[i_image*N_2d+i]/slices[i_slice*N_2d+i];
                //count[tid] += 1;
            }
        }
        inblock_reduce(correlation);
        //inblock_reduce(count);
        if(tid == 0){
            weighted_power[i_image] += respons[(slice_start+i_slice)*N_images+i_image]*correlation[tid];
            //weighted_power[i_image] += correlation[tid]/count[tid]*respons[(slice_start+i_slice)*N_images+i_image];
        }
    }
}

__global__ void slice_weighting_kernel(real * images,int * mask,
                                       real * scaling, real *weighted_power,
                                       int N_slices, int N_2d){
    __shared__ real image_power[TNUM];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = bid;
    image_power[tid] = 0.0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] != 0) {
            image_power[tid] += pow(images[i_image*N_2d+i],2);
        }
    }
    inblock_reduce(image_power);

    if(tid == 0){
        scaling[i_image] = image_power[tid]/weighted_power[i_image];
        //scaling[i_image] = weighted_power[i_image];
    }
}

void cuda_update_weighted_power(real * d_images, real * d_slices, int * d_mask,
                                real * d_respons, real * d_weighted_power, int N_images,
                                int slice_start, int slice_chunk, int N_2d) {


    int nblocks = N_images;
    int nthreads = TNUM;
    calculate_weighted_power_kernel<<<nblocks,nthreads>>>(d_images,d_slices,d_mask,
                                                          d_respons,d_weighted_power, N_images,
                                                          slice_start,slice_chunk,N_2d);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error: %s\n",cudaGetErrorString(status));
    }

}

void cuda_update_scaling(real * d_images, int * d_mask,
                         real * d_scaling, real *d_weighted_power, int N_images,
                         int N_slices, int N_2d, real * scaling){
    int nblocks = N_images;
    int nthreads = TNUM;

    slice_weighting_kernel<<<nblocks,nthreads>>>(d_images,d_mask,d_scaling,
                                                 d_weighted_power,N_slices,N_2d);
    cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (update scaling): %s\n",cudaGetErrorString(status));
    }
}

__global__ void calculate_best_rotation_kernel(real *respons, int *best_rotation, int N_slices) {
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int N_images = gridDim.x;

    __shared__ real max_resp[TNUM];
    __shared__ int max_index[TNUM];
    max_resp[tid] = -1.e100;
    max_index[tid] = 0;
    real this_resp;
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        this_resp = respons[i_slice*N_images+i_image];
        if (this_resp > max_resp[tid]) {
            //printf("new best resp found at %d\n", i_slice);
            max_resp[tid] = this_resp;
            max_index[tid] = i_slice;
            //printf("max_index set to %d\n", max_index[tid]);
        }
    }
    inblock_maximum_index(max_resp, max_index);
    if (tid == 0) {
        best_rotation[i_image] = max_index[0];

    }
}

__device__ real calculate_scaling_poisson(real *image, real *slice, int *mask, int N_2d, int tid, int step){
    __shared__ real sum_cache[TNUM];
    __shared__ int weight_cache[TNUM];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10) {
            sum_cache[tid] += image[i]*image[i]/slice[i];
            weight_cache[tid] += image[i];
        }
    }
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    __syncthreads();
    return sum_cache[0] / weight_cache[0];
}

__device__ real calculate_scaling_absolute(real *image, real *slice, int *mask, int N_2d, int tid, int step){
    __shared__ real sum_cache[TNUM];
    __shared__ int weight_cache[TNUM];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10 && image[i] > 1.e-10) {
            sum_cache[tid] += image[i]*image[i];
            weight_cache[tid] += image[i]*slice[i];
        }
    }
    __syncthreads();
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    return sum_cache[0] / weight_cache[0];

}



__device__ real calculate_scaling_relative(real *image, real *slice, int *mask, int N_2d, int tid, int step){
    __shared__ real sum_cache[TNUM];
    __shared__ int weight_cache[TNUM];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10) {
            //if (mask[i] > 0) {
            /*
      sum_cache[tid] += image[i] / slice[i];
      weight_cache[tid] += 1.;
      */
            sum_cache[tid] += image[i]*image[i]/(slice[i]*slice[i]);
            weight_cache[tid] += image[i]/slice[i];
        }
    }
    __syncthreads();
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    return sum_cache[0] / weight_cache[0];
}

__global__ void update_scaling_best_kernel(real *scaling, real *images, real *model, int *mask, real *rotations,
                                           real *x_coordinates, real *y_coordinates, real *z_coordinates,
                                           int side, int *best_rotation){
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    const int N_2d = side*side;
    extern __shared__ real this_slice[];

    cuda_get_slice(model, this_slice, &rotations[4*best_rotation[i_image]],
            x_coordinates, y_coordinates, z_coordinates,
            side, side, side, side, side, tid, step);


    real this_scaling = calculate_scaling_poisson(&images[N_2d*i_image], this_slice, mask, N_2d, tid, step);
    if (tid == 0) {
        scaling[i_image] = this_scaling;
    }
}

void cuda_update_scaling_best(real *d_images, int *d_mask,
                              real *d_model, real *d_scaling, real *d_respons, real *d_rotations,
                              real *x_coordinates, real *y_coordinates, real *z_coordinates,
                              int N_images, int N_slices, int side, real *scaling) {
    int nblocks = N_images;
    int nthreads = TNUM;
    const int N_2d = side*side;
    int *d_best_rotation;
    cudaMalloc(&d_best_rotation, N_images*sizeof(int));
    calculate_best_rotation_kernel<<<nblocks, nthreads>>>(d_respons, d_best_rotation, N_slices);
    nthreads = TNUM;
    nblocks = N_images;
    update_scaling_best_kernel<<<nblocks,nthreads,N_2d*sizeof(real)>>>(d_scaling, d_images, d_model, d_mask, d_rotations, x_coordinates, y_coordinates, z_coordinates, side, d_best_rotation);
    cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);
}

__global__ void update_scaling_full_kernel(real *images, real *slices, int *mask, real *scaling, int N_2d, int slice_start, enum diff_type diff) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int i_image = blockIdx.x;
    const int i_slice = blockIdx.y;
    const int N_images = gridDim.x;
    // printf(" global scaling full, %d %d %d %d %d \n",tid,step,i_image,i_slice,N_images);
    real this_scaling;
    //    printf(" global scaling full, %d %d %d %d %d \n",tid,step,i_image,i_slice,N_images);

    this_scaling = calculate_scaling_absolute(&images[N_2d*i_image], &slices[N_2d*i_slice], mask, N_2d, tid, step);
    __syncthreads();
    if (tid == 0) {
        //     printf("scaling is %f at %d \n", this_scaling,(slice_start+i_slice)*N_images+i_image);
        scaling[(slice_start+i_slice)*N_images+i_image] = this_scaling;
    }
}

void cuda_update_scaling_full(real *d_images, real *d_slices, int *d_mask, real *d_scaling,
                              int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff) {
    dim3 nblocks(N_images,slice_chunk);
    int nthreads = TNUM;
    update_scaling_full_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_scaling, N_2d, slice_start, diff);
}

/* function now takes a start slice and a number of slices to retrieve */
void cuda_get_slices(sp_3matrix * model, real * d_model, real * d_slices, real * d_rot,
                     real * d_x_coordinates, real * d_y_coordinates,
                     real * d_z_coordinates, int start_slice, int slice_chunk){

    int rows = sp_3matrix_x(model);
    int cols = sp_3matrix_y(model);
    int N_2d = sp_3matrix_x(model)*sp_3matrix_y(model);
    int nblocks = slice_chunk;
    int nthreads = TNUM;
    get_slices_kernel<<<nblocks,nthreads>>>(d_model, d_slices, d_rot,d_x_coordinates,
                                            d_y_coordinates,d_z_coordinates,
                                            rows,cols,
                                            sp_3matrix_x(model),sp_3matrix_y(model),
                                            sp_3matrix_z(model), start_slice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (get slices): %s\n",cudaGetErrorString(status));
    }
}

void cuda_update_slices(real * d_images, real * d_slices, int * d_mask,
                        real * d_respons, real * d_scaling, int * d_active_images, int N_images,
                        int slice_start, int slice_chunk, int N_2d,
                        sp_3matrix * model, real * d_model,
                        real *d_x_coordinates, real *d_y_coordinates,
                        real *d_z_coordinates, real *d_rot,
                        real * d_weight, sp_matrix ** images){
    dim3 nblocks = slice_chunk;//N_slices;
    int nthreads = TNUM;
    real * d_slices_total_respons;
    cudaMalloc(&d_slices_total_respons,sizeof(real)*slice_chunk);
    update_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                               d_scaling, d_active_images, N_images, slice_start, N_2d,
                                               d_slices_total_respons, d_rot,d_x_coordinates,
                                               d_y_coordinates,d_z_coordinates,d_model,
                                               sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));
    cudaThreadSynchronize();
    insert_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                               d_scaling, N_images, N_2d,
                                               d_slices_total_respons, d_rot,d_x_coordinates,
                                               d_y_coordinates,d_z_coordinates,d_model, d_weight,
                                               sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));

    cudaFree(d_slices_total_respons);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (update slices): %s\n",cudaGetErrorString(status));
    }
}

void cuda_update_slices_final(real * d_images, real * d_slices, int * d_mask,
                              real * d_respons, real * d_scaling, int * d_active_images, int N_images,
                              int slice_start, int slice_chunk, int N_2d,
                              sp_3matrix * model, real * d_model,
                              real *d_x_coordinates, real *d_y_coordinates,
                              real *d_z_coordinates, real *d_rot,
                              real * d_weight, sp_matrix ** images){
    dim3 nblocks = slice_chunk;//N_slices;
    int nthreads = TNUM;
    real * d_slices_total_respons;
    cudaMalloc(&d_slices_total_respons,sizeof(real)*slice_chunk);
    update_slices_final_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                                     d_scaling, d_active_images, N_images, slice_start, N_2d,
                                                     d_slices_total_respons, d_rot,d_x_coordinates,
                                                     d_y_coordinates,d_z_coordinates,d_model, d_weight,
                                                     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));

    cudaThreadSynchronize();
    insert_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                               d_scaling, N_images, N_2d,
                                               d_slices_total_respons, d_rot,d_x_coordinates,
                                               d_y_coordinates,d_z_coordinates,d_model, d_weight,
                                               sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (update slices): %s\n",cudaGetErrorString(status));
    }
}

real cuda_model_max(real * model, int model_size){
    thrust::device_ptr<real> p(model);
    real max = thrust::reduce(p, p+model_size, real(0), thrust::maximum<real>());
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_model_max): %s\n",cudaGetErrorString(status));
    }
    return max;
}

__global__ void model_average_kernel(real *model, int model_size, real *average) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    //const int i1 = blockIdx.x;
    __shared__ real sum_cache[TNUM];
    __shared__ int weight_cache[TNUM];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < model_size; i+=step) {
        if (model[i] > 0.) {
            sum_cache[tid] += model[i];
            weight_cache[tid] += 1;
        }
    }
    __syncthreads();
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    if (tid == 0) {
        if(weight_cache[0]>0.0f)
            *average = sum_cache[0] / weight_cache[0];
        else *average =0;
    }
}

real cuda_model_average(real * model, int model_size) {
    real *d_average;
    cudaMalloc(&d_average, sizeof(real));
    model_average_kernel<<<1,TNUM>>>(model, model_size, d_average);
    real average;
    cudaMemcpy(&average, d_average, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_average);
    return average;
}

void cuda_allocate_slices(real ** slices, int side, int N_slices){
    //cudaSetDevice(2);
    cudaMalloc(slices,sizeof(real)*side*side*N_slices);
}

void cuda_allocate_model(real ** d_model, sp_3matrix * model){
    cudaMalloc(d_model,sizeof(real)*sp_3matrix_size(model));
    cudaMemcpy(*d_model,model->data,sizeof(real)*sp_3matrix_size(model),cudaMemcpyHostToDevice);
}

void cuda_copy_model_2_device (real ** d_model, sp_3matrix * model){
    printf("copy model to device ...");
    cudaMemcpy(*d_model,model->data,sizeof(real)*sp_3matrix_size(model),cudaMemcpyHostToDevice);
    printf("done!\n");
}

void cuda_allocate_mask(int ** d_mask, sp_imatrix * mask){
    cudaMalloc(d_mask,sizeof(int)*sp_imatrix_size(mask));
    cudaMemcpy(*d_mask,mask->data,sizeof(int)*sp_imatrix_size(mask),cudaMemcpyHostToDevice);

}

void cuda_allocate_rotations(real ** d_rotations, Quaternion ** rotations, int start, int end, int taskid){
    cudaMalloc(d_rotations,sizeof(real)*4*(end-start));
    for(int i = 0;i<end-start ;i++){
        cudaMemcpy(&((*d_rotations)[4*i]),rotations[i+start]->q,sizeof(real)*4,cudaMemcpyHostToDevice);

    }
}

void cuda_allocate_images(real ** d_images, sp_matrix ** images,  int N_images){

    cudaMalloc(d_images,sizeof(real)*sp_matrix_size(images[0])*N_images);
    for(int i = 0;i<N_images;i++){
        cudaMemcpy(&(*d_images)[sp_matrix_size(images[0])*i],images[i]->data,sizeof(real)*sp_matrix_size(images[0]),cudaMemcpyHostToDevice);
    }
}

void cuda_allocate_masks(int ** d_images, sp_imatrix ** images,  int N_images){

    cudaMalloc(d_images,sizeof(int)*sp_imatrix_size(images[0])*N_images);
    for(int i = 0;i<N_images;i++){
        cudaMemcpy(&(*d_images)[sp_imatrix_size(images[0])*i],images[i]->data,sizeof(int)*sp_imatrix_size(images[0]),cudaMemcpyHostToDevice);
    }
}


void cuda_allocate_coords(real ** d_x, real ** d_y, real ** d_z, sp_matrix * x,
                          sp_matrix * y, sp_matrix * z){
    cudaMalloc(d_x,sizeof(real)*sp_matrix_size(x));
    cudaMalloc(d_y,sizeof(real)*sp_matrix_size(x));
    cudaMalloc(d_z,sizeof(real)*sp_matrix_size(x));
    cudaMemcpy(*d_x,x->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
    cudaMemcpy(*d_y,y->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
    cudaMemcpy(*d_z,z->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
}

void cuda_reset_model(sp_3matrix * model, real * d_model){
    cudaMemset(d_model,0,sizeof(real)*sp_3matrix_size(model));
}

void cuda_copy_model(sp_3matrix * model, real *d_model){
    cudaMemcpy(model->data,d_model,sizeof(real)*sp_3matrix_size(model),cudaMemcpyDeviceToHost);
}

void cuda_output_device_model(real *d_model, char *filename, int side) {
    real *model = (real *)malloc(side*side*side*sizeof(real));
    cuda_copy_real_to_host(model, d_model, side*side*side);
    Image *model_out = sp_image_alloc(side, side, side);
    for (int i = 0; i < side*side*side; i++) {
        if (model[i] >= 0.) {
            model_out->image->data[i] = sp_cinit(model[i], 0.);
            model_out->mask->data[i] = 1;
        } else {
            //model_out->image->data[i] = sp_cinit(0., 0.);
            model_out->image->data[i] = sp_cinit(model[i], 0.);
            model_out->mask->data[i] = 0;
        }
    }
    sp_image_write(model_out, filename, 0);
    free(model);
    sp_image_free(model_out);
}


__global__ void cuda_divide_model_kernel(real * model, real * weight, int n){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        if(weight[i] > 0.0f){
            model[i] /= weight[i];
        }else{
            //model[i] = 0.0f;
            model[i] = -1.f;
        }
    }
}

__global__ void cuda_mask_out_model_kernel(real *model, real *weight, int n){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        if(weight[i] <= 0.0f){
            model[i] = -1.0f;
        }
    }
}

void cuda_divide_model_by_weight(sp_3matrix * model, real * d_model, real * d_weight){
    int n = sp_3matrix_size(model);
    int nthreads = TNUM;
    int nblocks = (n+nthreads-1)/nthreads;
    cuda_divide_model_kernel<<<nblocks,nthreads>>>(d_model,d_weight,n);
    cudaThreadSynchronize();
    cuda_mask_out_model_kernel<<<nblocks,nthreads>>>(d_model,d_weight,n);
}

void cuda_normalize_model(sp_3matrix *model, real *d_model) {
    int n = sp_3matrix_size(model);
    thrust::device_ptr<real> p(d_model);
    real model_average = cuda_model_average(d_model, sp_3matrix_size(model));
    printf("model average before normalization = %g\n", model_average);
    //real model_sum = thrust::reduce(p, p+n, real(0), thrust::plus<real>());
    //model_sum /= (real) n;
    thrust::transform(p, p+n,thrust::make_constant_iterator(1.0f/model_average), p, thrust::multiplies<real>());
    model_average = cuda_model_average(d_model, sp_3matrix_size(model));
    printf("model average after normalization = %g\n", model_average);
}

void cuda_print_device_info() {
    int i_device = cuda_get_device();
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i_device);

    printf("Name: %s device %d\n", properties.name, i_device);
    printf("Compute Capability: %d.%d\n", properties.major, properties.minor);
    printf("Memory: %g GB\n", properties.totalGlobalMem/(1024.*1024.*1024.));
    printf("Number of cores: %d\n", 8*properties.multiProcessorCount);

}

int cuda_get_best_device() {
    int N_devices;
    cudaDeviceProp properties;
    cudaGetDeviceCount(&N_devices);
    int core_count = 0;
    int best_device = 0;
    for (int i_device = 0; i_device < N_devices; i_device++) {
        cudaGetDeviceProperties(&properties, i_device);
        if (properties.multiProcessorCount > core_count) {
            best_device = i_device;
            core_count = properties.multiProcessorCount;
        }
    }
    return best_device;
    //cuda_set_device(best_device);

    /* should use cudaSetValidDevices() instead */
}


int compare(const void *a, const void *b) {
    return *(int*)b - *(int*)a;
}

/* this function is much safer than cuda_get_best_device() since it works together
   with exclusive mode */
void cuda_choose_best_device() {
    int N_devices;
    cudaDeviceProp properties;
    cudaGetDeviceCount(&N_devices);
    int *core_count = (int *)malloc(N_devices*sizeof(int));
    int **core_count_pointers = (int **)malloc(N_devices*sizeof(int *));
    for (int i_device = 0; i_device < N_devices; i_device++) {
        cudaGetDeviceProperties(&properties, i_device);
        core_count[i_device] = properties.multiProcessorCount;
        core_count_pointers[i_device] = &core_count[i_device];
    }

    qsort(core_count_pointers, N_devices, sizeof(core_count_pointers[0]), compare);
    int *device_priority = (int *)malloc(N_devices*sizeof(int));
    for (int i_device = 0; i_device < N_devices; i_device++) {
        device_priority[i_device] = (int) (core_count_pointers[i_device] - core_count);
    }
    cudaSetValidDevices(device_priority, N_devices);
    free(core_count_pointers);
    free(core_count);
    free(device_priority);
}

int cuda_get_device() {
    int i_device;
    cudaGetDevice(&i_device);
    return i_device;
}

void cuda_set_device(int i_device, int taskid) {
    cudaSetDevice(i_device);
    printf("cuda set device %d success! at task id %d \n",i_device, taskid);
}

void cuda_allocate_real(real ** x, int n){
    cudaMalloc(x,n*sizeof(real));
}

void cuda_allocate_int(int ** x, int n){
    cudaMalloc(x,n*sizeof(real));
}

void cuda_set_to_zero(real * x, int n){
    cudaMemset(x,0.0,sizeof(real)*n);
}

void cuda_copy_real_to_device(real *x, real *d_x, int n){
    cudaMemcpy(d_x,x,n*sizeof(real),cudaMemcpyHostToDevice);
}

void cuda_copy_real_to_device(real* x, real* d_x, int start, int n){
    cudaMemcpy(&d_x[start],x,n*sizeof(real),cudaMemcpyHostToDevice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_real_to_real_to device: copy): %s\n",cudaGetErrorString(status));
    }
}
void cuda_copy_real_d2d(real* dist, real* ori, int start, int n){
    cudaMemcpy(&dist[start],ori,n*sizeof(real),cudaMemcpyDeviceToDevice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_real_d2d: copy): %s\n",cudaGetErrorString(status));
    }
}

void cuda_copy_weight_to_device(real *x, real *d_x, int n, int taskid){
    int y=taskid * n;
    cudaMemcpy(d_x,&(x[y]),n*sizeof(real),cudaMemcpyHostToDevice);

}

void cuda_copy_real_to_host(real *x, real *d_x, int n){
    printf("copy real to host %d ...",n);
    cudaMemcpy(x,d_x,n*sizeof(real),cudaMemcpyDeviceToHost);
    printf("done!\n");
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_real_to_host: copy): %s\n",cudaGetErrorString(status));
    }
}



void cuda_copy_int_to_device(int *x, int *d_x, int n){
    cudaMemcpy(d_x,x,n*sizeof(int),cudaMemcpyHostToDevice);
}

void cuda_copy_int_to_host(int *x, int *d_x, int n){
    cudaMemcpy(x,d_x,n*sizeof(int),cudaMemcpyDeviceToHost);
}

void cuda_allocate_scaling(real ** d_scaling, int N_images){
    cudaMalloc(d_scaling,N_images*sizeof(real));
    thrust::device_ptr<real> p(*d_scaling);
    thrust::fill(p, p+N_images, real(1));
}

void cuda_allocate_scaling_full(real **d_scaling, int N_images, int N_slices) {
    cudaMalloc(d_scaling, N_images*N_slices*sizeof(real));
    thrust::device_ptr<real> p(*d_scaling);
    thrust::fill(p, p+N_images*N_slices, real(1.));
}

__global__ void cuda_normalize_responsabilities_single_kernel(real *respons, int N_slices, int N_images) {
    __shared__ real max_cache[TNUM];
    __shared__ int index_cache[TNUM];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real this_resp;
    for (int i_slice= tid; i_slice < N_slices; i_slice += step) {
        this_resp = respons[i_slice*N_images+i_image];
        if (this_resp > max_cache[tid]) {
            max_cache[tid] = this_resp;
            index_cache[tid] = i_image;
        }
    }
    inblock_maximum_index(max_cache, index_cache);

    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        respons[i_slice*N_images+i_image] = 0.;
    }
    __syncthreads();
    if (tid == 0) {
        respons[index_cache[0]*N_images + i_image] = 1.;
    }
}

__global__ void cuda_normalize_responsabilities_uniform_kernel(real * respons, int N_slices, int N_images){
    __shared__ real cache[TNUM];
    /* enforce uniform orientations first */
    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_image = tid; i_image < N_images; i_image += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    __syncthreads();
    inblock_maximum(cache);
    real max_resp = cache[0];
    __syncthreads();
    for (int i_image = tid; i_image < N_images; i_image+= step) {
        respons[i_slice*N_images+i_image] -= max_resp;
    }

    cache[tid] = 0;
    for (int i_image = tid; i_image < N_images; i_image+=step) {
        if (respons[i_slice*N_images+i_image] > -1.0e10f) {
            respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    inblock_reduce(cache);
    real sum = cache[0];
    __syncthreads();
    for (int i_image = tid; i_image < N_images; i_image+=step) {
        respons[i_slice*N_images+i_image] /= sum;
    }

    /* nor normalize each images weight to one */
    int i_image = blockIdx.x;
    cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (respons[i_slice*N_images+i_image] > -1.0e10f) {
            //respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    __syncthreads();
    inblock_reduce(cache);
    //real sum = cache[0];
    sum = cache[0];
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        respons[i_slice*N_images+i_image] /= sum;
    }
}

__global__ void cuda_calculate_sum_vectors_kernel(real* respons, int N_images, int N_slices, real* d_sum){
    __shared__ real cache[TNUM];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (respons[i_slice*N_images+i_image] > -1.0e10f) {
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    __syncthreads();
    inblock_reduce(cache);
    real sum = cache[0];
    d_sum[i_image]=sum;
}

__global__ void cuda_calculate_max_vectors_kernel(real* respons, int N_images, int N_slices, real* d_maxr){
    __shared__ real cache[TNUM];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_slice = tid;i_slice < N_slices;i_slice += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    __syncthreads();
    inblock_maximum(cache);
    real max = cache[0];
    d_maxr[i_image]=max;
}

__global__ void cuda_normalize_responsabilities_kernel(real * respons, int N_slices, int N_images){
    __shared__ real cache[TNUM];

    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_slice = tid;i_slice < N_slices;i_slice += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    __syncthreads();
    inblock_maximum(cache);
    real max_resp = cache[0];
    __syncthreads();
    for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
        respons[i_slice*N_images+i_image] -= max_resp;
    }

    cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (respons[i_slice*N_images+i_image] > -1.0e10f) {
            respons[i_slice*N_images+i_image] =
                    expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    inblock_reduce(cache);
    real sum = cache[0];
    __syncthreads();
    //sum = cache[0];

    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        respons[i_slice*N_images+i_image] /= sum;
    }
}

void cuda_normalize_responsabilities_single(real *d_respons, int N_slices, int N_images) {
    int nblocks = N_images;
    int nthreads = TNUM;

    cuda_normalize_responsabilities_single_kernel<<<nblocks, nthreads>>>(d_respons, N_slices, N_images);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA Error (norm resp): %s\n", cudaGetErrorString(status));
    }
}

void cuda_normalize_responsabilities(real * d_respons, int N_slices, int N_images){
    int nblocks = N_images;
    int nthreads = TNUM;
    cuda_normalize_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons, N_slices, N_images);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
    }
}

/*
 *function to calculate sum vectors for normalization.
 * vector is a sum over y
 * x->N_images
 *y-> N_slices
*/
void cuda_calculate_sum_vectors(real* d_matrix, int N_images, int N_slices, real* d_sum){
    int nblocks = N_images;
    int nthreads= TNUM;
    printf("In cuda_calculate_sum_vectors  %d %d\n", N_images,N_slices);
    cuda_calculate_sum_vectors_kernel <<<nblocks, nthreads>>>(d_matrix,N_images,N_slices,d_sum);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (Res sum): %s\n",cudaGetErrorString(status));
    }
}

void cuda_max_vector(real* d_matrix, int N_images, int N_slices, real* d_maxr){
    int nblocks = N_images;
    int nthreads= TNUM;
    // printf("In cuda_max_vector  %d %d\n", N_images,N_slices);
    cuda_calculate_max_vectors_kernel <<<nblocks, nthreads>>>(d_matrix,N_images,N_slices,d_maxr);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (Res max vector): %s\n",cudaGetErrorString(status));
    }
}

__global__ void cuda_respons_max_expf_old_kernel(real* respons,real* max,int N_slices,int N_images, real* d_sum){
    __shared__ real cache[TNUM];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = 0;


    for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
        respons[i_slice*N_images+i_image] -= max[i_image];
    }
    //cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (respons[i_slice*N_images+i_image] > -1.0e10f) {
            respons[i_slice*N_images+i_image] =
                    expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    __syncthreads();
    inblock_reduce(cache);
    real sum = cache[0];
    d_sum[i_image] = sum;
    //d_sum[i_image] = log(sum);
}

__global__ void cuda_respons_max_expf_kernel(real* respons,real* d_tmp,real* max,int N_slices,int N_images, real* d_sum){
    __shared__ real cache[TNUM];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = 0;


    for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
        d_tmp[i_slice*N_images+i_image] =respons[i_slice*N_images+i_image] - max[i_image];
    }

    //cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (d_tmp[i_slice*N_images+i_image] > -1.0e10f) {
            d_tmp[i_slice*N_images+i_image] =
                    expf(d_tmp[i_slice*N_images+i_image]);
            cache[tid] += d_tmp[i_slice*N_images+i_image];
        } else {
            d_tmp[i_slice*N_images+i_image] = 0.0f;
        }
    }
    __syncthreads();
    inblock_reduce(cache);
    real sum = cache[0];
    //d_sum[i_image] = sum;
    d_sum[i_image] = sum;
}


void cuda_respons_max_expf(real* d_respons, real* max, int N_images, int allocate_slices, real* d_sum){
    int nblocks = N_images;
    int nthreads = TNUM;
    //cuda_respons_max_expf_kernel<<<nblocks,nthreads>>>(d_respons, max, allocate_slices, N_images,d_sum);
    real * d_tmp;
    cuda_allocate_real(&d_tmp,allocate_slices*N_images);
    cuda_respons_max_expf_kernel<<<nblocks,nthreads>>>(d_respons,d_tmp, max, allocate_slices, N_images,d_sum);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda max expf): %s\n",cudaGetErrorString(status));
    }
    cuda_mem_free(d_tmp);
}

void cuda_respons_max_expf_old(real* d_respons, real* max, int N_images, int allocate_slices, real* d_sum){
    int nblocks = N_images;
    int nthreads = TNUM;
    cuda_respons_max_expf_old_kernel<<<nblocks,nthreads>>>(d_respons, max, allocate_slices, N_images,d_sum);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda max expf old): %s\n",cudaGetErrorString(status));
    }

}



__global__ void cuda_normalize_responsabilities_sum_kernel(real * respons,  real* d_sum, real* max, int N_images, int allocate_slices){
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    for (int i_slice = tid; i_slice < allocate_slices; i_slice += step) {
        respons [i_slice*N_images + i_image] /= d_sum[i_image];
    }
}


void cuda_norm_respons_by_summax(real * d_respons,  real* d_sum, real* max, int N_images, int allocate_slices)
{
    int nblocks = N_images;
    int nthreads = TNUM;
    cuda_normalize_responsabilities_sum_kernel <<<nblocks,nthreads>>>(  d_respons,  d_sum, max,  N_images, allocate_slices);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (max normalize_responsabilities): %s\n",cudaGetErrorString(status));
    }
}

__global__ void cuda_norm_respons_sumexpf_kernel(real * respons,  real* d_sum, real* max, int N_images, int allocate_slices){
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    for (int i_slice = tid; i_slice < allocate_slices; i_slice += step) {
        respons [i_slice*N_images + i_image] = expf(respons[i_slice*N_images + i_image] -d_sum[i_image]);
    }
}


void cuda_norm_respons_sumexpf(real * d_respons,  real* d_sum, real* max, int N_images, int allocate_slices)
{
    int nblocks = N_images;
    int nthreads = TNUM;
    cuda_norm_respons_sumexpf_kernel <<<nblocks,nthreads>>>(  d_respons,  d_sum, max,  N_images, allocate_slices);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (max cuda_norm_respons_sumexpf): %s\n",cudaGetErrorString(status));
    }
}

void cuda_mem_free(real * d){
    cudaError_t status = cudaFree(d);
    if(status != cudaSuccess){
        printf("CUDA Error (mem free): %s\n",cudaGetErrorString(status));
    }
}

__global__ void collapse_responsabilities_kernel(real *respons, int N_slices) {
    int i_image = blockIdx.x;
    int N_images = gridDim.x;
    int step = blockDim.x;
    int tid = threadIdx.x;

    real this_resp;
    real best_resp = 0.;
    int best_resp_index = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        this_resp = respons[i_slice*N_images + i_image];
        if (this_resp > best_resp) {
            best_resp = this_resp;
            best_resp_index = i_slice;
        }
    }
    __syncthreads();
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        respons[i_slice*N_images + i_image] = 0.;
    }
    respons[best_resp_index*N_images + i_image] = 1.;
}

void cuda_collapse_responsabilities(real *d_respons, int N_slices, int N_images) {
    int nblocks = N_images;
    int nthreads = TNUM;
    collapse_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons, N_slices);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
    }
}

// x_log_x<T> computes the f(x) -> x*log(x)
template <typename T>
struct x_log_x
{
    __host__ __device__
    T operator()(const T& x) const {
        if(x > 0){
            return x * logf(x);
        }else{
            return 0;
        }
    }
};

real cuda_total_respons(real * d_respons,int n){
    thrust::device_ptr<real> p(d_respons);
    x_log_x<real> unary_op;
    thrust::plus<real> binary_op;
    real init = 0;
    // Calculates sum_0^n d_respons*log(d_respons)
    return thrust::transform_reduce(p, p+n, unary_op, init, binary_op);
}

void cuda_copy_slice_chunk_to_host(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d){

    cudaMemcpy(&slices[slice_start],d_slices,sizeof(real)*N_2d*slice_chunk,cudaMemcpyDeviceToHost);
}

void cuda_copy_slice_chunk_to_device(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d){

    cudaMemcpy(d_slices,&slices[slice_start],sizeof(real)*N_2d*slice_chunk,cudaMemcpyHostToDevice);
}

void cuda_calculate_fit(real * slices, real * d_images, int * d_mask,
                        real * d_scaling, real * d_respons, real * d_fit, real sigma,
                        int N_2d, int N_images, int slice_start, int slice_chunk){
    //call the kernel
    dim3 nblocks(N_images,slice_chunk);
    int nthreads = TNUM;
    calculate_fit_kernel<<<nblocks,nthreads>>>(slices, d_images, d_mask,
                                               d_respons, d_fit, sigma, d_scaling,
                                               N_2d, slice_start);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
    }
}

void cuda_calculate_fit_best_rot(real *slices, real * d_images, int *d_mask,
                                 real *d_scaling, int *d_best_rot, real *d_fit,
                                 int N_2d, int N_images, int slice_start, int slice_chunk) {
    dim3 nblocks(N_images, slice_chunk);
    int nthreads = TNUM;
    calculate_fit_best_rot_kernel<<<nblocks, nthreads>>>(slices, d_images, d_mask,
                                                         d_best_rot, d_fit, d_scaling,
                                                         N_2d, slice_start);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
    }
}


void cuda_calculate_radial_fit(real *slices, real *d_images, int *d_mask,
                               real *d_scaling, real *d_respons, real *d_radial_fit,
                               real *d_radial_fit_weight, real *d_radius,
                               int N_2d, int side, int N_images, int slice_start,
                               int slice_chunk){
    dim3 nblocks(N_images,slice_chunk);
    int nthreads = TNUM;
    calculate_radial_fit_kernel<<<nblocks,nthreads>>>(slices, d_images, d_mask,
                                                      d_respons, d_scaling, d_radial_fit,
                                                      d_radial_fit_weight, d_radius,
                                                      N_2d, side, slice_start);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess) {
        printf("CUDA Error (radial fit): %s\n",cudaGetErrorString(status));
    }
}

void cuda_calculate_best_rotation(real *d_respons, int *d_best_rotation, int N_images, int N_slices){
    int nblocks = N_images;
    int nthreads = TNUM;
    calculate_best_rotation_kernel<<<nblocks, nthreads>>>(d_respons, d_best_rotation, N_slices);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA Error (best rotation): %s\n", cudaGetErrorString(status));
    }
}

__global__ void multiply_by_gaussian_kernel(cufftComplex *model, const real sigma) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int y = blockIdx.x;
    const int z = blockIdx.y;
    const int model_side = gridDim.x;

    real radius2;
    real sigma2 = pow(sigma/(real)model_side, 2);
    int dx, dy, dz;
    if (model_side - y < y) {
        dy = model_side - y;
    } else {
        dy = y;
    }
    if (model_side - z < z) {
        dz = model_side - z;
    } else {
        dz = z;
    }

    for (int x = tid; x < (model_side/2+1); x += step) {
        if (model_side - x < x) {
            dx = model_side - x;
        } else {
            dx = x;
        }

        // find distance to top left
        radius2 = pow((real)dx, 2) + pow((real)dy, 2) + pow((real)dz, 2);
        // calculate gaussian kernel

        /*
    model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2/((real)model_side))/(pow((real)model_side, 3));
    model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2/((real)model_side))/(pow((real)model_side, 3));
    */

        model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
        model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));

        /*
    model[z*model_side*model_side + y*model_side + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
    model[z*model_side*model_side + y*model_side + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
    */
    }
}

__global__ void get_mask_from_model(real *model, int *mask, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        if (model[i] < 0.) {
            mask[i] = 0;
            model[i] = 0.;
        } else {
            mask[i] = 1;
        }
    }
}

__global__ void apply_mask(real *model, int *mask, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        if (mask[i] == 0) {
            model[i] = -1.;
        }
    }
}

void cuda_blur_model(real *d_model, const int model_side, const real sigma) {
    cufftComplex *ft;
    cudaMalloc((void **)&ft, model_side*model_side*(model_side/2+1)*sizeof(cufftComplex));

    int *d_mask;
    cudaMalloc(&d_mask, model_side*model_side*model_side*sizeof(int));
    get_mask_from_model<<<(pow(model_side,3)/TNUM+1), TNUM>>>(d_model, d_mask, pow(model_side, 3));

    cufftHandle plan;
    cufftPlan3d(&plan, model_side, model_side, model_side, CUFFT_R2C);
    //cufftExecR2C(plan, d_model, ft);//, CUFFT_FORWARD);
    //multiply by gaussian kernel

    int nthreads = TNUM;
    dim3 nblocks(model_side, model_side);
    multiply_by_gaussian_kernel<<<nblocks,nthreads>>>(ft, sigma);
    cufftPlan3d(&plan, model_side, model_side, model_side, CUFFT_C2R);
    //cufftExecC2R(plan, ft, d_model);//, CUFFT_INVERSE);
    apply_mask<<<(pow(model_side,3)/TNUM+1),TNUM>>>(d_model, d_mask, pow(model_side,3));

}


void cuda_reset_real(real *d_real, int len){
    cudaMemset(d_real,0,sizeof(real)*len);
}

__global__ void cuda_normalize_responsabilities_com_kernel(real * respons,
                                                           int N_slices, int N_images,real* d_sum,
                                                           real* d_max, real* respons1,
                                                           real* respons2){
    __shared__ real cache[TNUM];

    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_slice = tid;i_slice < N_slices;i_slice += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    inblock_maximum(cache);
    real max_resp = cache[0];
    __syncthreads();
    d_max[i_image] = max_resp;
    for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
        respons1[i_slice*N_images+i_image] = respons[i_slice*N_images+i_image] -max_resp;
    }

    cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (respons1[i_slice*N_images+i_image] > -1.0e10f) {
            respons2[i_slice*N_images+i_image] =
                    expf(respons1[i_slice*N_images+i_image]);
            cache[tid] += respons2[i_slice*N_images+i_image];
        } else {
            respons2[i_slice*N_images+i_image] = 0.0f;
        }
    }
    __syncthreads();


    inblock_reduce(cache);
    real sum = cache[0];
    d_sum[i_image] = sum;
    //sum = cache[0];
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        respons[i_slice*N_images+i_image] = respons2[i_slice*N_images+i_image]/sum;
    }
}



void cuda_normalize_responsabilities_com(real * d_respons, int N_slices,
                                         int N_images,real* d_sum, real* d_max,
                                         real* d_respons1, real* d_respons2){
    int nblocks = N_images;
    int nthreads = TNUM;
    cuda_normalize_responsabilities_com_kernel<<<nblocks,nthreads>>>(d_respons,
                                                                     N_slices, N_images,d_sum, d_max,
                                                                     d_respons1, d_respons2);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
    }
}
