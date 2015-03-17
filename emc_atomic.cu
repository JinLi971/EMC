#include "emc_cuda_help.h"
#include <spimage.h>
#include "emc_helper.h"
#include <stdio.h>


__device__ inline void atomicFloatAdd(float *address, float val)
{
    int i_val = __float_as_int(val);
    int tmp0 = 0;
    int tmp1;

    while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
    {
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}

#ifdef _SP_DOUBLE_PRECISION
__device__ inline void atomicFloatAdd(real *address, float val)
{
    atomicFloatAdd((float *)(address), val);
}


__device__ inline void atomicAdd(real *address, double val)
{
    atomicAdd(static_cast<double *>(address), val);
}
#endif

__device__ void cuda_insert_slice(real *model, real *weight, real *slice,
                                  int * mask, real w, real *rot, real *x_coordinates,
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
            //if (mask[y*x_max + x] == 1) {
            if (mask[y*x_max + x] == 1 && slice[y*x_max + x] >= 0.0) {
                /* This is just a matrix multiplication with rot */
                new_x = m00*x_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*z_coordinates[y*x_max+x];
                new_y =	m10*x_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*z_coordinates[y*x_max+x];
                new_z = m20*x_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*z_coordinates[y*x_max+x];
                /* changed the next lines +0.5 -> -0.5 (11 dec 2012)*/
                round_x = lroundf(model_x/2.0f - 0.5f + new_x);
                round_y = lroundf(model_y/2.0f - 0.5f + new_y);
                round_z = lroundf(model_z/2.0f - 0.5f + new_z);

                //printf(">>>>>round_x = %f", round_x);
                if (round_x >= 0 && round_x < model_x &&
                        round_y >= 0 && round_y < model_y &&
                        round_z >= 0 && round_z < model_z) {
                    /* this is a simple compile time check that can go bad at runtime, but such is life */
#if __CUDA_ARCH__ >= 200
                    atomicAdd(&model[round_z*model_x*model_y + round_y*model_x + round_x], w * slice[y*x_max + x]);
                    atomicAdd(&weight[round_z*model_x*model_y + round_y*model_x + round_x], w);
#else
                    atomicFloatAdd(&model[round_z*model_x*model_y + round_y*model_x + round_x], w * slice[y*x_max + x]);
                    atomicFloatAdd(&weight[round_z*model_x*model_y + round_y*model_x + round_x], w);
#endif
                    //	  model[(round_z*model_x*model_y + round_y*model_x + round_x)] += w * slice[y*x_max + x];
                    //	  weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;

                }
            }
        }
    }
}


__global__ void update_slices_kernel(real * images, real * slices, int * mask, real * respons,
                                     real * scaling, int * active_images, int N_images, int slice_start, int N_2d,
                                     real * slices_total_respons, real * rot,
                                     real * x_coord, real * y_coord, real * z_coord,
                                     real * model,
                                     int slice_rows, int slice_cols,
                                     int model_x, int model_y, int model_z){
    /* each block takes care of 1 slice */
    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = 0.0f;


    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] != 0) {
            real sum = 0;
            for (int i_image = 0; i_image < N_images; i_image++) {
                if (active_images[i_image] > 0) {
                    sum += images[i_image*N_2d+i]*
                            respons[(slice_start+i_slice)*N_images+i_image]/scaling[(slice_start+i_slice)*N_images+i_image];
                }
            }
            slices[i_slice*N_2d+i] = sum;
        } else {
            slices[i_slice*N_2d+i] = -1.0;
        }
    }
    for (int i_image = 0; i_image < N_images; i_image++) {
        if (active_images[i_image] > 0) {
            total_respons += respons[(slice_start+i_slice)*N_images+i_image];
        }
    }
    if(tid == 0){
        slices_total_respons[i_slice] =  total_respons;
    }
    if(total_respons > 1e-10f){
        // printf("any updated slices? %f \n",total_respons);
        for (int i = tid; i < N_2d; i+=step) {
            if (mask[i] != 0) {
                slices[i_slice*N_2d+i] /= total_respons;
            }
        }
    }
}

__global__ void update_slices_final_kernel(real * images, real * slices, int * mask, real * respons,
                                           real * scaling, int * active_images, int N_images,
                                           int slice_start, int N_2d,
                                           real * slices_total_respons, real * rot,
                                           real * x_coord, real * y_coord, real * z_coord,
                                           real * model, real * weight,
                                           int slice_rows, int slice_cols,
                                           int model_x, int model_y, int model_z){
    /* each block takes care of 1 slice */
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = 0.0f;
    int i_slice = bid;
    for (int i = tid; i < N_2d; i+=step) {
        real sum = 0.0;
        real count = 0.0;
        for (int i_image = 0; i_image < N_images; i_image++) {
            if (active_images[i_image] && images[i_image*N_2d+i] >= 0.0) {
                sum += images[i_image*N_2d+i]*
                        respons[(slice_start+i_slice)*N_images+i_image]/scaling[(slice_start+i_slice)*N_images+i_image];
                count += respons[(slice_start+i_slice)*N_images+i_image];
            }
        }
        if (count > 1.0e-10) {
            slices[i_slice*N_2d+i] = sum/count;
        } else {
            slices[i_slice*N_2d+i] = -1.0;
        }
    }
    for (int i_image = 0; i_image < N_images; i_image++) {
        if (active_images[i_image]) {
            total_respons += respons[(slice_start+i_slice)*N_images+i_image];
        }
    }
    if(tid == 0){
        slices_total_respons[bid] =  total_respons;
    }
    /*
  if(total_respons > 1e-10f){
    for (int i = tid; i < N_2d; i+=step) {
      slices[i_slice*N_2d+i] /= total_respons;
    }
  } else {
    slices[i_slice*N_2d+i] = -1.0;
  }
  */
}

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
                                     real * scaling, int N_images, int N_2d,
                                     real * slices_total_respons, real * rot,
                                     real * x_coord, real * y_coord, real * z_coord,
                                     real * model, real * weight,
                                     int slice_rows, int slice_cols,
                                     int model_x, int model_y, int model_z){
    /* each block takes care of 1 slice */
    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = slices_total_respons[i_slice];
    if(total_respons > 1e-10f){
        cuda_insert_slice(model, weight, &slices[i_slice*N_2d], mask, total_respons,
                          &rot[4*i_slice], x_coord, y_coord, z_coord,
                          slice_rows, slice_cols, model_x, model_y, model_z, tid, step);
    }
}

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

__device__ void cuda_calculate_responsability_absolute_atomic(float *slice, float *image, int *mask, real sigma, real scaling, int N_2d, int tid, int step, real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] >= 0.0f) {
            sum += pow(slice[i] - image[i]/scaling,2);
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_responsability_poisson_atomic(float *slice, float *image,
                                                             int *mask, real sigma, real scaling,
                                                             int N_2d, int tid, int step,
                                                             real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] >= 0.0f) {
            //sum += pow((slice[i] - image[i]/scaling) / (sqrt(image[i])+0.4), 2);
            sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]+0.02), 2); // 0.2 worked
            //sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]/0.5+10.0), 2); // 0.2 worked
            //sum += pow((slice[i]*scaling - image[i])/8.0/ (sqrt(image[i]/8.0 + 1.0)), 2); // 0.2 worked
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_responsability_true_poisson_atomic(float *slice, float *image,
                                                                  int *mask, real sigma, real scaling,
                                                                  int N_2d, int tid, int step,
                                                                  real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] >= 0.0f) {
            sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+0.2), 2); // 0.2 worked
            //sum += pow(slice[i]*scaling - image[i], 2) / ((image[i]/8.0 + 1.0)*pow(8.0, 2));
            //sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+1.0), 2);
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_fit_error(float* slice, float *image,
                                         int *mask, real scaling,
                                         int N_2d, int tid, int step,
                                         real *sum_cache, int *count_cache) {
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        //if (mask[i] != 0 && slice[i] >= 0.0f) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            sum += abs((slice[i] - image[i]/scaling) / (slice[i] + image[i]/scaling));
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}

__device__ void cuda_calculate_fit2_error(real* slice, real *image,
                                          int *mask, real scaling,
                                          int N_2d, int tid, int step,
                                          real *nom_cache, real *den_cache) {
    real nom = 0.0;
    real den = 0.0;
    const int i_max = N_2d;
    for (int i = tid; i < i_max; i+=step) {
        //if (mask[i] != 0 && slice[i] >= 0.0f) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            nom += abs(slice[i] - image[i]/scaling);
            den += abs(slice[i] + image[i]/scaling);
        }
    }
    nom_cache[tid] = nom;
    den_cache[tid] = den;
}


__device__ real cuda_calculate_single_fit_error(float* slice, float *image,
                                                int *mask, real scaling,
                                                int N_2d, int tid, int step) {
    __shared__ real sum_cache[TNUM];
    __shared__ int count_cache[TNUM];
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        //if (mask[i] != 0 && slice[i] >= 0.0f) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            sum += abs((slice[i] - image[i]/scaling) / (slice[i] + image[i]/scaling));
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    inblock_reduce(sum_cache);
    inblock_reduce(count_cache);
    return sum_cache[0]/count_cache[0];
}

__device__ real cuda_calculate_single_fit2_error(real* slice, real *image,
                                                 int *mask, real scaling,
                                                 int N_2d, int tid, int step) {
    __shared__ real nom_cache[TNUM];
    __shared__ real den_cache[TNUM];
    real nom = 0.0;
    real den = 0.0;
    const int i_max = N_2d;
    for (int i = tid; i < i_max; i+=step) {
        //if (mask[i] != 0 && slice[i] >= 0.0f) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            nom += abs(slice[i] - image[i]/scaling);
            den += abs(slice[i] + image[i]/scaling);
        }
    }
    nom_cache[tid] = nom;
    den_cache[tid] = den;
    inblock_reduce(nom_cache);
    inblock_reduce(den_cache);

    return nom_cache[0]/(den_cache[0]);
}


__global__ void calculate_fit_kernel(real *slices, real *images, int *mask,
                                     real *respons, real *fit, real sigma,
                                     real *scaling, int N_2d, int slice_start){
    //__shared__ real sum_cache[TNUM];
    __shared__ real nom_cache[TNUM];
    __shared__ real den_cache[TNUM];
    //__shared__ int count_cache[TNUM];
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    /*
  cuda_calculate_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
               scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step, sum_cache, count_cache);
  */

    cuda_calculate_fit2_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
                              scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step, nom_cache, den_cache);

    /*
  cuda_calculate_responsability_poisson_atomic(&slices[i_slice*N_2d],
                           &images[i_image*N_2d],mask,
                           sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid,step,
                           sum_cache,count_cache);
  */
    /*
  cuda_calculate_responsability_absolute_atomic(&slices[i_slice*N_2d],
                        &images[i_image*N_2d],mask,
                        sigma,scaling[(slisce_start+i_slice)*N_images+i_image], N_2d, tid,step,
                        sum_cache,count_cache);
  */
    //inblock_reduce(sum_cache);
    inblock_reduce(nom_cache);
    inblock_reduce(den_cache);
    //inblock_reduce(count_cache);
    __syncthreads();
    if (tid == 0) {
        //atomicAdd(&fit[i_image], sum_cache[0]/count_cache[0]*respons[(slice_start+i_slice)*N_images+i_image]);
        atomicAdd(&fit[i_image], nom_cache[0]/(den_cache[0])*respons[(slice_start+i_slice)*N_images+i_image]);
        //atomicFloatAdd(&fit[i_image], respons[(slice_start+i_slice)*N_images+i_image]);
        //atomicFloatAdd(&fit[i_image], 1.);
    }
    /*
  if(tid == 0){
    atomicFloatAdd(&fit[i_image], expf(-sum_cache[0]/2.0/(real)count_cache[0]/pow(sigma,2)) *
           respons[(slice_start+i_slice)*N_images+i_image]);
  }
  */
}

__global__ void calculate_fit_best_rot_kernel(real *slices, real *images, int *mask,
                                              int *best_rot, real *fit,
                                              real *scaling, int N_2d, int slice_start){
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    real this_fit;
    if (best_rot[i_image] == (slice_start+i_slice)) {
        /*
    this_fit = cuda_calculate_single_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
                           scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);
    */
        this_fit = cuda_calculate_single_fit2_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
                                                    scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);

        if (tid == 0) {
            fit[i_image] = this_fit;
        }
    }
}

/* calcualte the fit as a function of radius */
__global__ void calculate_radial_fit_kernel(real * slices , real * images, int * mask,
                                            real * respons, real * scaling, real * radial_fit,
                                            real * radial_fit_weight, real * radius,
                                            int N_2d,  int side,  int slice_start){
    __shared__ real sum_cache[TNUM]; //TNUM
    __shared__ real weight_cache[TNUM];
    const int max_radius = side/2;
    //printf("before alloc\n");
    //printf("after alloc\n");
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;
    real error;
    int rad;

    if (tid < max_radius) {
        sum_cache[tid] = 0.0;
        weight_cache[tid] = 0.0;
    }

    __syncthreads();
    real this_resp = respons[(slice_start+i_slice)*N_images+i_image];
    //printf("%g\n", this_resp);
    //if (this_resp > 1.0e-10) {
    for (int i = tid; i < N_2d; i += step) {
        if (mask[i] != 0 && slices[i_slice*N_2d+i] > 0.0f) {
            error = fabs((slices[i_slice*N_2d+i] - images[i_image*N_2d+i]/scaling[(slice_start+i_slice)*N_images+i_image]) /
                         (slices[i_slice*N_2d+i] + images[i_image*N_2d+i]/scaling[(slice_start+i_slice)*N_images+i_image]));
            /*
      if (i_image == 0 && i_slice == 0) {
    printf("error[i_slice=%d, i_image=%d, i=%d] = %g, (%g ~ %g) (resp = %g)\n", i_slice, i_image, i, error, slices[i_slice*N_2d+i], images[i_image*N_2d+i]/scaling[i_image], this_resp);
      }
      */
            rad = (int)radius[i];

            if (rad < max_radius) {
                atomicAdd(&sum_cache[rad],error*this_resp);
                atomicAdd(&weight_cache[rad],this_resp);
            }
        }
    }
    //}
    __syncthreads();
    if (tid < max_radius) {
        atomicAdd(&radial_fit[tid],sum_cache[tid]);
        atomicAdd(&radial_fit_weight[tid],weight_cache[tid]);
    }
}

/* 
__global__ void calulate_standard_deviation(real *slices, real *images, int *mask, real *respons, real *scaling,
                        real *radial_std, real *radius, int N_2d, int side, int slice_start) {
  __shared__ real nom[TNUM];
  __shared__ real den[TNUM];
  nom[tid] = 0.;
  den[tid] = 0.;

  for (int i = tid; i < N_2d; i += step) {
    nom[tid] +=
  }
}
*/
