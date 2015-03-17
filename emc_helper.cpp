#include "emc_helper.h"
#include <signal.h>
#include <sys/stat.h>
#include <gsl/gsl_rng.h>
#include <memory>
#include <hdf5.h>
emc_helper::emc_helper()
{
}
double emc_helper::real_real_validate(real* a, real*b, int len){
    double sum;
    for (int i =0; i<len; i++)
        sum += abs(a[i]-b[i]);
    return sum;
}

void emc_helper::reset_real(real* h_real, int len){
    for(int i =0; i<len; i++)
        h_real[i] =0;
}

void emc_helper::nice_exit(int sig){
    if (sig== 0) {
        sig = 1;
    } else {
        exit(1);
    }
}
void emc_helper::max_max_vector(real*a,real*b,int len){
    for(int i=0;i<len;i++)
        a[i]= a[i]>b[i]?a[i]:b[i];
}

void emc_helper :: move_real(real* source, real* dest, int start, int len){
    for(int i = 0; i<len; i++)
        dest[i+start]= source[i];
}

int emc_helper::get_allocate_len(int ntasks, int N_slices, int taskid){
    int slice_start = taskid* N_slices/ntasks;
    int slice_end =  (taskid+1)* N_slices/ntasks;
    if (taskid == ntasks -1)
        slice_end = N_slices;
    return slice_end - slice_start;
}

int emc_helper::compare_real(const void *pa, const void *pb) {
    real a = *(const real*)pa;
    real b = *(const real*)pb;
    if (a < b) {return -1;}
    else if (a > b) {return 1;}
    else {return 0;}
}

void emc_helper::calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength,
                                       sp_matrix *x_coordinates,
                                       sp_matrix *y_coordinates, sp_matrix *z_coordinates) {
    const int x_max = side;
    const int y_max = side;
    real pixel_r, real_r, fourier_r, angle_r, fourier_z;
    real pixel_x, pixel_y, pixel_z;
    //tabulate angle later
    for (int x = 0; x < x_max; x++) {
        for (int y = 0; y < y_max; y++) {
            pixel_r = sqrt(pow((real)(x-x_max/2)+0.5,2) + pow((real)(y-y_max/2)+0.5,2));
            real_r = pixel_r*pixel_size;
            angle_r = atan2(real_r,detector_distance);
            fourier_r = sin(angle_r)/wavelength;
            fourier_z = (1. - cos(angle_r))/wavelength;

            pixel_x = (real)(x-x_max/2)+0.5;
            pixel_y = (real)(y-y_max/2)+0.5;
            pixel_z = fourier_z/fourier_r*pixel_r;
            sp_matrix_set(x_coordinates,x,y,pixel_x);
            sp_matrix_set(y_coordinates,x,y,pixel_y);
            sp_matrix_set(z_coordinates,x,y,pixel_z);
        }
    }
}

void emc_helper::insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
                              sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
                              sp_matrix *y_coordinates, sp_matrix *z_coordinates)
{
    const int x_max = sp_matrix_rows(slice);
    const int y_max = sp_matrix_cols(slice);
    //tabulate angle later
    real new_x, new_y, new_z;
    int round_x, round_y, round_z;
    for (int x = 0; x < x_max; x++) {
        for (int y = 0; y < y_max; y++) {
            if (sp_imatrix_get(mask,x,y) == 1) {
                /* This is just a matrix multiplication with rot */
                new_x =
                        (rot->q[0]*rot->q[0] + rot->q[1]*rot->q[1] -
                        rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
                        (2.0*rot->q[1]*rot->q[2] -
                        2.0*rot->q[0]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
                        (2.0*rot->q[1]*rot->q[3] +
                        2.0*rot->q[0]*rot->q[2])*sp_matrix_get(z_coordinates,x,y);
                new_y =
                        (2.0*rot->q[1]*rot->q[2] +
                        2.0*rot->q[0]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
                        (rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] +
                        rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
                        (2.0*rot->q[2]*rot->q[3] -
                        2.0*rot->q[0]*rot->q[1])*sp_matrix_get(z_coordinates,x,y);
                new_z =
                        (2.0*rot->q[1]*rot->q[3] -
                        2.0*rot->q[0]*rot->q[2])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
                        (2.0*rot->q[2]*rot->q[3] +
                        2.0*rot->q[0]*rot->q[1])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
                        (rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] -
                        rot->q[2]*rot->q[2] + rot->q[3]*rot->q[3])*sp_matrix_get(z_coordinates,x,y);
                round_x = round((real)sp_3matrix_x(model)/2.0 + 0.5 + new_x);
                round_y = round((real)sp_3matrix_y(model)/2.0 + 0.5 + new_y);
                round_z = round((real)sp_3matrix_z(model)/2.0 + 0.5 + new_z);
                if (round_x >= 0 && round_x < sp_3matrix_x(model) &&
                        round_y >= 0 && round_y < sp_3matrix_y(model) &&
                        round_z >= 0 && round_z < sp_3matrix_z(model)) {
                    sp_3matrix_set(model,round_x,round_y,round_z,
                                   sp_3matrix_get(model,round_x,round_y,round_z)+w*sp_matrix_get(slice,x,y));
                    sp_3matrix_set(weight,round_x,round_y,round_z,sp_3matrix_get(weight,round_x,round_y,round_z)+w);
                }
            }//endif
        }
    }
}

/*void emc_helper::test_blur() {
    int i_device = cuda_get_device();
    printf("device id = %d\n", i_device);
    
    const int image_side = 10;
    const int N_3d = pow(image_side, 3);

    real *image = malloc(N_3d*sizeof(real));
    for (int i = 0; i < N_3d; i++) {
        image[i] = 0.;
    }
    //image[image_side*image_side*image_side/2 + image_side*image_side/2 + image_side/2] = 1.;
    image[image_side*image_side*2 + image_side*3 + 3] = 1.;
    image[image_side*image_side*7 + image_side*8 + 5] = 1.;
    image[image_side*image_side*5 + image_side*5 + 5] = -1.;
    FILE *blur_out = fopen("debug/blur_before.data", "wp");
    for (int i = 0; i < N_3d; i++) {
        fprintf(blur_out, "%g\n", image[i]);
    }
    fclose(blur_out);
    real *d_image;
    cuda_allocate_real(&d_image, N_3d);
    cuda_copy_real_to_device(image, d_image, N_3d);


    int *mask = malloc(N_3d*sizeof(int));
    for (int i = 0; i < N_3d; i++) {
        mask[i] = 1;
    }
    int *d_mask;
    cuda_allocate_int(&d_mask, N_3d);
    cuda_copy_int_to_device(mask, d_mask, N_3d);

    cuda_blur_model(d_image, image_side, 1.);

    cuda_copy_real_to_host(image, d_image, N_3d);
    blur_out = fopen("debug/blur_after.data", "wp");
    for (int i = 0; i < N_3d; i++) {
        fprintf(blur_out, "%g\n", image[i]);
    }
    fclose(blur_out);
    exit(0);
    
}*/

void emc_helper::average_Models(sp_3matrix ** models, int partition, int len,sp_3matrix *newMatrix){
    // show_models(models,partition,len);
    newMatrix->x = models[0]->x;
    newMatrix->y = models[0]->y;
    newMatrix->z = models[0]->z;
    int allLen =  len*len*len;
    for (int j = 0; j<allLen; j++){
        double tmp = 0;
        for(int i=0; i<partition; i++){
            tmp += models[i]->data[j];
        }
        newMatrix->data[j] = tmp/partition;
        //newMatrix->data[j] = tmp;
        tmp = 0;
    }
    // printf("\nmodels data after  average models %d %d %d %f %f \n", newMatrix->x, newMatrix->y, newMatrix->z, newMatrix->data[1568], newMatrix->data[1573]);
}

void emc_helper::show_models(sp_3matrix ** models, int partition, int len){
    for(int i = 0;i <partition; i++)
        printf("models data in average models %d %d %d %f %f \n", models[i]->x, models[i]->y, models[i]->z, models[i]->data[1568], models[i]->data[1573]);
}

sp_matrix ** emc_helper::read_images(Configuration conf, sp_imatrix **masks)
{
    sp_matrix **images =(sp_matrix **) malloc(conf.N_images*sizeof(sp_matrix *));
    //masks = malloc(conf.N_images*sizeof(sp_imatrix *));
    Image *img;
    real *intensities = (real*)malloc(conf.N_images*sizeof(real));
    char buffer[1000];

    for (int i = 0; i < conf.N_images; i++) {
        intensities[i] = 1.0;
    }

    for (int i = 0; i < conf.N_images; i++) {
        sprintf(buffer,"%s%.4d.h5", conf.image_prefix, i);
        img = sp_image_read(buffer,0);

        /* blur image if enabled */
        if (conf.blur_image == 1) {
            Image *tmp = sp_gaussian_blur(img,conf.blur_sigma);
            sp_image_free(img);
            img = tmp;
        }

        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);

        real pixel_sum;
        int mask_sum;
        for (int x = 0; x < conf.model_side; x++) {
            for (int y = 0; y < conf.model_side; y++) {
                pixel_sum = 0.0;
                mask_sum = 0;
                for (int xb = 0; xb < conf.read_stride; xb++) {
                    for (int yb = 0; yb < conf.read_stride; yb++) {
                        pixel_sum += sp_cabs(sp_image_get(img,(int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,(int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0));
                        mask_sum += sp_image_mask_get(img,(int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,(int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0);
                    }
                }
                sp_matrix_set(images[i],x,y,pixel_sum/(real)mask_sum);
                if (mask_sum > 1) {
                    sp_imatrix_set(masks[i],x,y,1);
                } else {
                    sp_imatrix_set(masks[i],x,y,0);
                }
            }
        }

        sp_image_free(img);
    }
    return images;
}

void emc_helper::read_images_from_models(Configuration conf, sp_imatrix **masks, sp_matrix ** images){
    //sp_matrix **images =(sp_matrix **) malloc(conf.N_images*sizeof(sp_matrix *));
    //masks = malloc(conf.N_images*sizeof(sp_imatrix *));
    Image *img;
    char buffer[1000];
    for (int i = 0; i < conf.N_images; i++) {
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);
        sprintf(buffer,"%s%.4d.h5", conf.image_prefix, i);
        img = sp_image_read(buffer,0);
        for (int x = 0; x < conf.model_side; x++) {
            for (int y = 0; y < conf.model_side; y++) {
                sp_matrix_set(images[i],x,y,sp_real(sp_image_get(img,x,y,0)));
                sp_imatrix_set(masks[i],x,y,sp_image_mask_get(img,x,y,0));
            }
        }
    }

}

sp_matrix ** emc_helper::read_images_from_models(Configuration conf, sp_imatrix **masks)
{
    sp_matrix **images =(sp_matrix **) malloc(conf.N_images*sizeof(sp_matrix *));
    //masks = malloc(conf.N_images*sizeof(sp_imatrix *));
    char buffer[1000];
    real img [conf.model_side*conf.model_side];
    int mask [conf.model_side*conf.model_side];
    //sp_matrix * img = (sp_matrix*) sp_matrix_alloc(conf.model_side ,conf.model_side);
    hid_t file_id,dataset_id_img, dataset_id_mask, mem_type_id;
    herr_t status;
    if(sizeof(real) ==sizeof(float))
        mem_type_id =H5T_NATIVE_FLOAT;
    else
        mem_type_id=H5T_NATIVE_DOUBLE;
    for (int i = 0; i < conf.N_images; i++) {
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);
        sprintf(buffer,"%s%.4d.h5", conf.image_prefix, i);
        //img = sp_image_read(buffer,0);
        file_id = H5Fopen(buffer,H5F_ACC_RDWR,H5P_DEFAULT);
        if(file_id <0)
            printf("cannot open file %s \n",buffer);
     //   printf("1...");
        dataset_id_img = H5Dopen(file_id,"/real",H5P_DEFAULT);
     //   printf("2...");
        if(dataset_id_img <0)
            printf("cannot open dataset /real \n");
        status = H5Dread(dataset_id_img,mem_type_id,H5S_ALL,
                         H5S_ALL,H5P_DEFAULT,img);
   //     printf("%f", images[i]->data[0]);
//        printf("3...");
        if (status <0)
            printf("read data fail : %s \n", status);
        //memcpy(images[i]->data,img,sizeof(real)*conf.model_side*conf.model_side);
        for(int x = 0; x<conf.model_side; x++){
            for(int y =0; y<conf.model_side; y++){
                sp_matrix_set(images[i],x,y,img[x*conf.model_side+y]);
            }
        }
        dataset_id_mask = H5Dopen(file_id,"/mask",H5P_DEFAULT);
        if(dataset_id_mask <0)
            printf("cannot open dataset /mask \n");
        status = H5Dread(dataset_id_mask,H5T_NATIVE_FLOAT,H5S_ALL,
                         H5S_ALL,H5P_DEFAULT,mask);
        if(status <0)
            printf("read mask fail: %s \n", status);
        for(int x = 0; x<conf.model_side; x++){
            for(int y =0; y<conf.model_side; y++){
                sp_imatrix_set(masks[i],x,y,mask[x*conf.model_side+y]);
            }
        }
        status = H5Dclose(dataset_id_img);
        status = H5Dclose(dataset_id_mask);
        status = H5Fclose(file_id);
        //printf("close all %s \n", status);
    }
 //   printf("%f",images[0]->data[0]);
    for(int x =0;x<64;x++){
        for(int y = 0; y<64; y++){
            printf("%f ", images[0]->data[x*64+y]);
        }
        printf("\n");
    }
    return images;
}

/* init mask */
sp_imatrix *emc_helper::read_mask(Configuration conf)
{
    sp_imatrix *mask = sp_imatrix_alloc(conf.model_side,conf.model_side);;
    Image *mask_in = sp_image_read(conf.mask_file,0);
    /* read and rescale mask */
    for (int x = 0; x < conf.model_side; x++) {
        for (int y = 0; y < conf.model_side; y++) {
            if (sp_cabs(sp_image_get(mask_in,
                                     (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+
                                           sp_image_x(mask_in)/2-0.5),
                                     (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+
                                           sp_image_y(mask_in)/2-0.5),0)) == 0.0) {
                sp_imatrix_set(mask,x,y,0);
            } else {
                sp_imatrix_set(mask,x,y,1);
            }
        }
    }
    sp_image_free(mask_in);

    /* mask out everything outside the central sphere */
    for (int x = 0; x < conf.model_side; x++) {
        for (int y = 0; y < conf.model_side; y++) {
            if (sqrt(pow((real)x - (real)conf.model_side/2.0+0.5,2) +
                     pow((real)y - (real)conf.model_side/2.0+0.5,2)) >
                    conf.model_side/2.0) {
                sp_imatrix_set(mask,x,y,0);
            }
        }
    }
    return mask;
}

/* normalize images so average pixel value is 1.0 */
void emc_helper::normalize_images(sp_matrix **images, sp_imatrix *mask, Configuration conf)
{
    real sum;
    int N_2d = conf.model_side*conf.model_side;
    for (int i_image = 0; i_image < conf.N_images; i_image++) {
        sum = 0.;
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i] == 1) {
                sum += images[i_image]->data[i];
            }
        }
        sum = (real)N_2d / sum;
        for (int i = 0; i < N_2d; i++) {
            images[i_image]->data[i] *= sum;
        }
    }
}

void emc_helper::normalize_images_central_part(sp_matrix **images, sp_imatrix *mask, real radius, Configuration conf) {
    const int x_max = conf.model_side;
    const int y_max = conf.model_side;
    sp_imatrix * central_mask = sp_imatrix_alloc(x_max, y_max);
    real r;
    for (int x = 0; x < x_max; x++) {
        for (int y = 0; y < y_max; y++) {
            r = pow(x-x_max/2+0.5, 2) + pow(y-y_max/2+0.5, 2);
            if (r < pow(radius,2)) {
                sp_imatrix_set(central_mask, x, y, 1);
            } else {
                sp_imatrix_set(central_mask, x, y, 1);
            }
        }
    }
    real sum;
    int N_2d = conf.model_side*conf.model_side;
    for (int i_image = 0; i_image < conf.N_images; i_image++) {
        sum = 0.;
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i] == 1 && central_mask->data[i] == 1) {
                sum += images[i_image]->data[i];
            }
        }
        sum = (real) N_2d / sum;
        for (int i = 0; i < N_2d; i++) {
            images[i_image]->data[i] *= sum;
        }
    }
}

/* normalize images so average pixel value is 1.0 */
void emc_helper::normalize_images_individual_mask(sp_matrix **images, sp_imatrix **masks,
                                                  Configuration conf)
{
    real sum;
    int N_2d = conf.model_side*conf.model_side;
    for (int i_image = 0; i_image < conf.N_images; i_image++) {
        sum = 0.;
        for (int i = 0; i < N_2d; i++) {
            if (masks[i_image]->data[i] == 1) {
                sum += images[i_image]->data[i];
            }
        }
        sum = (real)N_2d / sum;
        for (int i = 0; i < N_2d; i++) {
            images[i_image]->data[i] *= sum;
        }
    }
}

void emc_helper::copy_real(int len, real* source , real* dst){
    for(int i = 0; i< len; i++)
        dst[i] = source[i];
}

void  emc_helper::model_init(Configuration conf, sp_3matrix * model,
                             sp_3matrix * weight, sp_matrix ** images, sp_imatrix * mask,
                             sp_matrix *x_coordinates, sp_matrix *y_coordinates, sp_matrix *z_coordinates){
    const int N_model = conf.model_side*conf.model_side*conf.model_side;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rng,0);
    if (conf.model_input == 0) {
        printf("uniform density model\n");
        for (int i = 0; i < N_model; i++) {
            //model->data[i] = 1.0;
            model->data[i] = gsl_rng_uniform(rng);
        }
        printf("model uniform end!\n");
    } else if (conf.model_input == 3) {
        printf("radial average model\n");
        real *radavg = (real*) malloc(conf.model_side/2*sizeof(real));
        int *radavg_count = (int*) malloc(conf.model_side/2*sizeof(int));
        int r;
        for (int i = 0; i < conf.model_side/2; i++) {
            radavg[i] = 0.0;
            radavg_count[i] = 0;
        }
        for (int i_image = 0; i_image < conf.N_images; i_image++) {
            for (int x = 0; x < conf.model_side; x++) {
                for (int y = 0; y < conf.model_side; y++) {
                    r = (int)sqrt(pow((real)x - conf.model_side/2.0 + 0.5,2) +
                                  pow((real)y - conf.model_side/2.0 + 0.5,2));
                    if (r < conf.model_side/2.0) {
                        radavg[r] += sp_matrix_get(images[i_image],x,y);
                        radavg_count[r] += 1;
                    }
                }
            }
        }
        for (int i = 0; i < conf.model_side/2; i++) {
            if (radavg_count[i] > 0) {
                radavg[i] /= (real) radavg_count[i];
            } else {
                radavg[i] = 0.0;
            }
        }
        real rad;
        for (int x = 0; x < conf.model_side; x++) {
            for (int y = 0; y < conf.model_side; y++) {
                for (int z = 0; z < conf.model_side; z++) {
                    rad = sqrt(pow((real)x - conf.model_side/2.0 + 0.5,2) +
                               pow((real)y - conf.model_side/2.0 + 0.5,2) +
                               pow((real)z - conf.model_side/2.0 + 0.5,2));
                    r = (int)rad;
                    if (r < conf.model_side/2.0) {
                        sp_3matrix_set(model,x,y,z,(radavg[r]*(1.0 - (rad - (real)r)) +
                                                    radavg[r+1]*(rad - (real)r)));
                    } else {
                        sp_3matrix_set(model,x,y,z,-1.0);
                    }
                }
            }
        }
    } else if (conf.model_input == 1) {
        printf("random orientations model\n");
        Quaternion *random_rot;
        for (int i = 0; i < conf.N_images; i++) {
            random_rot = quaternion_random(rng);
            insert_slice(model, weight, images[i], mask, 1.0, random_rot,
                         x_coordinates, y_coordinates, z_coordinates);
            free(random_rot);
        }
        for (int i = 0; i < N_model; i++) {
            if (weight->data[i] > 0.0) {
                model->data[i] /= (weight->data[i]);
            } else {
                model->data[i] = 0.0;
            }
        }
    } else if (conf.model_input == 2) {
        printf("model from file %s\n",conf.model_file);
        Image *model_in = sp_image_read(conf.model_file,0);
        if (conf.model_side != sp_image_x(model_in) ||
                conf.model_side != sp_image_y(model_in) ||
                conf.model_side != sp_image_z(model_in)) {
            printf("Input model is of wrong size.\n");
            exit(1);
        }
        for (int i = 0; i < N_model; i++) {
            model->data[i] = sp_cabs(model_in->image->data[i]);
        }
        sp_image_free(model_in);
    }
    printf("Master init model end \n");
}

void emc_helper:: model_init(Configuration conf, sp_3matrix *model){
    char buffer[256];
    sprintf(buffer,"/home/jing.liu/EMCMPI/build/output/model_0149.h5");
    printf("read model from file %s\n",buffer);
    const int N_model = conf.model_side*conf.model_side*conf.model_side;
    Image *model_in = sp_image_read(buffer,0);
    for (int i = 0; i < N_model; i++) {
        model->data[i] = sp_cabs(model_in->image->data[i]);
    }
    sp_image_free(model_in);
}

void  emc_helper::model_init(Configuration conf, sp_3matrix * model, int iteration ){
    char buffer[256];
    sprintf(buffer,"outputn11/model_%.4d.h5", iteration);
    printf("read model from file %s\n",buffer);
    const int N_model = conf.model_side*conf.model_side*conf.model_side;
    Image *model_in = sp_image_read(buffer,0);
    /*if (conf.model_side != sp_image_x(model_in) ||
            conf.model_side != sp_image_y(model_in) ||
            conf.model_side != sp_image_z(model_in)) {
        printf("Input model is of wrong size.\n");
        exit(1);
    }*/
    for (int i = 0; i < N_model; i++) {
        model->data[i] = sp_cabs(model_in->image->data[i]);
    }
    sp_image_free(model_in);
}


void emc_helper::mask_model(int N_images,  sp_matrix ** images, sp_imatrix ** masks, int N_2d){
    for (int i_image = 0; i_image < N_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (masks[i_image]->data[i] == 0) {
                images[i_image]->data[i] = -1.0;
            }
        }
    }
}


//sum up ori and tmp to ori
void emc_helper :: sum_vectors(real* ori, real* tmp, int len){
    for(int i = 0; i< len; i++)
        ori[i] += tmp[i];
}
//take the log of vector a
void emc_helper::log_vector(real* a, int len ){
    for(int i = 0; i< len; i++)
        a[i]= log(a[i]);
}
void emc_helper::devide_part(real* data, int len, int part){
    if(part <=0 )
        part =1;
    for(int i = 0; i<len; i++)
        data[i]/=part;
}

void emc_helper::minus_vector(real* sub_out,real* min, int len)
{
    for(int i =0; i<len; i++)
        sub_out[i]= min[i]-sub_out[i];
}

void emc_helper::set_vector(real *dst, real* ori, int len){
    for(int i =0; i<len; i++)
        dst[i]= ori[i];
}
