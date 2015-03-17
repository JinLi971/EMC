#ifndef EMC_HELPER_H
#define EMC_HELPER_H

#include "configs.h"
#include <spimage.h>
#include "rotations.h"
#include <cmath>
class emc_helper
{
public:
    emc_helper();
    //int quit_requested = 0;
    void nice_exit(int);
    void reset_real(real* h_real, int len);
    int compare_real(const void*, const void*);
    void calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength,
                               sp_matrix *x_coordinates,
                               sp_matrix *y_coordinates, sp_matrix *z_coordinates) ;
    void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
                      sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
                      sp_matrix *y_coordinates, sp_matrix *z_coordinates);
    void test_blur() ;
    sp_matrix **read_images(Configuration conf, sp_imatrix **masks);
    sp_matrix **read_images_from_models(Configuration conf, sp_imatrix **masks);
    void read_images_from_models(Configuration conf, sp_imatrix **masks, sp_matrix ** image);

    sp_imatrix *read_mask(Configuration conf);
    void normalize_images(sp_matrix **images, sp_imatrix *mask, Configuration conf);
    void normalize_images_central_part(sp_matrix **images, sp_imatrix *mask, real radius, Configuration conf) ;
    void normalize_images_individual_mask(sp_matrix **images, sp_imatrix **masks,
                                          Configuration conf);
    void  model_init(Configuration conf, sp_3matrix * model,
                     sp_3matrix * weight, sp_matrix ** images, sp_imatrix * mask,
                     sp_matrix *x_coordinates, sp_matrix *y_coordinates, sp_matrix *z_coordinates);
    void  model_init(Configuration conf, sp_3matrix * model, int iteration );
    void  model_init(Configuration conf,sp_3matrix *model);
    void mask_model(int N_images,  sp_matrix ** images, sp_imatrix ** masks,int N_2d);
    void average_Models(sp_3matrix ** models, int partition, int len,  sp_3matrix*  model);
    void show_models(sp_3matrix ** models, int partition, int len);
    void move_real(real* source, real* dest, int start, int len);
    int get_allocate_len(int ntasks, int N_slices, int taskid);
    void copy_real(int len, real* source , real* dst);
    void sum_vectors(real* ori, real* tmp, int len);
    void max_max_vector(real* a, real* b, int len);
    void log_vector(real* a, int len);
    double real_real_validate(real* a, real*b , int len);
    void devide_part(real* data, int len, int part);
    void minus_vector (real* sub_out, real* min, int len);
    void set_vector(real* dst, real* ori, int len);
};

#endif // EMC_HELPER_H
