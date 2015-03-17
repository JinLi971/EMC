/*
 * Author : Jing Liu@ Biophysics and TDB
 * Modified to collective reduce, to C version
 * Time getdaytime
*/

#include <gsl/gsl_rng.h>
#include <time.h>
#include "configs.h"
#include <iostream>
#include "emc_helper.h"
#include <stdio.h>
#ifdef IN_PLACE
#undef IN_PLACE
#undef BOTTOM
#endif
#include "mpi_helper.h"
#include "file_helper.h"
#include "emc_cuda_help.h"
#include "timer_helper.h"
#include <mpi.h>
#include <spimage.h>
using namespace std;

int compare_real(const void *pa, const void *pb) {
    real a = *(const real*)pa;
    real b = *(const real*)pb;
    if (a < b) {return -1;}
    else if (a > b) {return 1;}
    else {return 0;}
}


int main(int argc, char *argv[]){
    //cout<<"real: "<<sizeof(real)<<"double: "<<sizeof(double)<<"float: "<<sizeof(float)<<endl;
    Configuration conf_R;
    timer_helper th;
    /*-----------------------------------------------------------Do Master Node Initial Job start------------------------------------------------*/
    cout<<"Init MPI...";
    MPI_Init(&argc, &argv);
    //clock_t now;
    unsigned long int now, before; //in milisecond
    now = th.gettimenow();
    int taskid, ntasks;
    int master = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
    cout<<taskid <<" MPI started!"<<endl;
    Config confClass ;
    emc_helper emcHelp;
    MPIHelper mpiHelp ;
    file_helper fh;
    ConfigD conf;
    sp_imatrix ** masks;
    sp_matrix ** images;
    sp_imatrix *mask;
    sp_matrix *x_coordinates;
    sp_matrix *y_coordinates;
    sp_matrix *z_coordinates;
    sp_3matrix *model;
    sp_3matrix *model_weight;
    //mpiHelp.NewAllTypes(conf.model_side);
    mpiHelp.Create_Configuration();
    //read configuration and broadcast it
    if(taskid == master){
        if (argc > 1) {
            conf_R = confClass.read_configuration_file(argv[1]);
        } else {
            conf_R = confClass.read_configuration_file("../emc.conf");
        }
        conf = confClass.Get_Distribute_Config(conf_R);
        fh.Init_Time_file();   
    }
    mpiHelp.Broadcast_Config(&conf,master);
    //model data collective from all GPUs
    real  model_data[conf.model_side*conf.model_side*conf.model_side];
    //model_weight data collective from all GPUs
    real  model_weight_data[conf.model_side*conf.model_side*conf.model_side];
    model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
    model_weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
    x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    images= (sp_matrix **) malloc(conf.N_images*sizeof(sp_matrix *));
    mask = sp_imatrix_alloc(conf.model_side,conf.model_side);
    masks= (sp_imatrix **)malloc(conf.N_images*sizeof(sp_imatrix *));
    for(int i=0;i<conf.N_images; i++){
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);

    }
    if(taskid == master){
        images = emcHelp.read_images(conf_R,masks);
        //images=emcHelp.read_images_from_models(conf_R,masks);
        mask = emcHelp.read_mask(conf_R);
        if (conf_R.normalize_images) {
            emcHelp.normalize_images_central_part(images, mask, 20., conf_R);
        }
        emcHelp.calculate_coordinates(conf.model_side,conf.pixel_size, conf.detector_distance, conf.wavelength,
                                      x_coordinates, y_coordinates, z_coordinates);

        const int N_model = conf.model_side*conf.model_side*conf.model_side;
        for (int i = 0; i < N_model; i++) {
            model->data[i] = 0.0;
            model_weight->data[i] = 0.0;
        }
        emcHelp.model_init(conf_R,model, model_weight,images,mask, x_coordinates, y_coordinates, z_coordinates);
        emcHelp.mask_model(conf.N_images,images,masks, conf.model_side*conf.model_side);
    }

    /*-----------------------------------------------------------Do Master Node Initial Job end------------------------------------------------*/
    /*-----------------------------------------------------------Init MPI and distribute data start --------------------------------------*/
    mpiHelp.Broadcast_Coordinate(z_coordinates,master);
    mpiHelp.Broadcast_Coordinate(x_coordinates,master);
    mpiHelp.Broadcast_Coordinate(y_coordinates,master);
    mpiHelp.Broadcast_Images(images,conf.N_images,master);
    mpiHelp.Broadcast_Mask(mask,master);
    mpiHelp.Broadcast_Masks(masks,conf.N_images,master);
    mpiHelp.Broadcast_Model(model,master);
    mpiHelp.Broadcast_Weight(model_weight,master);
    MPI_Barrier(MPI_COMM_WORLD);
    /*-----------------------------------------------------------Init MPI and distribute data end --------------------------------------*/
    /*-----------------------------------------------------------Do local Init start-----------------------------------------------------------*/
    //set GPUs if only one GPU, use this, otherwise use cuda_set_device(taskid)
    //cuda_choose_best_device();
    //cuda_print_device_info();
    cuda_set_device(taskid % conf.nGPUs,taskid);
    const int N_images = conf.N_images;
    const int slice_chunk = conf.slice_chunk;
    const int N_2d = conf.model_side*conf.model_side;
    const int N_3d = conf.model_side*conf.model_side*conf.model_side;
    Quaternion **rotations;
    const int n = conf.rotations_n;
    real *weights_rotation;
    real **weightsPtr_rotation = &weights_rotation;
    const int N_slices = generate_rotation_list(n,&rotations,(static_cast<real **>(weightsPtr_rotation)));
    /*------------------------------task devision of respons------------------------------------*/
    int *lens;
    //int* offsets;
    lens = (int *)malloc(sizeof(int)*ntasks);
    int slice_start = taskid* N_slices/ntasks;
    int slice_backup = slice_start;
    int slice_end =  (taskid+1)* N_slices/ntasks;
    if (taskid == ntasks -1)
        slice_end = N_slices;
    int allocate_slice = slice_end - slice_backup;
    for (int i = 0; i<ntasks-1; i++){
        lens[i] = emcHelp.get_allocate_len(ntasks,N_slices,i);
        //  off+=lens[i];
        // offsets [i+1] = off;
        cout <<"lens " << i <<" " << lens[i];
    }
    lens[ntasks-1] = emcHelp.get_allocate_len(ntasks,N_slices,ntasks-1);
    cout <<"lens " << ntasks-1 <<" " << lens[ntasks-1];

    /*------------------------------task devision of respons------------------------------------*/

    real *d_weights_rotation;
    cuda_allocate_real(&d_weights_rotation, allocate_slice);
    cuda_copy_weight_to_device(weights_rotation, d_weights_rotation, allocate_slice,taskid);
    real * d_rotations;
    cuda_allocate_rotations(&d_rotations,rotations,slice_start,slice_end,taskid);
    real* d_sum;
    cuda_allocate_real(&d_sum,N_images);
    real h_sum_vector[N_images];
    real* sum_vector = &h_sum_vector[0];
    real* d_maxr;
    cuda_allocate_real(&d_maxr,N_images);
    real h_maxr[N_images];
    real* maxr = &h_maxr[0];
    real tmpbuf_images[N_images];//MPI recv buff
    real* tmpbuf_images_ptr = &tmpbuf_images[0];
    real *respons= (real*) malloc(allocate_slice*N_images*sizeof(real));
    //real *h_scaling = (real*) malloc(allocate_slice*N_images*sizeof(real));

    real * d_slices;
    cuda_allocate_slices(&d_slices,conf.model_side,slice_chunk);
    real * d_model;
    cuda_allocate_model(&d_model,model);
    cuda_normalize_model(model, d_model);
    real * d_model_updated;
    real * d_model_tmp;
    cuda_allocate_model(&d_model_updated,model);
    real * d_model_weight; //used to d_weight
    cuda_allocate_model(&d_model_weight,model_weight);
    int * d_mask;
    cuda_allocate_mask(&d_mask,mask);
    real * d_x_coord;
    real * d_y_coord;
    real * d_z_coord;
    cuda_allocate_coords(&d_x_coord, &d_y_coord, &d_z_coord, x_coordinates,
                         y_coordinates,  z_coordinates);
    real * d_images;
    cuda_allocate_images(&d_images,images,N_images);
    //int * d_masks;
    //cuda_allocate_masks(&d_masks,masks,N_images);
    real * d_respons;
    cuda_allocate_real(&d_respons,allocate_slice*N_images);
    real * d_scaling;
    cuda_allocate_real(&d_scaling,N_images*allocate_slice);
    //real *d_weighted_power;
    //cuda_allocate_real(&d_weighted_power,N_images);
    //real *d_fit;
    //cuda_allocate_real(&d_fit,N_images);
    //real *d_fit_best_rot;
    //cuda_allocate_real(&d_fit_best_rot, N_images);
    int *active_images = (int*)malloc(N_images*sizeof(int));
    int *d_active_images;
    cuda_allocate_int(&d_active_images,N_images);
    //real *d_radius;
    //cuda_allocate_real(&d_radius, N_2d);
    //cuda_copy_real_to_device(radius->data, d_radius, N_2d);
    //real *d_radial_fit;
    //real *d_radial_fit_weight;
    //cuda_allocate_real(&d_radial_fit, conf.model_side/2);
    //cuda_allocate_real(&d_radial_fit_weight, conf.model_side/2);
    //int *d_best_rotation;
    //cuda_allocate_int(&d_best_rotation, N_images);
    //real * d_masked_images;
    //cuda_allocate_images(&d_masked_images,images,N_images);
    real sigma;
    int current_chunk;
    int start_iteration =0;

    /*-----------------------------------------------------------Do local Init END-----------------------------------------------------------*/

    /*------------------------------------EMC Iterations start -----------------------------------------------------------------------*/
    cout << "isDebug "<<conf.isDebug<< endl;

    //update time used in initialization.
    if(taskid ==master){
        before = now;
        now = th.gettimenow();
        double timeinterval = th.update_time(before,now);
        fh.write_time(-1,master, timeinterval);
        cout<<"time updated for pre-loading, uses " <<timeinterval<<"seconds" <<endl;

    }
    for (int iteration = start_iteration; iteration < conf.max_iterations; iteration++) {
        cout<<"Iteration "<<iteration<<"starts at rank "<<taskid<<endl;
        /*---------------------- reset local variable-------------------------------*/
        cuda_reset_real(d_sum, N_images);
        cuda_reset_real(d_maxr, N_images);
        cuda_reset_real(d_respons, N_images*allocate_slice);
        emcHelp.reset_real(tmpbuf_images,N_images);
        emcHelp.reset_real(maxr, N_images);
        emcHelp.reset_real(sum_vector,N_images);
        /*---------------------- reset local variable done-------------------------*/

        //compute flunence (scaling)
        sigma = conf.sigma_final + (conf.sigma_start-conf.sigma_final)*exp(-iteration/(float)conf.sigma_half_life*log(2.));
        real sum = cuda_model_max(d_model, N_3d);
        cout<<"model max = " <<sum <<" model average is ";
        sum = cuda_model_average(d_model, N_3d);
        cout<<sum <<" at rank "<<taskid<<endl;
        /*generation of new images*/
        /* char buffer[1000];
        *        real* slices = (real*) malloc(sizeof(real) * N_2d * N_slices);
        for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= slice_end) {
                current_chunk = slice_end - slice_start;
            } else {
                current_chunk = slice_chunk;
            }
            int current_start = slice_start- slice_backup;
            cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
                            current_start   , current_chunk);
            cuda_copy_real_to_host(slices[current_start],d_slices,current_chunk*N_2d);
            Image *slice_out_img = sp_image_alloc(conf.model_side, conf.model_side, 1);
            for(int co = 0; co<current_chunk; co++){
                for(int pi = 0; pi<N_2d; pi++){
                    slice_out_img ->image->data[pi] = sp_cinit(slices[(co+current_start)*N_2d + pi],0);
                }
                sprintf(buffer, "%s/best_slice_%.4d.h5", "output", co+current_start);
                sp_image_write(slice_out_img, buffer, 0);

            }
        }
        exit(0);

*/

        if (conf.known_intensity == 0) {
            for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
                if (slice_start + slice_chunk >= slice_end) {
                    current_chunk = slice_end - slice_start;
                } else {
                    current_chunk = slice_chunk;
                }
                int current_start = slice_start- slice_backup;
                cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
                                current_start   , current_chunk);
                cuda_update_scaling_full(d_images, d_slices, d_mask, d_scaling, N_2d, N_images, current_start, current_chunk, diff_type(conf.diff));
            }
        }
        cout << "Update scaling done! at "<<taskid<<endl;
        /*//Write scaling
        for(int i =0; i<ntasks; i++)
            cout <<"lens[i] " << lens[i] <<endl;
        if(taskid!=master){
            if(conf.isDebug){
                cuda_copy_real_to_host(h_scaling,d_scaling,allocate_slice*N_images);
                mpiHelp.Send_Respons(h_scaling,master,allocate_slice*N_images);
            }
        }
        if(taskid==master){
            real ** allRespons = (real**) malloc (sizeof (real*)*ntasks);
            for (int i = 1; i<ntasks; i++){
                if (conf.isDebug){
                    allRespons[i] = mpiHelp.Recv_Respons(lens[i],N_images);
                }
            }
            if(conf.isDebug){
                cuda_copy_real_to_host(h_scaling,d_scaling,allocate_slice*N_images);
                allRespons[0]=h_scaling;
                fh.write_respons(allRespons,lens,N_images,iteration+300, ntasks);
                cout <<"write respons " << iteration+300<< " done!" <<endl;
            }
            delete allRespons
        }*/



        ///start calculate many fits
        // int radial_fit_n = 1;
        // cuda_set_to_zero(d_fit,N_images);
        // cuda_set_to_zero(d_radial_fit,conf.model_side/2);
        // cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2);
        /*  for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= slice_end) {
                current_chunk = slice_end - slice_start;
            } else {
                current_chunk = slice_chunk;
            }
            int current_start = slice_start- slice_backup;
            cuda_get_slices(model,d_model,d_slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,
                            current_start,current_chunk);
            cuda_calculate_fit(d_slices, d_images, d_mask, d_scaling,
                               d_respons, d_fit, sigma, N_2d, N_images,
                               current_start, current_chunk);
           if (iteration % radial_fit_n == 0 && iteration != 0) {
                cuda_calculate_radial_fit(slices, d_images, d_mask,
                                          d_scaling, d_respons, d_radial_fit,
                                          d_radial_fit_weight, d_radius,
                                          N_2d, conf.model_side, N_images,
                                          current_start, current_chunk);
            }
        }*/

        //Calculate respons_self
        for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= slice_end) {
                current_chunk = slice_end - slice_start;
            } else {
                current_chunk = slice_chunk;
            }
            int current_start = slice_start- slice_backup;
            cuda_get_slices(model,d_model,d_slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,
                            current_start,current_chunk);
            cuda_calculate_responsabilities(d_slices, d_images, d_mask,
                                            sigma, d_scaling, d_respons, d_weights_rotation,
                                            N_2d, N_images, current_start,
                                            current_chunk, diff_type(conf.diff));

        }
        //cuda_copy_real_to_host(respons,d_respons,N_slices*N_images);
        cuda_max_vector(d_respons, N_images, allocate_slice,d_maxr);
        cuda_copy_real_to_host(maxr,d_maxr,N_images);
        //cout << "local maxr at" << taskid;
        //for(int i =0; i<N_images; i++)
         //   cout << maxr[i] <<" ";
        //cout <<endl;
        mpiHelp.Global_Allreduce(maxr, tmpbuf_images,N_images,MPI_MAX);
        //mpiHelp.set_vector(maxr,tmpbuf_images,N_images);
        memcpy((void*)maxr,  tmpbuf_images,N_images* sizeof(real));
        //cout << "gobal maxr at" << taskid;
        //for(int i =0; i<N_images; i++)
         //   cout << maxr[i] <<" ";
        //cout <<endl;
        cuda_copy_real_to_device(maxr, d_maxr, N_images);
        //take -max expf and get local sum
        emcHelp.reset_real(tmpbuf_images_ptr,N_images);

        cuda_respons_max_expf(d_respons,d_maxr,N_images, allocate_slice, d_sum);
        cuda_copy_real_to_host(sum_vector,d_sum,N_images);
        mpiHelp.Global_Allreduce(sum_vector,tmpbuf_images_ptr,N_images,MPI_SUM);
        memcpy((void*)sum_vector,(const void*) tmpbuf_images_ptr, N_images* sizeof(real));
        emcHelp.log_vector(sum_vector,N_images);
        emcHelp.sum_vectors(sum_vector,maxr,N_images);

        cuda_copy_real_to_device(sum_vector,d_sum,N_images);
        cuda_norm_respons_sumexpf(d_respons,d_sum,maxr,N_images,allocate_slice);

        // output debugging
        if(conf.isDebug && taskid !=master){
            cuda_copy_real_to_host(respons,d_respons,allocate_slice*N_images);
            mpiHelp.Send_Respons(respons,master,allocate_slice*N_images,taskid);
            cout << "respons 1 " << respons[0] << respons[1] << endl;
        }
        if(conf.isDebug&&taskid==master){
            real ** allRespons = (real**) malloc (sizeof (real*)*ntasks);
            real* tmp;
            int recv_rank;
            for (int i = 1; i<ntasks; i++){
                tmp = mpiHelp.Recv_Respons(lens,N_images,&recv_rank);
                allRespons[recv_rank]=tmp;
                cout << "respons 0 " << tmp[0] << tmp[1] << endl;

            }
            cuda_copy_real_to_host(respons,d_respons,allocate_slice*N_images);
            allRespons[0]=respons;
            cout << "respons 0 "
                 << allRespons[0][0]<<" "<< allRespons[0][1]<<respons[0] <<" " << respons[1]<< endl;
            fh.write_respons(allRespons,lens,N_images,iteration, ntasks);
            delete allRespons;
            //delete tmp;
        }


        /*-----------------------------respons normalization (distribution ) done-------------------------------*/
        if (iteration == 0) {
            for (int i_image = 0; i_image < N_images; i_image++) {
                active_images[i_image] = 1;
            }
        }
        /*if (conf.exclude_images == 1 && iteration > -1) {
                   real *fit_copy =(real*) malloc(N_images*sizeof(real));
                   memcpy(fit_copy,fit,N_images*sizeof(real));
                   qsort(fit_copy, N_images, sizeof(real), compare_real);
                   real threshold = fit_copy[(int)((real)N_images*conf.exclude_ratio)];
                   for (int i_image = 0; i_image < N_images; i_image++) {
                       if (fit[i_image] > threshold) {
                           active_images[i_image] = 1;
                       } else {
                           active_images[i_image] = 0;
                       }
                   }
               }*/
        cuda_copy_int_to_device(active_images, d_active_images, N_images);


        /* start update model */
        cuda_reset_model(model,d_model_updated);
        cuda_reset_model(model_weight,d_model_weight);
        if(iteration < conf.max_iterations){
            cout <<"UPDATE MODEL!" <<endl;
            for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
                if (slice_start + slice_chunk >= slice_end) {
                    current_chunk = slice_end - slice_start;
                } else {
                    current_chunk = slice_chunk;
                }
                int current_start = slice_start- slice_backup;
                cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
                                current_start, current_chunk);
                cuda_update_slices(d_images, d_slices, d_mask,
                                   d_respons, d_scaling, d_active_images,
                                   N_images, current_start, current_chunk, N_2d,
                                   model,d_model_updated, d_x_coord, d_y_coord,
                                   d_z_coord, &d_rotations[current_start * 4],
                        d_model_weight,images);
            }
            d_model_tmp = d_model_updated;
            d_model_updated = d_model;
            d_model = d_model_tmp;
        }
        cuda_copy_model(model,d_model);
        cuda_copy_model(model_weight,d_model_weight);
        mpiHelp.Global_Allreduce(model->data,model_data,N_3d,MPI_SUM);
        mpiHelp.Global_Allreduce(model_weight->data,model_weight_data,N_3d,MPI_SUM);
        emcHelp.devide_part(model_data,N_3d,ntasks);
        emcHelp.devide_part(model_weight_data,N_3d,ntasks);
        memcpy(model->data,model_data,sizeof(real)*N_3d);
        memcpy(model_weight->data,model_weight_data,N_3d*sizeof(real));
        cuda_copy_model_2_device(&d_model,model);
        cuda_copy_model_2_device(&d_model_weight,model_weight);
        cuda_divide_model_by_weight(model, d_model, d_model_weight);
        cuda_normalize_model(model, d_model);
        cuda_copy_model(model,d_model);
        cuda_copy_model(model_weight,d_model_weight);
        /*--------------model output ----------------------------------*/
        if(taskid ==master){
            before = now;
            now = th.gettimenow();
            double t = th.update_time(before, now);
            fh.write_time(iteration,master,t);
            cout<<"time updated for iteration " <<iteration <<", uses " <<t<<"seconds for master"  <<endl;
            if(conf_R.isOutput)
            {
                if(conf_R.output_loop ){
                    cout<<"output and output loop " << conf_R.output_period<<endl;
                    if(iteration%conf_R.output_period==0){
                        fh.write_model(model,model_weight,iteration);
                        fh.write_weight(model_weight,iteration);
                    }
                }
                else{
                    fh.write_model(model,model_weight,iteration);
                    fh.write_weight(model_weight,iteration);
                }
            }

            if(conf.isDebug ==4 && ntasks >1 && taskid == master){
                emcHelp.model_init(conf_R,model,iteration);
            }
        }
        if(conf.isDebug ==4){
            COMM_WORLD.Barrier();

            mpiHelp.Broadcast_Model(model,master);
            cuda_copy_model_2_device (&d_model, model);
            COMM_WORLD.Barrier();
        }
        cout << "EMC iteration " << iteration << "done at" <<
                taskid<< endl;
    }

    cout<<"EMC done ! "<<endl;
    MPI::Finalize();
    return 0;
}




