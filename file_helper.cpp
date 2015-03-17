/*
 *File : file_helper.cpp
 *Author: Jing Liu @ TDB & Biophysics, Uppsala University, 2013
 *Description: File common operators for emc algorithm
*/
#include "file_helper.h"
//#include <string.h>
//using namespace std;
file_helper::file_helper()
{
}

//write_respons to debug/resopns_XXXX.data Mdata*Mrot
void file_helper::write_respons(real* res, int N_slices, int N_images, int iteration){
    char buffer[256];
    int len = N_slices*N_images;
    sprintf(buffer,"debug/respons_%.4d.data", iteration);
    FILE* respons_file = fopen(buffer,"wb+");
    /*for (int i = 0; i<len; i=i+N_images){
        for(int j = 0 ; j<N_images; j++){
            fprintf(respons_file," %e  ",(double)res[i+j]);
        }
        fprintf(respons_file,"%s","\n");
    }*/
    fwrite(res, sizeof(real), N_slices * N_images,respons_file);
    fclose(respons_file);
}

//write_respons to debug/respons_XXXX.data Mdata* Mrot. for Multi-GPUs version
void file_helper::write_respons(real** res, int* N_slices, int N_images, int iteration,int ntasks ){
    char buffer[256];
    sprintf(buffer,"debug/respons_%.4d.data", iteration);
    FILE* respons_file = fopen(buffer,"wb+");
    /*for(int t = 0; t< ntasks; t++){
        int len = N_slices[t] * N_images;
        for (int i = 0; i<len; i=i+N_images){
            for(int j = 0 ; j<N_images; j++){
                fprintf(respons_file,"  %e  ",(double)res[t][i+j]);
            }
            fprintf(respons_file,"%s","\n");
        }
    }*/
    //printf("respons %f %f %f %f\n",res[0][0],res[0][1],res[1][0],res[1][1]);
    for(int i = 0; i<ntasks; i++)
        fwrite(res[i],sizeof(real),N_slices[i]*N_images,respons_file);
    fclose(respons_file);
}

//write 3d intensity model to output/model_XXXX.h5. model is filtered by weight.
void file_helper::write_model(sp_3matrix * model, sp_3matrix * weight, int iteration){
    int N_model = model->x *model->y*model->z;
    char buffer[256];
    Image *model_out = sp_image_alloc(model->x ,model->y,model->z);
    for (int i = 0; i < N_model; i++) {
        if (weight->data[i] > 0.0 && model->data[i] > 0.) {
            model_out->mask->data[i] = 1;
            model_out->image->data[i] = sp_cinit(model->data[i],0.0);
        } else {
            model_out->mask->data[i] = 0;
            model_out->image->data[i] = sp_cinit(0., 0.);
        }
    }
    sprintf(buffer,"output/model_%.4d.h5", iteration);
    sp_image_write(model_out,buffer,0);
}

void file_helper::write_model(sp_3matrix * model, sp_3matrix * weight, int iteration, int validation){
    int N_model = model->x *model->y*model->z;
    char buffer[256];
    Image *model_out = sp_image_alloc(model->x ,model->y,model->z);
    for (int i = 0; i < N_model; i++) {
        if (weight->data[i] > 0.0 && model->data[i] > 0.) {
            model_out->mask->data[i] = 1;
            model_out->image->data[i] = sp_cinit(model->data[i],0.0);
        } else {
            model_out->mask->data[i] = 0;
            model_out->image->data[i] = sp_cinit(0., 0.);
        }
    }
    sprintf(buffer,"/scratch/fhgfs/jing/validation/output_acc/model_%.4d_%.4d.h5", iteration,validation);
    sp_image_write(model_out,buffer,0);
}

//debug function. write model to a specified name.
void write_model_by_name(sp_3matrix * model, sp_3matrix * weight, int iteration, std::string filename){
    int N_model = model->x *model->y*model->z;
    //printf("N is %d", N_model);
    char buffer[256];
    Image *model_out = sp_image_alloc(model->x ,model->y,model->z);
    for (int i = 0; i < N_model; i++) {
        if (weight->data[i] > 0.0 && model->data[i] > 0.) {
            model_out->mask->data[i] = 1;
            model_out->image->data[i] = sp_cinit(model->data[i],0.0);
        } else {
            model_out->mask->data[i] = 0;
            model_out->image->data[i] = sp_cinit(0., 0.);
        }
    }
    sprintf(buffer,"output/model_%.4d_%s.h5", iteration, filename.c_str());
    sp_image_write(model_out,buffer,0);
}

//write weight to outpu/weight_XXXX.h5
void file_helper::write_weight( sp_3matrix * weight, int iteration){
    printf("writing model...");
    int N_model = weight->x *weight->y*weight->z;
    char buffer[256];
    Image *model_out = sp_image_alloc(weight->x ,weight->y,weight->z);
    /* write weight */
    for (int i = 0; i < N_model; i++) {
        model_out->image->data[i] = sp_cinit(weight->data[i], 0.);
        model_out->mask->data[i] = 1;
    }
    sprintf(buffer, "output/weight_%.4d.h5", iteration);
    sp_image_write(model_out, buffer, 0);
}

//open file ExeTime.data to store execution time
void file_helper::Init_Time_file(){
    time_file = fopen("ExeTime.data","wb");
}

//write execution time by iteration rank time
void file_helper::write_time( int iteration, int rank, double time){
    fprintf(time_file,"%d %d   %e \n", iteration, rank,  time);
}

//close ExeTime.data file
void file_helper::close_time(){
    fclose(time_file);
}

