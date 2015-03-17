#ifndef FILE_HELPER_H
#define FILE_HELPER_H

#include <iostream>
#include <string>
#include <fstream>
#include <spimage.h>
class file_helper
{
public:
    file_helper();
    FILE* weights_file;
    FILE* rotaion_file;
    FILE* image_file;
    FILE* likelihood_file;
    FILE* fit_file;
    FILE* scaling_file;
    FILE* radial_fit_file;
    FILE* total_res_file;
    FILE* model_file;
    FILE* time_file;
    void Init_Debug_Files();
    void Init_Output_Files();
    void Init_Time_file();
    void write_file(char* , FILE* );
    void write_model(sp_3matrix * model, sp_3matrix * weight, int);
    void write_model(sp_3matrix * model, sp_3matrix * weight, int iteration,int validation);
    void write_time(int iteration, int rank, double time);
    void write_weight( sp_3matrix * weight, int iteration);
    void close_time();
    void write_respons(real* res, int N_slices, int N_images, int iteration );
    void write_respons(real** res, int* N_slices, int N_images, int iteration,int ntasks );
    void write_model_by_name(sp_3matrix * model, sp_3matrix * weight, int iterations, std::string filename);
};

#endif // FILE_HELLPER_H
