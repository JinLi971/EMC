/*
 * Author : Jing Liu  @ Biophysics & TDB of Uppsala University
 * Date 2013-1-16
 * File : config.cpp
 * Description:  structures and functions for configuration, by default it reads emc.conf
 */
#include "configs.h"
#include <libconfig.h>
#include <signal.h>
#include <sys/stat.h>
#include<cstdlib>
#include<string.h>
Config::Config()
{
}
ConfigD Config::Get_Distribute_Config(Configuration conf){
    ConfigD confD;
    confD.model_side = conf.model_side;
    confD.slice_chunk = conf.slice_chunk;
    confD.N_images = conf.N_images;
    confD.rotations_n = conf.rotations_n;
    confD.max_iterations = conf.max_iterations;
    confD.known_intensity = conf.known_intensity;
    confD.isDebug = conf.isDebug;
    confD.nGPUs = conf.nGPUs;
    confD.diff = (int)conf.diff;
    confD.pixel_size = conf.pixel_size;
    confD.detector_size = conf.detector_size;
    confD.sigma_half_life = conf.sigma_half_life;
    confD.wavelength = conf.wavelength;
    confD.detector_distance = conf.detector_distance;
    confD.sigma_final = conf.sigma_final;
    confD.sigma_start = conf.sigma_start;
    confD.exclude_images = conf.exclude_images;
    confD.exclude_ratio = conf.exclude_ratio;
    confD.model_blur = conf.model_blur;
    return confD;
}

Configuration Config::read_configuration_file(const char *filename) {
    Configuration config_out;
    config_t config;
    config_init(&config);
    if (!config_read_file(&config,filename)) {
        fprintf(stderr,"%d - %s\n",
                config_error_line(&config),
                config_error_text(&config));
        config_destroy(&config);
        exit(1);
    }
    config_lookup_int(&config,"model_side",&config_out.model_side);
    config_lookup_int(&config,"read_stride",&config_out.read_stride);
    config_lookup_float(&config,"wavelength",&config_out.wavelength);
    config_lookup_float(&config,"pixel_size",&config_out.pixel_size);
    config_lookup_int(&config,"detector_size",&config_out.detector_size);
    config_lookup_float(&config,"detector_distance",&config_out.detector_distance);
    config_lookup_int(&config,"rotations_n",&config_out.rotations_n);
    config_lookup_float(&config,"sigma_start",&config_out.sigma_start);
    config_lookup_float(&config,"sigma_final",&config_out.sigma_final);
    config_lookup_int(&config,"sigma_half_life",&config_out.sigma_half_life);
    config_lookup_int(&config,"slice_chunk",&config_out.slice_chunk);
    config_lookup_int(&config,"N_images",&config_out.N_images);
    config_lookup_int(&config,"max_iterations",&config_out.max_iterations);
    config_lookup_bool(&config,"blur_image",&config_out.blur_image);
    config_lookup_float(&config,"blur_sigma",&config_out.blur_sigma);
    config_lookup_string(&config,"mask_file",&config_out.mask_file);
    config_lookup_string(&config,"image_prefix",&config_out.image_prefix);
    config_lookup_bool(&config,"normalize_images",&config_out.normalize_images);
    config_lookup_bool(&config,"known_intensity",&config_out.known_intensity);
    config_lookup_int(&config,"model_input",&config_out.model_input);
    config_lookup_string(&config,"model_file",&config_out.model_file);
    config_lookup_bool(&config,"exclude_images",&config_out.exclude_images);
    config_lookup_float(&config,"exclude_ratio",&config_out.exclude_ratio);
    char *diff_type_string = (char *) malloc(20*sizeof(char));
    config_lookup_string(&config,"diff_type",(const char **)(&diff_type_string));
    if (strcmp(diff_type_string, "absolute") == 0) {
        config_out.diff = absolute;
    } else if (strcmp(diff_type_string, "poisson") == 0) {
        config_out.diff = poisson;
    } else if (strcmp(diff_type_string, "poisson") == 0) {
        config_out.diff = relative;
    }
    config_lookup_float(&config,"model_blur",&config_out.model_blur);
    config_lookup_int(&config,"isDebug", &config_out.isDebug);
    config_lookup_int(&config,"isOutput", &config_out.isOutput);
    config_lookup_int(&config,"output_loop", &config_out.output_loop);
    config_lookup_int(&config,"output_period", &config_out.output_period);
    config_lookup_int(&config,"nGPU", &config_out.nGPUs);
    return config_out;
}
