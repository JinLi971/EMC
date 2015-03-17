/*
 * Author : Jing Liu  @ Biophysics & TDB of Uppsala University
 * Date 2013-1-16
 * File : config.h
 * Description:  structures and functions for configuration, by default it reads emc.conf
 */

#ifndef CONFIGS_H
#define CONFIGS_H

enum diff_type {absolute=0, poisson=1, relative=2};
typedef struct{
    /* various data dimensions */
    int model_side;    // the size of model is model_size * model_size
    // *model_size. #64*64 or 128*128
    int read_stride;   // binning number # 2,4

    /* the number of slices in one chunk. Due to the large number of
      * rotations, the rotational probability table will be too large to
      * fit into GPU memory, so that slices will be cut into chunks. */
    int slice_chunk;
    int N_images; // the number of diffraction images.
    int max_iterations; // the maximal number of iterations
    int rotations_n;    // the number of rotations is a function of n, #
    // 8 10 12

    /* detector characteristics */
    double wavelength; // wave length of xray, got from cheetah
    double pixel_size; // magnification of image, got from cheetah
    int detector_size; // the size of detector is
    // detector_size*detector_size, got from cheetah
    // # 1024*1024
    double detector_distance; // the distance between detector, got from
    // cheetah

    /*
      * sigma_start, sigma_end and sigma_half_life, are used to compute
      * how similar a slice and an image is. This similarity will be
      * measured in a gaussian distribution, with an expected value E[Si
      * - Ki] = 0, and a dynamic sigma. In the begining, slices and
      * images differ a lot, so a larger sigma is needed. After a few
      * iterations, it should be smaller. Specifically,
      *
      * sigma = sigma_final +
      * (sigma_start-sigma_final)*exp(-iteration/sigma_half_life*log(2));
      */
    double sigma_start;
    double sigma_final;
    int sigma_half_life;

    /* model for rotational probability */
    enum diff_type diff;

    int blur_image; // defines if blur is needed for input images # true or false.
    double blur_sigma; // if blur is needed, then a gaussian blur is
    // applied with blur_sigma.
    const char *mask_file; // path to the mask file
    const char *image_prefix; // path to the image folder
    int normalize_images; // defines if images need to normalized before
    // EMC, # true or false

    /* defines if the model intensity is known or not.  if not, needs
      * to update scaling # true or false, false usually  */
    int known_intensity;
    /* defines how the model will be initialized:
      *
      * 0: "uniform density model"
      * 1: random orientations model
      * 2: model read from file
      * 3: radial average mode
      */
    int model_input;
    const char *model_file; // if model_input = 2, model_file is the
    // path to model file,
    int exclude_images; // defines if any images should be excluded from
    // dataset # true or false
    /* if exclude_images is true, then all the image which is not in
        exclude_ratio, will be excluded */
    double exclude_ratio;
    double model_blur; // the gaussian blur sigma for the model blur
    int isDebug; // control debug option
    int isOutput; // control output option
    int output_loop; //if isAllOutput = true, then for every output_loop save a temp result
    int output_period;
    int nGPUs;//to checked
}Configuration;




typedef struct{
    /* various data dimensions */
    int model_side;    // the size of model is model_size * model_size
    // *model_size. #64*64 or 128*128
    int slice_chunk;
    int N_images; // the number of diffraction images.
    int rotations_n;    // the number of rotations is a function of n, #
    int max_iterations; // the maximal number of iterations
    int detector_size; // the size of detector is
    // detector_size*detector_size, got from cheetah
    // # 1024*1024
    int sigma_half_life;
    int known_intensity;
    /* model for rotational probability */
    int diff;
    int exclude_images; // defines if any images should be excluded from
    int isDebug;
    int nGPUs;
    double exclude_ratio;
    double model_blur; // the gaussian blur sigma for the model blur
    double wavelength; // wave length of xray, got from cheetah
    double pixel_size; // magnification of image, got from cheetah
    double detector_distance; // the distance between detector, got from
    // cheetah
    double sigma_start;
    double sigma_final;
}ConfigD;




class Config
{
public:
    Config();
    //read configuration file from filename (include path to file)
    Configuration read_configuration_file(const char *filename);
    ConfigD Get_Distribute_Config(Configuration conf);
};

#endif // CONFIGS_H
