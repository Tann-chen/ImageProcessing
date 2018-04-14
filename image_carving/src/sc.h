#ifndef SEAMCARVINGCOMP665156
#define SEAMCARVINGCOMP665156

#include <opencv2/opencv.hpp>
#include <vector>

// the function you need to implement - by defaiult it calls seam_carving_trivial
bool seam_carving(cv::Mat& in_image, int new_width, int new_height, cv::Mat& out_image);

bool seam_carving_trivial(cv::Mat& in_image, int new_width, int new_height, cv::Mat& out_image);

void reduce_horizontal_seam(cv::Mat &in_image, cv::Mat &out_image);

void reduce_vertical_seam(cv::Mat &in_image, cv::Mat &out_image);

void compute_energy_by_gradient(cv::Mat& in_image, cv::Mat& magn_img);

std::vector<int> find_horizontal_seam_by_dp(cv::Mat &in_image);

std::vector<int> find_vertical_seam_by_dp(cv::Mat &in_image);


#endif
