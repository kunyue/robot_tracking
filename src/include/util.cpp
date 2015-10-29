#include "util.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
using std::string;

int readCalibrationResult(char* filename, cv::Mat& K, cv::Mat& distCoeff, int& image_width, int& image_height)
{
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {return -1;}
	fs["camera_matrix"] >> K;
	fs["distortion_coefficients"] >> distCoeff;
	fs["image_width"] >> image_width;
	fs["image_height"] >> image_height;
	return 0;
}



double normalize_angle_radian(double ang)
{
	while(ang > M_PI)
	{
		ang -= 2*M_PI;
	}
	while(ang < -M_PI)
	{
		ang += 2*M_PI;
	}
	return ang;
}


