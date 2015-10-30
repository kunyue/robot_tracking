#include "util.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
using std::string;
using cv::Mat;
using cv::RotatedRect;
using cv::Scalar;
using std::cout;
using std::endl;

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


void drawrect(Mat& img, RotatedRect rect, Scalar color)
{
	Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; ++i)
	{
		line(img, vertices[i], vertices[(i + 1)%4], color, 2, 8);
	}
}

void rect_to_contour(RotatedRect rect, std::vector<Point2f>& contour)
{
	if(contour.size() != 4)
	{
		contour.resize(4);
	}

	Point2f vertices[4];

	rect.points(vertices);
	for (int i = 0; i < 4; ++i)
	{
		contour[i] = vertices[i];
	}
	
	cout << "##contour: " << contour << endl;
}


