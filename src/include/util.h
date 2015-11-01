#ifndef __UTIL_H__
#define __UTIL_H__
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

using namespace cv;

int readCalibrationResult(char* filename, cv::Mat& K, cv::Mat& distCoeff, int& image_width, int& image_height);
double normalize_angle(double angle);
void drawrect(Mat& img, RotatedRect rect, Scalar color);


void rect_to_contour(RotatedRect rect, std::vector<Point>& contour);
//Rect maxRect(RotatedRect rect);

RotatedRect resize_rect(RotatedRect rect, double scale);

template<class TYPE>
double angle(TYPE p1, TYPE p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	double theta = atan2(dy, dx);
	return theta;
}

template<class TYPE>
double distance(TYPE p1, TYPE p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	double dist = sqrt(dx*dx + dy*dy);
	return dist;
}

template<class TYPE>
double cross_product(TYPE p1, TYPE p2)
{
	return p1.x*p2.x + p1.y*p2.y;
}

double normalize_angle_radian(double ang);	


static inline uint64_t get_timestamp()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec * 1000000ULL + t.tv_usec;
}


#endif

