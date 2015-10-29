#ifndef __ROBOTFEATURE_H
#define __ROBOTFEATURE_H
#include <opencv2/opencv.hpp>
using cv::Point2f;

typedef struct 
{
	Point2f shape_center;
	Point2f mass_center;
	Point2f dir_center;
	//double vx, vy, w;
	double area;
	double r, g, b;//mean color
	
}RobotFeature;

double feature_dist(RobotFeature& r1, RobotFeature& r2);	

#endif 
