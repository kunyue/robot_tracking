#include <string>
#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "robotFeature.h"
#include "Eigen/Dense"

using std::vector;
using std::queue;
using cv::Point;
using cv::Mat;
//using cv::ml::SVM;
//using namespace cv::ml;


void robotSegment(Mat& frame, Mat& mask);
int svmInit(char*  filename1, char* filename2);
int calibInit(char* calib_filename);
int robotTrackInit(char* calib_filename, char*  green_svm_filename, char* red_svm_filename) ;

vector< vector<Point> > findPattern(Mat& bwImg);;

std::vector<Eigen::VectorXd> robotTrack(cv::Mat& frame);

void robotMatch(vector<RobotFeature>& r1, vector<RobotFeature>& r2, vector<std::pair<int, int> >& matches);

std::vector< std::vector<cv::Point> > robotDetect(cv::Mat &frame);
//void calc_features(cv::Mat img, std::vector< std::vector<cv::Point> >& contours, std::vector<RobotFeature>& robot_features);
void calc_features(cv::Mat img, 
	vector< vector<Point> >& contours, 
	std::vector<Point2f>& shape_centers, 
	std::vector<Point2f>& mass_centers, 
	std::vector<Point2f>& dir_centers,
	vector<RobotFeature>& robot_features);

vector< vector<Point> > effective_contourPoly(vector< vector<Point> >& all_contourPoly, vector< queue<int> > observe_cnt);

void update_robot_list( vector<RobotFeature>& all_robot, 
						vector< vector<Point> >& all_contourPoly, 
						vector< std::queue<int> > & observe_cnt,
						vector<RobotFeature>& current_robot, 
						vector< vector<Point> >& current_contourPoly, 
						vector<std::pair<int, int> >& matches);
						

//std::vector< std::vector<cv::Point> > robotDetect(cv::Mat &frame, cv::Mat& K, cv::Mat& distCoeff);

template<class TYPE>
double adaptiveFitThreshold(std::vector<TYPE>& p)
{
	double min_x = DBL_MAX;
	double max_x = DBL_MIN;
	double min_y = DBL_MAX;
	double max_y = DBL_MIN;
	for(int i = 0; i < p.size(); i++)
	{
		min_x = std::min(min_x, (double)p[i].x);
		max_x = std::max(max_x, (double)p[i].x);
		min_y = std::min(min_y, (double)p[i].y);
		max_y = std::max(max_y, (double)p[i].y);
	}
	
	double dist = distance(cv::Point2d(min_x, min_y), cv::Point2d(max_x, max_y));
	double threshold = 0.04*dist;
	return threshold;
}

//P0 P1 as the x-axis
template<class TYPE>
vector<cv::Point2d> relativePosition(vector<TYPE>& v1)
{
	assert(v1.size() >= 2);
	vector<cv::Point2d> v2(v1.size());
	double ang = angle(v1[1], v1[0]);
	for(int i = 0; i < v1.size(); i++)
	{
		TYPE p = v1[i] - v1[0];
		v2[i].x = p.x*cos(ang) + p.y*sin(ang);
		v2[i].y = -p.x*sin(ang) + p.y*cos(ang);
	}
	return v2;
}

template<class TYPE>
double shapeMatchScore(vector<TYPE>& v1, vector<cv::Point2d> v2)
{
	assert(v1.size() == v2.size());
	vector<cv::Point2d> r1 = relativePosition(v1);
	vector<cv::Point2d> r2 = relativePosition(v2);
	
	cv::Mat m1 = cv::Mat::zeros(2*v1.size(), 1, CV_64FC1);
	cv::Mat m2 = cv::Mat::zeros(2*v1.size(), 1, CV_64FC1);
	for(int i = 0; i < v1.size(); i++)
	{
		m1.at<double>(2*i, 0) = r1[i].x;
		m1.at<double>(2*i + 1, 0) = r1[i].y;
		m2.at<double>(2*i, 0) = r2[i].x;
		m2.at<double>(2*i + 1, 0) = r2[i].y;
	}
	m1 /= cv::norm(m1);
	m2 /= cv::norm(m2);
	double score = m1.dot(m2);
	return score;
}









