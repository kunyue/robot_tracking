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
using std::cout;
using std::endl;
//using cv::ml::SVM;
//using namespace cv::ml;


void robotSegment(Mat& frame, Mat& mask);
int svmInit(char*  filename1, char* filename2);
int calibInit(char* calib_filename);
int robotTrackInit(char* calib_filename, char*  green_svm_filename, char* red_svm_filename) ;

vector< vector<Point> > findPattern(Mat& bwImg);;

std::vector<Eigen::VectorXd> robotTrack(cv::Mat& frame);
std::vector<Eigen::VectorXd> camshiftTrack(Mat& frame);

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
						
std::vector<Eigen::VectorXd> normalized_robot_pose(
	std::vector<Point2f> shape_centers, 
	std::vector<Point2f> dir_centers);


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



//estimate direction of the robot
//use the mass center 
template<class T>
vector<Point2f> mass_center(std::vector< std::vector<T> > contours, Mat& color_mask)
{
	cv::Mat mask = cv::Mat::zeros(color_mask.rows, color_mask.cols, CV_8UC1);
	vector<Point> contour;
	std::vector<Point2f> robotCenter;
	
	for(int i = 0; i < contours.size(); i++)
	{
		mask.setTo(Scalar(0));
		cout << __FILE__ << " " << __LINE__ << endl;
		drawContours(mask, contours, i, Scalar(1), CV_FILLED);	
		cout << __FILE__ << " " << __LINE__ << endl;
		mask &= color_mask;

		float sum_x = 0.0f;
		float sum_y = 0.0f;

		int count = 0;
		unsigned int r = 0, g = 0, b = 0;
		for (unsigned int j = 0; j < mask.rows; j++)
		{
			for (unsigned int k = 0; k < mask.cols; k++)
			{
				if(mask.at<unsigned char>(j, k))
				{
					sum_x += k;
					sum_y += j;
					count++;
				}				
			}
		}	
		Point2f center = Point2f(sum_x/count, sum_y/count);		
		robotCenter.push_back(center);
	}
	return robotCenter;
}

template<class T>
vector<Point2f> shape_center(std::vector< std::vector<T> >& contours)
{
	Point2f center;
	float radius = 0.0;
	std::vector<Point2f> centers;
	for(int i = 0; i < contours.size(); i++)
	{
		 minEnclosingCircle( contours[i], center, radius);
		 centers.push_back(center);
	}
	return centers;
}

//use the shape center and the mass center to estimate robot direction
template<class T>
std::vector<Point2f> robot_direction(std::vector< std::vector<T> > contours, vector<T>&mass_centers, vector<T>& shape_centers)
{
	std::vector<Point2f> robot_directions;
	Point2f p0, p1, p2;
	for (int i = 0; i < contours.size(); ++i)
	{
		p1.x = (contours[i][0].x + contours[i][1].x)/2.0f - shape_centers[i].x;
		p1.y = (contours[i][0].y + contours[i][1].y)/2.0f - shape_centers[i].y;
		p2.x = (contours[i][2].x + contours[i][3].x)/2.0f - shape_centers[i].x;
		p2.y = (contours[i][2].y + contours[i][3].y)/2.0f - shape_centers[i].y;
		p0.x = mass_centers[i].x - shape_centers[i].x;
		p0.y = mass_centers[i].y - shape_centers[i].y;


		//Point2f p1 = contours[i][0]/2.0 + contours[i][1]/2.0;//(contours[i][0] + contours[i][1])/2.0 - shape_centers[i];
		// Point2f p2 = (contours[i][2] + contours[i][3])/2 - shape_centers[i];
		//Point2f p0 = mass_centers[i] - shape_centers[i];

		if(p0.x == 0.0 && p0.y == 0.0)
		{
			robot_directions.push_back( Point2f( (contours[i][0].x + contours[i][1].x)/2.0f, (contours[i][0].y + contours[i][1].y)/2.0f) );
		}else if(cross_product(p0, p1) > 0.0 && cross_product(p0, p2) < 0.0)
		{
			robot_directions.push_back( Point2f ( (contours[i][0].x + contours[i][1].x)/2.0f, (contours[i][0].y + contours[i][1].y)/2.0f) );
		}else if(cross_product(p0, p1) < 0.0 && cross_product(p0, p2) > 0.0)
		{
			robot_directions.push_back( Point2f( (contours[i][2].x + contours[i][3].x)/2.0f, (contours[i][2].y + contours[i][3].y)/2.0f) );
		}else 
		{
			robot_directions.push_back( Point2f( (contours[i][0].x + contours[i][1].x)/2.0f, (contours[i][0].y + contours[i][1].y)/2.0f) );
		}
	}
	return robot_directions;
}







