#include <string>
#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "Eigen/Dense"

using std::vector;
using std::queue;
using cv::Point;
using cv::Mat;
using std::cout;
using std::endl;


void robotSegment(Mat& frame, Mat& mask);
int svmInit(char*  filename1, char* filename2);
int calibInit(char* calib_filename);
int robotTrackInit(char* calib_filename, char*  green_svm_filename, char* red_svm_filename) ;

std::vector<Eigen::VectorXd> robotTrack(cv::Mat& frame);
std::vector<Eigen::Vector3d> camshiftTrack(Mat& frame);


std::vector< std::vector<cv::Point> > robotDetect(cv::Mat &frame);
						
std::vector<Eigen::Vector3d> normalized_robot_pose(std::vector<Point2f> shape_centers);


int readCetaCamera(const std::string& filename);

 vector<Eigen::Vector3d> cetaUndistPoint(std::vector<Point2f>& points);
 
 void colorTableInit();







