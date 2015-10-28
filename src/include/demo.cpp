#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <queue>

#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
#include "robotTracking.h"
#include "robotFeature.h"
#include "util.h"



using namespace std;
using namespace cv;

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);
const Scalar WHITE = Scalar(255, 255, 255);
const int COLOR_CNT = 6;
const Scalar ColorTable[COLOR_CNT] = {RED, PINK, BLUE, LIGHTBLUE, GREEN, WHITE};

int demo_edge_based(int argc, char** argv);

int main(int argc, char** argv) 
{
	
	//demo_edge_based(argc, argv);
	//return 0;

	//1. init
	int ret = robotTrackInit("../../../config/camera_binning.yml", "../../../config/color_red_bluefox.yml", "../../../config/color_green_bluefox.yml");
	std::vector<Eigen::VectorXd> robotPosition;//normalized position
	Mat frame;
    //VideoCapture cap("/home/libing/irobot_2015-10-27-23-05-33.avi"); 
    VideoCapture cap("/home/libing/irobot_2015-10-27-23-05-33.avi"); 
   
    if ( !cap.isOpened()  )  // if not success, exit program
    {
         cout << "Cannot open the camera or the video file " << endl;
         return -1;
    }
    int cnt = 0;
    while(1)
    {
		cap >> frame;
		if(frame.empty()) break;
		cnt++;
		if(cnt < 200) continue;
		//2. track
		robotPosition = robotTrack(frame);
		//cout << robotPosition.size() << " robotPos: " << "\n";
		for (unsigned int i = 0; i < robotPosition.size(); i++)
		{
			//cout << robotPosition[i].transpose() <<  endl;
		}
		
		imshow("frame", frame);
		char key = waitKey(30);
		if(key == 27)
		{
			break;
		}else if(key == ' ')
		{
			waitKey(0);
		}
		
	}
	
}


int demo_edge_based(int argc, char** argv)
{
    Mat frame, dst;
    char filename[1024];
    char calib_filename[1024];
    if(argc == 3)
    {
    	sprintf(filename, "%s", argv[1]);
    	sprintf(calib_filename, "%s", argv[2]);
    }else 
    {
    	sprintf(filename, "../data/1.avi");
    	sprintf(calib_filename, "../data/camera.yml");
    }
    
    Mat K, distCoeff;
    int image_width, image_height;
    int frame_cnt = 0;
    int detect_cnt = 0;
    int64_t start = 0, end = 0;
    double diff = 0.0;
    bool ret = readCalibrationResult(calib_filename, K, distCoeff, image_width, image_height);
    if(ret == -1) 
    {
    	printf("\ncan not read the calibration result");
    	//return 0;
    }
    
    VideoCapture cap(filename); 
    if ( !cap.isOpened()  )  // if not success, exit program
    {
         cout << "Cannot open the camera or the video file " << endl;
         return -1;
    }
    
    vector< vector<Point> > contourPoly, contourPoly_prev;
    vector<RobotFeature>  robot_features, robot_features_prev;
    vector<RobotFeature>  all_robot;//all the robots
    vector< vector<Point> > all_contourPoly;
    vector< queue<int> > observe_cnt;
    
    vector<std::pair<int, int> > matches;
	char label[3];
	Mat map1, map2;
	cv::initUndistortRectifyMap(
		    K,
		    distCoeff,
		    cv::Mat(),
		    K,
		    cv::Size(image_width, image_height),
		    CV_32FC1,
		    map1, map2);
	
                             
    while(1)
    {
		start = get_timestamp();  
		
		cap >> frame;
		if(frame.empty() ) break;
		
		//remap( frame, frame, map1, map2, INTER_LINEAR, BORDER_CONSTANT );
		
		contourPoly = robotDetect(frame);	
		calc_features(frame, contourPoly, robot_features);
		robotMatch(all_robot, robot_features, matches);
	
		
		frame.copyTo(dst);
	 
		//show the current and past robot position  
		#if 0	 
		for (unsigned int i = 0; i < matches.size(); i++)
		{
			int id1 = matches[i].first;
			int id2 = matches[i].second;
			//cout << "id1: " << id1 << "  id2: " << id2 << endl;
			Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
			
			if(id1 >= 0)
			{
				
				//drawContours(dst, contourPoly_prev, id1, color, 2, 8);
				//circle(dst, contourPoly_prev[id1][0], 5, color, 2, 8);
				
				drawContours(dst, all_contourPoly, id1, color, 2, 8);
				//circle(dst, all_contourPoly[id1][0], 5, color, 2, 8);
			}
			if(id2 >= 0)
			{
				
				drawContours(dst, contourPoly, id2, color, 2, 8);
				//circle(dst, contourPoly[id2][0], 5, color, 2, 8);
			}
			if(id1 >= 0 && id2 >= 0)
			{
				
				//line(dst, contourPoly_prev[id1][0], contourPoly[id2][0], color, 1, 8);
				line(dst, all_contourPoly[id1][0], contourPoly[id2][0], color, 1, 8);
				circle(dst, all_contourPoly[id1][0], 5, color, 2, 8);
				circle(dst, contourPoly[id2][0], 5, color, 2, 8);
			}
		}
		
		#else 
			for (unsigned int i = 0; i < contourPoly.size(); i++)
			{
				Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
				drawContours(dst, contourPoly, i, color, 2, 8);
			}
		#endif
		
		update_robot_list(all_robot, all_contourPoly, observe_cnt, robot_features, contourPoly, matches);
		imshow("detectRobot", dst);
		if(waitKey(1) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl; 
			break; 
		}
		
		
		//robot_features_prev = robot_features;
		//contourPoly_prev = contourPoly;
		frame_cnt++;
		detect_cnt += contourPoly.size();
		int64_t current = get_timestamp();   
		double cost_time = (current - start)/1000000.0;
		cout << "cost time: " << cost_time << endl;  
    }
    double detect_rate = (double)detect_cnt/(frame_cnt*2);
    printf("\ndetect rate: %f", detect_rate);
    
    return 0;
}


int demo_color_based()
{
	
	//1. init
	int ret = svmInit("../data/color_red_bluefox.yml", "../data/color_green_bluefox.yml");
	//std::vector<Eigen::Vector3d> robotPosition;//normalized position
	Mat frame;
	Mat mask, dst;
	
	Mat K, distCoeff;
    int image_width, image_height;
    int frame_cnt = 0;
    int detect_cnt = 0;
    double diff = 0.0;
	
    vector< vector<Point> > contourPoly, contourPoly_prev;
    vector<RobotFeature>  robot_features, robot_features_prev;
    vector<RobotFeature>  all_robot;//all the robots
    vector< vector<Point> > all_contourPoly;
    vector< queue<int> > observe_cnt;
    
    vector<std::pair<int, int> > matches;
	char label[3];
	
	
    VideoCapture cap("../data/1.avi"); 
    int64_t start, end;
    if ( !cap.isOpened()  )  // if not success, exit program
    {
         cout << "Cannot open the camera or the video file " << endl;
         return -1;
    }
   	
    while(1)
    {
		cap >> frame;
		if(frame.empty()) break;
		frame.copyTo(dst);
		//2. track
		//robotPosition = robotTrack(frame);
		
		start = get_timestamp();
		
		robotSegment(frame, mask);
		contourPoly = findPattern(mask);
		calc_features(frame, contourPoly, robot_features);
		robotMatch(all_robot, robot_features, matches);
		
		end = get_timestamp();
		cout << "tracking cost time: " << (end - start)/1000000.0 << endl;
		
				//show the current and past robot position  
		#if 1	 
		for (unsigned int i = 0; i < matches.size(); i++)
		{
			int id1 = matches[i].first;
			int id2 = matches[i].second;
			//cout << "id1: " << id1 << "  id2: " << id2 << endl;
			Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
			
			if(id1 >= 0)
			{
				
				//drawContours(dst, contourPoly_prev, id1, color, 2, 8);
				//circle(dst, contourPoly_prev[id1][0], 5, color, 2, 8);
				
				drawContours(dst, all_contourPoly, id1, color, 2, 8);
				//circle(dst, all_contourPoly[id1][0], 5, color, 2, 8);
			}
			if(id2 >= 0)
			{
				
				drawContours(dst, contourPoly, id2, color, 2, 8);
				//circle(dst, contourPoly[id2][0], 5, color, 2, 8);
			}
			if(id1 >= 0 && id2 >= 0)
			{
				
				//line(dst, contourPoly_prev[id1][0], contourPoly[id2][0], color, 1, 8);
				line(dst, all_contourPoly[id1][0], contourPoly[id2][0], color, 1, 8);
				circle(dst, all_contourPoly[id1][0], 5, color, 2, 8);
				circle(dst, contourPoly[id2][0], 5, color, 2, 8);
			}
		}
		#else 
			for (unsigned int i = 0; i < contourPoly.size(); i++)
			{
				Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
				drawContours(dst, contourPoly, i, color, 2, 8);
			}
		#endif
		
		update_robot_list(all_robot, all_contourPoly, observe_cnt, robot_features, contourPoly, matches);
		
		frame_cnt++;
		detect_cnt += contourPoly.size();
		int64_t current = get_timestamp();   
		//double cost_time = (current - start)/1000000.0;
		//cout << "cost time: " << cost_time << endl;  
		
		imshow("detectRobot", dst);
		imshow("mask", mask);
		char key = waitKey(30);
		if(key == 27)
		{
			break;
		}
	}
	
    double detect_rate = (double)detect_cnt/(frame_cnt*2);
    printf("\ndetect rate: %f", detect_rate);
    
}

