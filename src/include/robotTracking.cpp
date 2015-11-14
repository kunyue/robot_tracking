#include "robotTracking.h"
#include <string>
#include <iostream>
#include <algorithm>
#include "stdlib.h"
#include "stdio.h"
#include <time.h>
#include <sys/time.h>

#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>
#include "util.h"


using namespace std;
using namespace cv;
using Eigen::Vector3d;
//using cv::ml::SVM;

#define EDGE_METHOD 2 //0 hsv canny, 1 all channel canny; 2, RGB canny; 3, R G canny, 4, grayscale canny 5, R channel
//#define EDGE_BASED 0//1 edge based; color based method 

#define HSV_MODE 1

#define EFF_AREA_MIN_THRESHOLD 80.0
#define EFF_AREA_MAX_THRESHOLD 10000.0//5000.0
#define MIN_AXIS_RATIO 1.05
#define MAX_AXIS_RATIO 5.0
#define OUTLIER_THRESHOLD 800.0

#define EFF_ROBOT_THRESHOLD 5
#define OBSERVE_HISTORY_LEN 10 //observe_history_len



#define RED_ROBOT 255
#define GREEN_ROBOT 122

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);
const Scalar WHITE = Scalar(255, 255, 255);
const int COLOR_CNT = 6;
const Scalar ColorTable[COLOR_CNT] = {RED, PINK, BLUE, LIGHTBLUE, GREEN, WHITE};

enum
{
	TRACKING = 0x01,
	DETECTING = 0x02,
}TRACKING_STATE;



//vector< vector<Point> > findBox(Mat& bwImg);
vector< RotatedRect > findBox(Mat& bwImg);

vector<Point2f> shape_center(std::vector< RotatedRect >& contours);
std::vector<Eigen::VectorXd> robot_pos_dir(vector< Point2f >& shape_centers, std::vector<Point2f>& robot_directions);
bool effitive_detect_box(RotatedRect rect);
bool effitive_track_box(RotatedRect rect);


cv::Mat K, distCoeff;
int image_width, image_height;
cv::Mat map1, map2; //for undistortion


const int table_scale = 1; 
unsigned char colorMap[256/table_scale][256/table_scale][256/table_scale];

//calibration parameter for cetacamera
double m_xi, m_k1, m_k2, m_p1, m_p2, m_gamma1, m_gamma2, m_u0, m_v0; 

bool effitive_detect_box(RotatedRect rect)
{
	Point2f vertices[4];
	rect.points(vertices);
	double len1 = distance(vertices[0], vertices[1]);
	double len2 = distance(vertices[1], vertices[2]);
	double long_axis = max(len1, len2);
	double short_axis = min(len1, len2);
	
	double area = long_axis*short_axis;
	if(area < EFF_AREA_MIN_THRESHOLD || area > EFF_AREA_MAX_THRESHOLD)
	{
		//cout << "area 1: " << area << endl;
		return false;
	}

	double ratio = long_axis/short_axis;
	if(ratio < MIN_AXIS_RATIO || ratio > MAX_AXIS_RATIO)
	{
		//cout << "ratio: " << ratio << endl;
		return false;
	}

	Point2f center = rect.center;
	double corner_threshold = 50;
	if(center.x < corner_threshold || center.x > (image_width - corner_threshold))
	{
		//cout << "robot at corner " << endl;
		return false;
	}
	return true;
}


bool effitive_track_box(RotatedRect rect)
{
	
	double _EFF_AREA_MIN_THRESHOLD =  50;//80.0
	double _EFF_AREA_MAX_THRESHOLD =  10000.0;//5000.0
	double _MIN_AXIS_RATIO  = 1.05;
	double _MAX_AXIS_RATIO  = 6.0;
	
	Point2f vertices[4];
	rect.points(vertices);
	double len1 = distance(vertices[0], vertices[1]);
	double len2 = distance(vertices[1], vertices[2]);
	double long_axis = max(len1, len2);
	double short_axis = min(len1, len2);
	
	double area = long_axis*short_axis;
	if(area < _EFF_AREA_MIN_THRESHOLD || area > _EFF_AREA_MAX_THRESHOLD)
	{
		//cout << "area 2: " << area << endl;
		return false;
	}

	double ratio = long_axis/short_axis;
	if(ratio < _MIN_AXIS_RATIO || ratio > _MAX_AXIS_RATIO)
	{
		//cout << "ratio: " << ratio << endl;
		return false;
	}

	Point2f center = rect.center;
	double corner_threshold = 50;
	if(center.x < corner_threshold || center.x > (image_width - corner_threshold))
	{
		//cout << "robot at corner " << endl;
		return false;
	}
	return true;
}


vector< RotatedRect > findBox(Mat& bwImg)
{
	vector< vector<Point> > contours, contoursPoly;
	vector<Vec4i>hierarchy;
	int64_t start = 0, end = 0; 
	findContours(bwImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector< RotatedRect > rects;


	for(int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		
		if(area > EFF_AREA_MIN_THRESHOLD && area < EFF_AREA_MAX_THRESHOLD)
		{
			
			//cout << "area: " << area << endl;
			RotatedRect rect = minAreaRect( Mat(contours[i]) );
			
			Point2f vertices[4];
			rect.points(vertices);
			double len1 = distance(vertices[0], vertices[1]);
			double len2 = distance(vertices[1], vertices[2]);
			double long_axis = max(len1, len2);
			double short_axis = min(len1, len2);
			//double ratio = long_axis/short_axis;
			double rectArea = long_axis*short_axis;
			double area_ratio = area/rectArea;

			if(area_ratio < 0.6)
			{
				continue;
			}

			if(effitive_detect_box(rect))
			{
				rects.push_back(rect);
			}

			//cout << "ratio: " << ratio << " area_ratio: " << area_ratio << endl;
			// vector<Point> contour;
			// for (int j = 0; j < 4; j++)
			// {
			// 	contour.push_back(vertices[j]);
			// }
			//contour = alignShape(contour);
			//contoursPoly.push_back(contour);
			
		}
	}

	//return contoursPoly;
	return rects;
}


//main cost is at the edge part
vector< vector<Point> > robotDetect(cv::Mat &frame/*, cv::Mat& K, cv::Mat& distCoeff*/)
{
	assert(frame.channels() == 3);
	int64_t start = 0, end = 0;
	
	int const lowThreshold = 80;
	int ratio = 2;
	
	vector< vector<Point> > contourPoly;
	if(frame.empty() ) 
	{
		return contourPoly;
	}
	
	Mat edgeImg, hsvImg, gray;
    vector<Mat> rgb, hsv;
    
    Mat bwB, bwG, bwR, bwS;
   
   //start = get_timestamp();  
   
//	split(frame, rgb);
//	blur(rgb[0], rgb[0], Size(3, 3));
//	blur(rgb[1], rgb[1], Size(3, 3));
//	blur(rgb[1], rgb[2], Size(3, 3));
//	
//	Canny(rgb[0], bwB, lowThreshold, lowThreshold*ratio,  3);
//	Canny(rgb[1], bwG, lowThreshold, lowThreshold*ratio,  3);
//	Canny(rgb[2], bwR, lowThreshold, lowThreshold*ratio,  3);


#if EDGE_METHOD == 0
	cvtColor(frame, hsvImg, CV_BGR2HSV);
	split(hsvImg, hsv);
	blur(hsv[1], hsv[1], Size(3, 3));

	Canny(hsv[1], bwS, lowThreshold, lowThreshold*ratio,  3);

	//edgeImg = bwB | bwG | bwR | bwS;
	edgeImg = bwS;
#elif	EDGE_METHOD == 1
	cvtColor(frame, hsvImg, CV_BGR2HSV);
	split(hsvImg, hsv);
	blur(hsv[1], hsv[1], Size(3, 3));
	Canny(hsv[1], bwS, lowThreshold, lowThreshold*ratio,  3);
	
	split(frame, rgb);
	blur(rgb[0], rgb[0], Size(3, 3));
	blur(rgb[1], rgb[1], Size(3, 3));
	blur(rgb[1], rgb[2], Size(3, 3));
	
	Canny(rgb[0], bwB, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[1], bwG, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[2], bwR, lowThreshold, lowThreshold*ratio,  3);
	edgeImg = bwB | bwG | bwR | bwS;
#elif	 	EDGE_METHOD == 2
	
	split(frame, rgb);
	blur(rgb[0], rgb[0], Size(3, 3));
	blur(rgb[1], rgb[1], Size(3, 3));
	blur(rgb[1], rgb[2], Size(3, 3));
	
	Canny(rgb[0], bwB, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[1], bwG, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[2], bwR, lowThreshold, lowThreshold*ratio,  3);
	edgeImg = bwB | bwG | bwR;	
	
#elif	 EDGE_METHOD == 3
	
	split(frame, rgb);
	//blur(rgb[0], rgb[0], Size(3, 3));
	blur(rgb[1], rgb[1], Size(3, 3));
	blur(rgb[1], rgb[2], Size(3, 3));
	
	//Canny(rgb[0], bwB, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[1], bwG, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[2], bwR, lowThreshold, lowThreshold*ratio,  3);
	edgeImg =  bwG | bwR;	
		
#elif	 EDGE_METHOD == 4
	cvtColor(frame, gray, CV_BGR2GRAY);
	blur(gray, gray, Size(3, 3));
	Canny(gray, edgeImg, lowThreshold, lowThreshold*ratio,  3);
	
#elif EDGE_METHOD == 5
	split(frame, rgb);
	//blur(rgb[0], rgb[0], Size(3, 3));
	blur(rgb[1], rgb[1], Size(3, 3));
	//blur(rgb[1], rgb[2], Size(3, 3));
	
	//Canny(rgb[0], bwB, lowThreshold, lowThreshold*ratio,  3);
	Canny(rgb[1], bwG, lowThreshold, lowThreshold*ratio,  3);
	//Canny(rgb[2], bwR, lowThreshold, lowThreshold*ratio,  3);
	edgeImg =  bwG;
#endif 	


	//end = get_timestamp();
	//cout << "edges: " << (end - start)/1000000.0 << endl;
	
	
	//TODO 
	//start = get_timestamp();  
	//contourPoly = findPattern(edgeImg);
	//contourPoly = findBox(edgeImg);
	//end = get_timestamp();
	
	
	//cout << "find patterns: " << (end - start)/1000000.0 << endl;
	
	return contourPoly;
}



int robotTrackInit(char* calib_filename, char* green_svm_filename, char* red_svm_filename) 
{

	//int ret1 =  svmInit(green_svm_filename, red_svm_filename);
	int ret1 = 0;
	#if HSV_MODE
	hsvTableInit(); //TODO
	#else 
	colorTableInit(); //TODO
	#endif
	
	//int ret2 = calibInit(calib_filename);
	int ret2 = readCetaCamera(calib_filename);

	if(ret1 == -1 || ret2 == -1)
	{
		return -1;
	}else 
	{
		return 0;
	}

}


int calibInit(char* calib_filename) 
{
	bool ret = readCalibrationResult(calib_filename, K, distCoeff, image_width, image_height);
    if(ret == -1) 
    {
    	printf("\ncan not read the calibration result");
    	return -1;
    }
    cout << "K: " << K << "\ndistCoeff: " << distCoeff << endl; 
    cv::initUndistortRectifyMap(
		    K,
		    distCoeff,
		    cv::Mat(),
		    K,
		    cv::Size(image_width, image_height),
		    CV_32FC1,
		    map1, map2);
    return 0;
}


//use camshift to track the robot
std::vector<Eigen::Vector3d> camshiftTrack(Mat& frame)
{
	

	static int track_state = DETECTING;

    int hsize = 16;

    float hranges[] = {0, 180};//TODO 

    const float* phranges = hranges;

    Mat hsv, hue, mask, color_mask, white_mask;

    Mat hist, backproj;

    static std::vector<Mat> hists;

    Rect trackWindow;
    static vector<Rect> trackWindows;

    vector<RotatedRect> robot_rects;
   
    std::vector<Point> contour;
	std::vector< vector<Point> > contours;

	static int frame_cnt = 0;
    static int detect_cnt = 0;



	std::vector<Eigen::Vector3d> robotPose;//output

	if(frame.empty() ) 
	{
		return robotPose;
	}
	

	static int untracked_cnt = 0, track_cnt = 0;
	//remap( frame, frame, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	

	if(track_state == TRACKING)
	{
 	 	
 	 	cvtColor(frame, hsv, COLOR_BGR2HSV);	
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);

        int success_track_cnt = 0;
        robot_rects.clear();

     	//cout << "tracking: " << hists.size() << endl;
        for (int i = 0; i < 1/*hists.size()*/; ++i)
        {
        	hists[i].copyTo(hist);

        	trackWindow = trackWindows[i]; 
    		
    		whiteMask(frame, white_mask);
    		//imshow("white_mask", white_mask);
		 	calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
	        backproj &= white_mask; //TODO 


	        RotatedRect trackBox = CamShift(backproj, trackWindow,
	                            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
	        //cout << "tracking: " << trackWindow << endl;

	       

	       // if( trackWindow.area() >= EFF_AREA_MIN_THRESHOLD 
	        //	&& trackWindow.area() <= EFF_AREA_MAX_THRESHOLD)
	        if(effitive_track_box(trackBox))
	        {
	        	trackWindows[i] = trackWindow;
	        	success_track_cnt++;
        		robot_rects.push_back(trackBox);
	        }

	        //TODO 
	        // if(success_track_cnt >= 1)//only track one robot
	        // {
	        // 	break;
	        // }
	        
        }

        //cout << "tracked: " << success_track_cnt << endl;
        if(success_track_cnt <= 0)
        {	
        	untracked_cnt++;
        	cout << "robot lost:" << endl;
        	track_state = DETECTING;
        	if(untracked_cnt >= 1)
        	{
        		//track_state = DETECTING;
        	}
        	
        }else
        {
        	track_cnt++;
        	//cout << "tracked " << success_track_cnt << "robots" << endl;
        }
       
	}

	if(track_state == DETECTING)
	{
		#if HSV_MODE
		cvtColor(frame, hsv, CV_BGR2HSV);
		robotSegment(hsv, color_mask);//mask for red and green region 
		#else 
		robotSegment(frame, color_mask);//mask for red and green region 
		#endif

		

		//imshow("color_mask", color_mask);

		robot_rects = findBox(color_mask);//find contours will destory the image 

		if(robot_rects.size() >= 1 && robot_rects.size() <= 10)
		{
			
			cvtColor(frame, hsv, COLOR_BGR2HSV);

			//initialize for tracking
			hists.clear();
			trackWindows.clear();

			int eff_detect = 0;
			//cout << "detect: " << robot_rects.size() << endl;

			for (int i = 0; i < robot_rects.size(); ++i)
			{
				
				RotatedRect scaledRoatatedRect = resize_rotatedrect( robot_rects[i], 0.3);

				eff_detect++;	
				rect_to_contour(scaledRoatatedRect, contour);
				Rect selection = boundingRect(contour);

				//cout << "selection : " << selection << endl;
				
				cvtColor(frame, hsv, COLOR_BGR2HSV);

				if(selection.x < 0) selection.x = 0;  
				if(selection.y < 0) selection.y = 0;
				if(selection.x + selection.width > frame.cols) selection.width = frame.cols - selection.x - 1;
				if(selection.y + selection.height > frame.rows) selection.height = frame.rows - selection.y - 1;   

				            	
                int _vmin = 30; 
                int _vmax = 255;
                int smin = 50;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);

                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                //cout << "selection: " << selection << endl;
                
                Mat roi(hue, selection), maskroi(mask, selection);
                
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                
                normalize(hist, hist, 0, 255, NORM_MINMAX);

                trackWindow = selection;

                hists.push_back(hist.clone());
                trackWindows.push_back(trackWindow);

			}



			if(eff_detect >= 1 && eff_detect < 10)
			{
				track_state = TRACKING; //TODO 
				track_cnt = 0;
				untracked_cnt = 0;
			}
		

		}
	}

	for (unsigned int i = 0; i < robot_rects.size(); i++)
	{
		Scalar color = ColorTable[i%COLOR_CNT];
		drawrect(frame, robot_rects[i], color);
	}


	if(track_cnt >= 1)
	{
		std::vector<Point2f> shape_centers = shape_center(robot_rects);
		//robotPose = normalized_robot_pose(shape_centers);
		robotPose = cetaUndistPoint(shape_centers);
	}

	return robotPose;
}

//
std::vector<Eigen::VectorXd> robot_pos_dir(vector< Point2f >& shape_centers, std::vector<Point2f>& robot_directions)
{
	std::vector<Eigen::VectorXd> normlizedCenter(shape_centers.size());
	cv::Mat src = Mat::zeros(1, 2, CV_32FC2);
	cv::Mat m = Mat::zeros(3, 1, CV_64FC1);

	for(int i = 0; i < shape_centers.size(); i++)
	{
		 Eigen::VectorXd cc(6);
	
		 m.at<double>(0) = shape_centers[i].x;
		 m.at<double>(1) = shape_centers[i].y;
		 m.at<double>(2) = 1.0;
		 m = K.inv()*m;
		 cc(0) = m.at<double>(0);
		 cc(1) = m.at<double>(1);
		 cc(2) = m.at<double>(2);
		 
		 //direction:
		 m.at<double>(0) = robot_directions[i].x;
		 m.at<double>(1) = robot_directions[i].y;
		 m.at<double>(2) = 1.0;
		 m = K.inv()*m;
		 cc(3) = m.at<double>(0);
		 cc(4) = m.at<double>(1);
		 cc(5) = m.at<double>(2);
		 
	 	normlizedCenter[i] = cc;
	}
	return normlizedCenter;
}




//normlized position
std::vector<Eigen::Vector3d> normalized_robot_pose(std::vector<Point2f> shape_centers)
{
	
	float radius = 0.0;
	std::vector<Eigen::Vector3d> normlizedCenter;

	cv::Mat src = Mat::zeros(1, 2, CV_32FC2);
	cv::Mat m = Mat::zeros(3, 1, CV_64FC1);
	std::vector<Point2f> p2;

	for(int i = 0; i < shape_centers.size(); i++)
	{
		 Eigen::Vector3d cc;
		 m.at<double>(0) = shape_centers[i].x;
		 m.at<double>(1) = shape_centers[i].y;
		 m.at<double>(2) = 1.0;
		 m = K.inv()*m;
		 cc(0) = m.at<double>(0);
		 cc(1) = m.at<double>(1);
		 cc(2) = m.at<double>(2);

	 	 normlizedCenter.push_back(cc);
	}
	return normlizedCenter;
}



int svmInit(char*  filename1, char* filename2)
{
	cout << "initializing color table" << endl;	
	Mat m = Mat::zeros(6, 1, CV_32FC1);
	Mat sv_green;// = svm_green->getSupportVectors();
	Mat sv_red;// = svm_red->getSupportVectors();
	double rho_green;// = svm_green->getDecisionFunction(0, alpha_green, svidx_green);
	double rho_red;// = svm_red->getDecisionFunction(0, alpha_red, svidx_red);
	cv::FileStorage fs, fs2;
	fs.open(filename1, cv::FileStorage::READ);
	
	fs2.open(filename2, cv::FileStorage::READ);
	
	if (!fs.isOpened()) {return -1;}
	if (!fs2.isOpened()) {return -1;}
	
	fs["support_vectors"] >> sv_green;
	fs["rho"] >> rho_green;
	
	fs2["support_vectors"] >> sv_red;
	fs2["rho"] >> rho_red;


	cout << "sv_green " << sv_green <<  " sv_red" << sv_red << " rho_green: " << rho_green << "rho_red" << endl;
	 
	double tmp11, tmp12, tmp21, tmp22, tmp31, tmp32;
	for (unsigned int i = 0; i < 256; i += table_scale)
	{
		//m.at<float>(0) = i;
		//m.at<float>(3) = (float)i*i;
		
		tmp11 = sv_green.at<double>(0)*i + sv_green.at<double>(3)*i*i - rho_green;
		tmp12 = sv_red.at<double>(0)*i   + sv_red.at<double>(3)*i*i - rho_red;
		for (unsigned int j = 0; j < 256; j += table_scale)
		{
			//m.at<float>(1) = j;
			//m.at<float>(4) = (float)j*j;
			
			tmp21 = tmp11 + sv_green.at<double>(1)*j + sv_green.at<double>(4)*j*j;
			tmp22 = tmp12 + sv_red.at<double>(1)*j + sv_red.at<double>(4)*j*j;
			
			for (unsigned int k = 0; k < 256; k += table_scale)
			{
				tmp31 = tmp21 + sv_green.at<double>(2)*k + sv_green.at<double>(5)*k*k;
				tmp32 = tmp22 + sv_red.at<double>(2)*k + sv_red.at<double>(5)*k*k;
				
				//TODO
				// if(tmp31 < 0.0 || tmp32 < 0.0)
				// {
				// 	colorMap[i/table_scale][j/table_scale][k/table_scale] = 255;
				// }else 
				// {
				// 	colorMap[i/table_scale][j/table_scale][k/table_scale] = 0;
				// }
				if(tmp31 < 0.0 )
				{
					colorMap[i/table_scale][j/table_scale][k/table_scale] = RED_ROBOT;
				}else if(tmp32 < 0.0)
				{
					colorMap[i/table_scale][j/table_scale][k/table_scale] = GREEN_ROBOT;
				}
				else 
				{
					colorMap[i/table_scale][j/table_scale][k/table_scale] = 0;
				}
				
			}
		}
	} 
	cout << "initialization OK" << endl;
}


void colorTableInit()
{
	bool is_red = false, is_green = false;
	for (unsigned int b = 0; b < 256; b += table_scale)
	{		

		for (unsigned int g = 0; g < 256; g += table_scale)
		{
					
			for (unsigned int r = 0; r < 256; r += table_scale)
			{
				if(b < 100 && g < 100 && r > 70) 
				{
					is_red = true;
				}else
				{
					is_red = false;
				}

				if( (b > 5 && b < 40) && (g > 50 && g < 80) && (r > 20 && r < 45) ) 
				{
					is_green = true;
				}else 
				{
					is_green = false;
				}

				if(is_red || is_green)
				{
					colorMap[b/table_scale][g/table_scale][r/table_scale] = 255;
				}else 
				{
					colorMap[b/table_scale][g/table_scale][r/table_scale] = 0;
				}

			}
		}
	} 
}

void hsvTableInit()
{
	bool is_red = false, is_green = false;

	for (unsigned int h = 0; h < 256; h += table_scale)
	{		

		for (unsigned int s = 0; s < 256; s += table_scale)
		{
					
			for (unsigned int v = 0; v < 256; v += table_scale)
			{
				if((h < 60 || h > 130) && s > 55 && v > 55) //old
				// if((h < 45 || h > 210) && s > 140 && v > 110) //red
				{
					is_red = true;
				}else
				{
					is_red = false;
				}

				// if((h > 34 && h < 70) && (s > 150 && s < 200) && (v > 50 && v < 80))
				// {
				// 	is_green = true;
				// }else 
				// {
				// 	is_green = false;
				// }
				if(is_red || is_green)
				{
					colorMap[h/table_scale][s/table_scale][v/table_scale] = 255;
				}else 
				{
					colorMap[h/table_scale][s/table_scale][v/table_scale] = 0;
				}

			}
		}
	} 
}





void robotSegment(Mat& frame, Mat& mask)
{
	unsigned char* p = NULL;
	
	if(mask.rows != frame.rows || mask.cols != frame.cols)
	{
		mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	}
	
	int id1, id2, id3;
	for (unsigned int i = 0; i < frame.rows; i++)
	{
		
		p = (unsigned char*)(frame.data + i*frame.step);

		for (unsigned int j = 0; j < frame.cols; j++)
		{
			//Vec3b p = frame.at<Vec3b>(i, j);
			id1 = p[0]/table_scale;
			id2 = p[1]/table_scale;
			id3 = p[2]/table_scale;
			p += 3;
			//cout << "id: " << id1 << " " << id2 << "  " << id3 << endl;
			mask.at<unsigned char>(i, j) = colorMap[id1][id2][id3];
		}
	}
}



// vector<Point2f> shape_center(std::vector< std::vector<Point> >& contours)
// {
	
// 	Point2f center;
// 	float radius = 0.0;
// 	std::vector<Point2f> centers;   
// 	for(int i = 0; i < contours.size(); i++)
// 	{
// 		minEnclosingCircle( contours[i], center, radius);
// 		centers.push_back(center);
// 	}
// 	return centers;
// } 

vector<Point2f> shape_center(std::vector< RotatedRect >& rects)
{
	std::vector<Point2f> centers;   
	for(int i = 0; i < rects.size(); i++)
	{
		centers.push_back(rects[i].center);
	}
	return centers;
} 

   
//CataCamera
int readCetaCamera(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
 	
    if (!fs.isOpened())
    {
        return -1;
    }
    
    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType.compare("MEI") != 0)
        {
            return -1;
        }
    }
    
    string m_cameraName;
    fs["camera_name"] >> m_cameraName;
    image_width = static_cast<int>(fs["image_width"]);
    image_height = static_cast<int>(fs["image_height"]);
    
    cv::FileNode n = fs["mirror_parameters"];
    m_xi = static_cast<double>(n["xi"]);

    n = fs["distortion_parameters"];
    m_k1 = static_cast<double>(n["k1"]);
    m_k2 = static_cast<double>(n["k2"]);
    m_p1 = static_cast<double>(n["p1"]);
    m_p2 = static_cast<double>(n["p2"]);
    
    n = fs["projection_parameters"];
    m_gamma1 = static_cast<double>(n["gamma1"]);
    m_gamma2 = static_cast<double>(n["gamma2"]);
    m_u0 = static_cast<double>(n["u0"]);
    m_v0 = static_cast<double>(n["v0"]);

    //cout << "m_xi: " << m_xi << "m_k1: " << m_k1 << " m_k2:" << m_k2 << "  m_gamma1: " << m_gamma1
    //	<< " m_gamma2: " << m_gamma2 << endl;

    return 0;
}


 vector<Eigen::Vector3d> cetaUndistPoint(std::vector<Point2f>& points)
 {
 	vector<Eigen::Vector3d> normlized_points;
	double m_inv_K11 = 1.0 / m_gamma1;
    double m_inv_K13 = -m_u0 / m_gamma1;
    double m_inv_K22 = 1.0 / m_gamma2;
    double m_inv_K23 = -m_v0 / m_gamma2;

    const double fScale = 1.0;//TODO 

 	for (int i = 0; i < points.size(); ++i)
 	{
		//cout << "point: " << points[i] << endl;
		double u = points[i].x;
		double v = points[i].y;

		double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
		double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

		double xi = m_xi;
		double d2 = mx_u * mx_u + my_u * my_u;

		Eigen::Vector3d P;
		P << mx_u, my_u, 1.0 - xi * (d2 + 1.0) / (xi + sqrt(1.0 + (1.0 - xi * xi) * d2));

		P /= P(2);		
		normlized_points.push_back(P);

 	}

 	return normlized_points;

 }

void whiteMask(Mat& frame, Mat& mask)
{
	if(mask.cols != frame.cols || mask.rows != frame.rows)
	{
		mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	}
	unsigned char* bgr = NULL;

	for (int i = 0; i < frame.rows; ++i)
	{
		bgr = (unsigned char*)(frame.data + i*frame.step);

		for(int j = 0; j < frame.cols; j++)
		{
			//Vec3b bgr = frame.at<Vec3b>(i, j);
			
			if(bgr[0]  > 135 && bgr[1] > 75 && bgr[2] > 75)//old
			// if(bgr[0]  > 65 && bgr[1] > 150 && bgr[2] > 70)//old
			{
				mask.at<unsigned char>(i, j) = 0;
			}else 
			{
				mask.at<unsigned char>(i, j) = 255;
			}
			bgr += 3;
		}
	}
}
