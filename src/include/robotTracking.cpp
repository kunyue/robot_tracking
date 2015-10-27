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
#include "lap.h"

using namespace std;
using namespace cv;
//using cv::ml::SVM;

#define EDGE_METHOD 2 //0 hsv canny, 1 all channel canny; 2, RGB canny; 3, R G canny, 4, grayscale canny 5, R channel
#define EDGE_BASED 0//1 edge based; color based method 

#define EFF_AREA_MIN_THRESHOLD 100.0
#define EFF_AREA_MAX_THRESHOLD 30000.0
#define OUTLIER_THRESHOLD 1000.0

#define EFF_ROBOT_THRESHOLD 3
#define OBSERVE_HISTORY_LEN 10 //observe_history_len


const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);
const Scalar WHITE = Scalar(255, 255, 255);
const int COLOR_CNT = 6;
const Scalar ColorTable[COLOR_CNT] = {RED, PINK, BLUE, LIGHTBLUE, GREEN, WHITE};


vector<Point> alignShape(vector<Point>& contour);
vector< vector<Point> > findPattern(Mat& bwImg);
vector<cv::Point2d> getTemplate();
std::vector<Eigen::VectorXd> robotCenter(vector< vector<Point> > contours);

cv::Mat K, distCoeff;
int image_width, image_height;
cv::Mat map1, map2; //for undistortion


//color-based 
//Ptr<SVM> svm_red;
//Ptr<SVM> svm_green;

const int table_scale = 2; 
unsigned char colorMap[256/table_scale][256/table_scale][256/table_scale];
vector<cv::Point2d> getTemplate()
{
	vector<Point2d> robotPattern(8);
	robotPattern[0] = Point2d(-12.7, 0);
	robotPattern[1] = Point2d(12.7, 0);
	robotPattern[2] = Point2d(12.7, 11.43);
	robotPattern[3] = Point2d(5.08, 11.43);
	robotPattern[4] = Point2d(5.08, 15.24);
	robotPattern[5] = Point2d(-5.08, 15.24);
	robotPattern[6] = Point2d(-5.08, 11.43);
	robotPattern[7] = Point2d(-12.7, 11.43);	
	return robotPattern;
}


vector< vector<Point> > findPattern(Mat& bwImg)
{
	vector< vector<Point> > contours, contoursPoly;
	vector<Point> contour;
	vector<Vec4i>hierarchy;
	int64_t start = 0, end = 0;
	
	//start = get_timestamp();  
	findContours(bwImg, contours, hierarchy, /*CV_RETR_LIST*/ CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//end = get_timestamp();
	//cout << "find contours: " << (end - start)/1000000.0 << endl;
	
	start = get_timestamp();	
	for(int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		
		if(area > EFF_AREA_MIN_THRESHOLD && area < EFF_AREA_MAX_THRESHOLD)
		{
			approxPolyDP(Mat(contours[i]), contour, adaptiveFitThreshold(contours[i]), true);
			//printf("threshold: %f", adaptiveFitThreshold(contours[i]));
			//cout << "ploy count: " << contour.size() << endl;	
			if(contour.size() == 8)
			{
				contour = alignShape(contour);
			
				double score = shapeMatchScore(contour, getTemplate());
				//cout << "matchScore: " << score << endl;
				//cout << "area " << area << endl;
				if(score > 0.95)
				{
					contoursPoly.push_back(contour);
				}
				
			}
			
		}
	}
	//end = get_timestamp();
	//cout << "select contours: " << (end - start)/1000000.0 << endl;

	return contoursPoly;
}

vector<Point> alignShape(vector<Point>& contour)
{
	assert(contour.size() == 8);
	vector<Point> alignedContour(contour.size());
	vector<float> lengths(contour.size());
	for(int i = 0; i < contour.size(); i++)
	{
		lengths[i] = norm(contour[(i + 1)%contour.size()] - contour[i]);
	} 
	float maxVal = lengths[0];
	int maxID = 0;
	for(int i = 1; i < lengths.size(); i++)
	{
		if(lengths[i] > maxVal)
		{
			maxVal = lengths[i];
			maxID = i;
		}
	}
	//clockwire and anti-clockwise
	Point p0 = contour[maxID];
	Point p1 = contour[(maxID + 1)%contour.size()];
	Point p2 = contour[(maxID + 2)%contour.size()];
	
	bool counterClockwise = true;
	int ang1 = atan2(p1.y - p0.y, p1.x - p0.x)/M_PI*180;
	int ang2 = atan2(p2.y - p1.y, p2.x - p1.x)/M_PI*180;
	if((ang2 - ang1 + 360) % 360 < 180)
	{
		counterClockwise = true;
	}else
	{
		counterClockwise = false;
	}
	
	if(counterClockwise)
	{
		for(int i = 0; i < contour.size(); i++)
		{
			alignedContour[i] = contour[(i + maxID)%contour.size()];
		}
		return alignedContour;
	}else 
	{
		for(int i = 0; i < contour.size(); i++)
		{
			alignedContour[i] = contour[(maxID + 1 - i)%contour.size()];
		}
		return alignedContour;
	}
	
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
	
	
	
	//start = get_timestamp();  
	contourPoly = findPattern(edgeImg);
	//end = get_timestamp();
	
	
	//cout << "find patterns: " << (end - start)/1000000.0 << endl;
	
	return contourPoly;
}

//compute features in the polygon
void calc_features(cv::Mat img, vector< vector<Point> >& contours, vector<RobotFeature>& robot_features)
{
	assert(img.channels() == 3);
	robot_features.clear();
	cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	vector<Point> contour;
	vector<RobotFeature> features; 
	
	for(int i = 0; i < contours.size(); i++)
	{
		mask.setTo(Scalar(0));
		RobotFeature features;
		contour = contours[i];
		features.x = contour[0].x;
		features.y = contour[0].y;
		features.theta = angle(contour[1], contour[0]);
		
//		fillPoly(mask, &contours[i], &npts, 1, Scalar(1)); 
		drawContours(mask, contours, i, Scalar(1), CV_FILLED);	
		int count = 0;
		unsigned int r = 0, g = 0, b = 0;
		for (unsigned int j = 0; j < img.rows; j++)
		{
			for (unsigned int k = 0; k < img.cols; k++)
			{
				if(mask.at<unsigned char>(j, k))
				{
					count++;
					b += img.at<cv::Vec3b>(j, k)[0];
					g += img.at<cv::Vec3b>(j, k)[1];
					r += img.at<cv::Vec3b>(j, k)[2];
				}				
			}
		}	
		features.area =  contourArea(contours[i]);;
		features.b = (unsigned char)(b/count);
		features.g = (unsigned char)(g/count);
		features.r = (unsigned char)(r/count);	
		
		robot_features.push_back(features);
	}
}



//track the robot using k-nearest path 
void robotMatch(vector<RobotFeature>& r1, vector<RobotFeature>& r2, vector<std::pair<int, int> >& matches)
{
	int size1 = r1.size();
	int size2 = r2.size();
	int size = std::max(size1, size2);
	matches.clear();
	
	double** dist_tabel = (double **) malloc(sizeof(double *) * size);
	for (int i=0; i< size; i++)
	{
		dist_tabel[i] = (double *) malloc(sizeof(double) * size);
	}
	
	for (int i=0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (i >= size1 || j >= size2)//> add some dummy nodes
			{
				dist_tabel[i][j] = OUTLIER_THRESHOLD;
				continue;
			}
			dist_tabel[i][j] = feature_dist(r1[i], r2[j]);
		}
	}
	//Linear assignment problem
	double *u, *v;
	int *colsol, *rowsol;

	rowsol = (int *) malloc(sizeof(int) * size);
	colsol = (int *) malloc(sizeof(int) * size);
	u = (double *) malloc(sizeof(double) * size);
	v = (double *) malloc(sizeof(double) * size);

	struct timeval t1, t2;

	gettimeofday(&t1, NULL);
	lap(size, dist_tabel, rowsol, colsol, u, v);
	gettimeofday(&t2, NULL);

	//cout << "time taken :: " << (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0 << " s" << endl;
	//printf("\nsize: %d, %d", size1, size2);	
	//Mat img = Mat::zeros(960, 1280, CV_8UC3);
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		//int j = rowsol[i];
		int j = colsol[i];
		//printf("\nC[%d][%d]: %f", i, j, dist_tabel[i][j]);	
		if(i >= r1.size())
		{
			matches.push_back(std::pair<int, int>(-1, j));
			continue;
		}else if(j >= r2.size())
		{
			matches.push_back(std::pair<int, int>(i, -1));
			continue;
		}else 
		{
			matches.push_back(std::pair<int, int>(i, j));
		}
		
	}
	
	for (int i=0; i< size; i++) free(dist_tabel[i]);
	free(dist_tabel);
	free(colsol); free(rowsol);
	free(u); free(v);
}


void update_robot_list( vector<RobotFeature>& all_robot, 
						vector< vector<Point> >& all_contourPoly, 
						vector< queue<int> >& observe_cnt,
						vector<RobotFeature>& current_robot, 
						vector< vector<Point> >& current_contourPoly, 
						vector<std::pair<int, int> >& matches)
{

	//const int observe_history_len = 5;
	for(int i = 0; i < matches.size(); i++)
	{
		int id1 = matches[i].first;
		int id2 = matches[i].second;
		if(id1 >= 0 && id2 >= 0)
		{
			all_robot[id1] = current_robot[id2];
			all_contourPoly[id1] = current_contourPoly[id2];
			observe_cnt[id1].push(observe_cnt[id1].back() + 1);
		}else if(id1 < 0 && id2 >= 0) 
		{
			all_robot.push_back(current_robot[id2]);
			all_contourPoly.push_back(current_contourPoly[id2]);
			queue<int> cnt;
			cnt.push(1);
			observe_cnt.push_back(cnt);
		}else if(id1 >= 0 && id2 < 0) //the robot is in the table but is not observed in this frame 
		{
			observe_cnt[id1].push(observe_cnt[id1].back());
		}
		
		if(observe_cnt[i].size() > OBSERVE_HISTORY_LEN)//observe cnt
		{
			observe_cnt[i].pop();
		}
		
	}
	
	for(int i = all_robot.size() - 1; i >= 0; i--)
	{
		if(observe_cnt[i].size() >= OBSERVE_HISTORY_LEN && (observe_cnt[i].back() - observe_cnt[i].front()) == 0)
		{
			//printf("\nremove a robot from the list");
			all_robot.erase(all_robot.begin() + i);
			all_contourPoly.erase(all_contourPoly.begin() + i);
			observe_cnt.erase(observe_cnt.begin() + i);
		}
	}
	
}

int robotTrackInit(char* calib_filename, char* green_svm_filename, char* red_svm_filename) 
{
	#if EDGE_BASED == 1
		return calibInit(calib_filename);
	#else
		int ret2 = calibInit(calib_filename);
		int ret1 =  svmInit(green_svm_filename, red_svm_filename);
		if(ret1 == -1 || ret2 == -1)
		{
			return -1;
		}else 
		{
			return 0;
		}
	#endif 
}


int calibInit(char* calib_filename) 
{
	bool ret = readCalibrationResult(calib_filename, K, distCoeff, image_width, image_height);
    if(ret == -1) 
    {
    	printf("\ncan not read the calibration result");
    	return -1;
    }
    
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

vector< vector<Point> > effective_contourPoly(vector< vector<Point> >& all_contourPoly, vector< queue<int> > observe_cnt)
{
	vector< vector<Point> > eff_contourPoly;
	for (unsigned int i = 0; i < all_contourPoly.size(); i++)
	{
		if(observe_cnt[i].front() > EFF_ROBOT_THRESHOLD)
		{
			eff_contourPoly.push_back(all_contourPoly[i]);
		}
	}
	
	return eff_contourPoly;
}


std::vector<Eigen::VectorXd> robotTrack(Mat& frame)
{
	
	int64_t start = get_timestamp();   
	static int frame_cnt = 0;
    static int detect_cnt = 0;
    double diff = 0.0;
	
    vector< vector<Point> > contourPoly, contourPoly_prev;
    vector<RobotFeature>  robot_features, robot_features_prev;
    
    static vector<RobotFeature>  all_robot;//all the robots
    static vector< vector<Point> > all_contourPoly;
    static vector< queue<int> > observe_cnt;
    
    vector< vector<Point> > eff_contourPoly;//if a robot is seen for more than n times, it is an effective_tobot
    vector<std::pair<int, int> > matches;
	char label[3];
	Mat mask;
	std::vector<Eigen::VectorXd> robotPosition;
	if(frame.empty() ) 
	{
		return robotPosition;
	}
	
	#if EDGE_BASED == 1
	//remap( frame, frame, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	contourPoly = robotDetect(frame);	
	
	#else 
	robotSegment(frame, mask);
	//imshow("mask", mask);
	//waitKey();
	contourPoly = findPattern(mask);
	#endif 
	calc_features(frame, contourPoly, robot_features);
	
	robotMatch(all_robot, robot_features, matches);
	
	update_robot_list(all_robot, all_contourPoly, observe_cnt, robot_features, contourPoly, matches);
	
	eff_contourPoly = effective_contourPoly(all_contourPoly, observe_cnt);
	
	//robotPosition = robotCenter(all_contourPoly);
//	for (unsigned int i = 0; i < contourPoly.size(); i++)
//	{
//		Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
//		drawContours(frame, all_contourPoly, i, color, 2, 8);
//	}
	
	robotPosition = robotCenter(eff_contourPoly);
	for (unsigned int i = 0; i < eff_contourPoly.size(); i++)
	{
		Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
		drawContours(frame, eff_contourPoly, i, color, 2, 8);
	}
	
	
	frame_cnt++;
	detect_cnt += contourPoly.size();
	
	double detect_rate = (double)detect_cnt/frame_cnt; //average robot in one frame
    //printf("\ndetect rate: %f", detect_rate);
	
	int64_t current = get_timestamp();   
	double cost_time = (current - start)/1000000.0;
	//cout << "cost time: " << cost_time << endl;  
	return robotPosition;
}


/*
std::vector<Eigen::Vector3d> robotTrack(Mat& frame)
{
	
	int64_t start = get_timestamp();   
	static int frame_cnt = 0;
    static int detect_cnt = 0;
    double diff = 0.0;
	
    vector< vector<Point> > contourPoly, contourPoly_prev;
    vector<RobotFeature>  robot_features, robot_features_prev;
    
    static vector<RobotFeature>  all_robot;//all the robots
    static vector< vector<Point> > all_contourPoly;
    static vector< queue<int> > observe_cnt;
    
    vector<std::pair<int, int> > matches;
	char label[3];
	
	std::vector<Eigen::Vector3d> robotPosition;
	if(frame.empty() ) 
	{
		return robotPosition;
	}
	
	
	
	remap( frame, frame, map1, map2, INTER_LINEAR, BORDER_CONSTANT );
	contourPoly = robotDetect(frame);	
	calc_features(frame, contourPoly, robot_features);
	robotMatch(all_robot, robot_features, matches);
	update_robot_list(all_robot, all_contourPoly, observe_cnt, robot_features, contourPoly, matches);
	
	robotPosition = robotCenter(all_contourPoly);
	
	
	frame_cnt++;
	detect_cnt += contourPoly.size();
	
	double detect_rate = (double)detect_cnt/frame_cnt; //average robot in one frame
    //printf("\ndetect rate: %f", detect_rate);
	
	int64_t current = get_timestamp();   
	double cost_time = (current - start)/1000000.0;
	//cout << "cost time: " << cost_time << endl;  
	return robotPosition;
}
*/


//normlized position
std::vector<Eigen::VectorXd> robotCenter(vector< vector<Point> > contours)
{
	Point2f center;
	float radius = 0.0;
	std::vector<Eigen::VectorXd> normlizedCenter(contours.size());
	cv::Mat src = Mat::zeros(1, 2, CV_32FC2);
	cv::Mat m = Mat::zeros(3, 1, CV_64FC1);
	std::vector<Point2f> p2;
	
	
	for(int i = 0; i < contours.size(); i++)
	{
		 Eigen::VectorXd cc(6);
		 
		 minEnclosingCircle( contours[i], center, radius);
		 src.at<cv::Vec2f>(0, 0)[0] = center.x;
		 src.at<cv::Vec2f>(0, 0)[1] = center.y;
		 
		 src.at<cv::Vec2f>(0, 1)[0] = (contours[i][1].x + contours[i][0].x)/2.0;
		 src.at<cv::Vec2f>(0, 1)[1] = (contours[i][1].y + contours[i][0].y)/2.0;
		 undistortPoints(src, src, K, distCoeff);
		 
		 m.at<double>(0) = src.at<cv::Vec2f>(0, 0)[0];
		 m.at<double>(1) = src.at<cv::Vec2f>(0, 0)[1];
		 m.at<double>(2) = 1.0;
		 //m = K.inv()*m;
		 cc(0) = m.at<double>(0);
		 cc(1) = m.at<double>(1);
		 cc(2) = m.at<double>(2);
		 
		 //direction:
		 m.at<double>(0) = src.at<cv::Vec2f>(0, 1)[0];
		 m.at<double>(1) = src.at<cv::Vec2f>(0, 1)[1];
		 m.at<double>(2) = 1.0;
		 //m = K.inv()*m;
		
		 cc(3) = m.at<double>(0);
		 cc(4) = m.at<double>(1);
		 cc(5) = m.at<double>(2);
		 
	 	normlizedCenter[i] = cc;
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
				
				//TODO, I don't understant
				if(tmp31 < 0.0 || tmp32 < 0.0)
				{
					colorMap[i/table_scale][j/table_scale][k/table_scale] = 255;
				}else 
				{
					colorMap[i/table_scale][j/table_scale][k/table_scale] = 0;
				}
				
//				m.at<float>(2) = k;
//				m.at<float>(5) = (float)k*k;
//				float response_red = svm_red->predict(m.t());
//				float response_green = svm_green->predict(m.t());
//				
//				cout << tmp31 << " " << tmp32 << " " << response_green << " " << response_red << 	 endl;				
//				if(response_red == 1 || response_green == 1)
//				{
//					colorMap[i/table_scale][j/table_scale][k/table_scale] = 255;
//				}else
//				{
//					colorMap[i/table_scale][j/table_scale][k/table_scale] = 0;
//				}
			}
		}
	} 
	cout << "initialization OK" << endl;
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
			
			id1 = p[0]/table_scale;
			id2 = p[1]/table_scale;
			id3 = p[2]/table_scale;
			p += 3;
			//cout << "id: " << id1 << " " << id2 << "  " << id3 << endl;
			mask.at<unsigned char>(i, j) = colorMap[id1][id2][id3];
		}
	}
}


void free_svm()
{
	for (unsigned int i = 0; i < 256; i += table_scale)
	{
		for (unsigned int j = 0; j < 256; j += table_scale)
		{
			free(colorMap[i/table_scale][j/table_scale]);
		}
		free(colorMap[i/table_scale]);
	}
	free(colorMap);
}


   
     
   
