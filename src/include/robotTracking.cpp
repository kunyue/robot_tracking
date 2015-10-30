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

#define EFF_AREA_MIN_THRESHOLD 200.0
#define EFF_AREA_MAX_THRESHOLD 5000.0
#define OUTLIER_THRESHOLD 800.0

#define EFF_ROBOT_THRESHOLD 5
#define OBSERVE_HISTORY_LEN 10 //observe_history_len


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


vector<Point> alignShape(vector<Point>& contour);
vector< vector<Point> > findPattern(Mat& bwImg);
vector< vector<Point> > findBox(Mat& bwImg);
vector<cv::Point2d> getTemplate();
std::vector<Eigen::VectorXd> robotCenter(vector< RobotFeature > & features, std::vector<int>& eff_id);

vector<Point2f> shape_center(std::vector< std::vector<Point> >& contours);


vector<Point2f> mass_center(std::vector< std::vector<Point> > contours, Mat& color_mask);

// template<class T>
// std::vector<Point2f> robot_direction(std::vector< std::vector<T> > contours, vector<T>&mass_centers, vector<T>& shape_centers);

std::vector<Eigen::VectorXd> robot_pos_dir(vector< Point2f >& shape_centers, std::vector<Point2f>& robot_directions);


vector< int > effective_id(vector< queue<int> > observe_cnt);

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



vector< vector<Point> > findBox(Mat& bwImg)
{
	vector< vector<Point> > contours, contoursPoly;

	vector<Vec4i>hierarchy;
	int64_t start = 0, end = 0;
	
	//start = get_timestamp();  
	findContours(bwImg, contours, hierarchy, /*CV_RETR_LIST*/ CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//end = get_timestamp();
	//cout << "find contours: " << (end - start)/1000000.0 << endl;
	//start = get_timestamp();	
	//vector<RotatedRect> minRect;
	for(int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		
		if(area > EFF_AREA_MIN_THRESHOLD && area < EFF_AREA_MAX_THRESHOLD)
		{
			
			RotatedRect rect = minAreaRect( Mat(contours[i]) );
			//minRect.push_back(rect);
			Point2f vertices[4];
			rect.points(vertices);
			double len1 = distance(vertices[0], vertices[1]);
			double len2 = distance(vertices[1], vertices[2]);
			double long_axis = max(len1, len2);
			double short_axis = min(len1, len2);
			double ratio = long_axis/short_axis;
			double rectArea = long_axis*short_axis;
			double area_ratio = area/rectArea;


			if(ratio < 1.2 || ratio > 3.0 || area_ratio < 0.6)
			{
				continue;
			}
			//cout << "ratio: " << ratio << " area_ratio: " << area_ratio << endl;
			vector<Point> contour;
			for (int j = 0; j < 4; j++)
			{
				contour.push_back(vertices[j]);
			}
			contour = alignShape(contour);
			contoursPoly.push_back(contour);
		}
	}

	return contoursPoly;
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
	//start = get_timestamp();	
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

//contour[0] and contour[1] is the long axis
vector<Point> alignShape(vector<Point>& contour)
{
	//assert(contour.size() == 8);
	vector<Point> alignedContour(contour.size());
	vector<float> lengths(contour.size());
	for(int i = 0; i < contour.size(); i++)
	{
		lengths[i] = norm( contour[(i + 1)%contour.size()] - contour[i] );
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
#if 0	
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
#else 
	bool counterClockwise = true;
#endif 

	
	if(counterClockwise)
	{
		for(int i = 0; i < contour.size(); i++)
		{
			alignedContour[i] = contour[(i + maxID)%contour.size()];
		}
	}else 
	{
		for(int i = 0; i < contour.size(); i++)
		{
			alignedContour[i] = contour[(maxID + 1 - i)%contour.size()];
		}
	}

	return alignedContour;
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
	//contourPoly = findBox(edgeImg);
	//end = get_timestamp();
	
	
	//cout << "find patterns: " << (end - start)/1000000.0 << endl;
	
	return contourPoly;
}

//compute features in the polygon
void calc_features(cv::Mat img, 
	vector< vector<Point> >& contours, 
	std::vector<Point2f>& shape_centers, 
	std::vector<Point2f>& mass_centers, 
	std::vector<Point2f>& dir_centers,
	vector<RobotFeature>& robot_features)
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
		features.shape_center = shape_centers[i];
		features.mass_center = mass_centers[i];
		features.dir_center = dir_centers[i];

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

//just for output
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

vector< int > effective_id(vector< queue<int> > observe_cnt)
{
	vector< int > eff_id(observe_cnt.size() );
	for (unsigned int i = 0; i < observe_cnt.size(); i++)
	{
		if(observe_cnt[i].front() > EFF_ROBOT_THRESHOLD)
		{
			eff_id[i] = 1;
		}else
		{
			eff_id[i] = 0;
		}
	}
	return eff_id;
}


/*
std::vector<Eigen::VectorXd> robotTrack(Mat& frame)
{
	
	int64_t start = get_timestamp();   
	static int frame_cnt = 0;
    static int detect_cnt = 0;
    double diff = 0.0;
	
    vector< vector<Point> > contourPoly, contourPoly_prev;
    vector<RobotFeature>  robot_features, robot_features_prev;
    std::vector< Point > robot_center; //mass center
    
    static vector<RobotFeature>  all_robot;//all the robots
    static vector< vector<Point> > all_contourPoly;
    static vector< queue<int> > observe_cnt;
    
    vector< vector<Point> > eff_contourPoly;//if a robot is seen for more than n times, it is an effective_tobot
    vector<std::pair<int, int> > matches;
	char label[3];
	Mat mask, mask2;
	std::vector<Eigen::VectorXd> robotPosition;
	if(frame.empty() ) 
	{
		return robotPosition;
	}
	
	remap( frame, frame, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	#if EDGE_BASED == 1
	contourPoly = robotDetect(frame);	
	
	#else 
	robotSegment(frame, mask);
	
	//imshow("mask", mask);
	//waitKey(10);
	//contourPoly = findPattern(mask);
	mask.copyTo(mask2);
	contourPoly = findBox(mask2);//find contours will destory the image 


	//imshow("mask", mask);
	//waitKey(10);
	#endif 

	std::vector<Point2f> mass_centers = mass_center(contourPoly ,mask);
	std::vector<Point2f> shape_centers = shape_center(contourPoly);
	std::vector<Point2f> robot_directions = robot_direction(contourPoly, mass_centers, shape_centers);


	// for (unsigned int i = 0; i < contourPoly.size(); i++)
	// {
	// 	Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
		
	// 	line(frame, mass_centers[i], shape_centers[i],  color, 2, 8);
	// 	circle(frame, shape_centers[i], 2.0, color, 2, 8);
	// 	circle(frame, shape_centers[i], 2.0, color, 2, 8);
	// 	circle(frame, robot_directions[i], 2.0, color, 2, 8);
	// 	cout << (shape_centers[i] - mass_centers[i]) << endl;
	// }



	//calc_features(frame, contourPoly, robot_features);
	calc_features(frame, contourPoly, shape_centers, mass_centers, robot_directions, robot_features);
	robotMatch(all_robot, robot_features, matches);
	


	update_robot_list(all_robot, all_contourPoly, observe_cnt, robot_features, contourPoly, matches);
	
	//eff_contourPoly = effective_contourPoly(all_contourPoly, observe_cnt);

	std::vector<int> eff_ids = effective_id(observe_cnt);


	int count = 0;
	for (unsigned int i = 0; i < all_contourPoly.size(); i++)
	{
		if(eff_ids[i])
		{
			Scalar color = ColorTable[count%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
			drawContours(frame, all_contourPoly, i, color, 2, 8);
			
			circle(frame, all_robot[i].shape_center, 2.0, color, 2, 8);
			circle(frame, all_robot[i].dir_center, 2.0, color, 2, 8);
			count++;
		}
		
	}
	
	//TODO 
	robotPosition = robotCenter(all_robot, eff_ids);

	
	// for (unsigned int i = 0; i < eff_contourPoly.size(); i++)
	// {
	// 	Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
	// 	drawContours(frame, eff_contourPoly, i, color, 2, 8);
	// 	circle(frame, eff_contourPoly[i][0], 2.0, color, 2, 8);
	// 	circle(frame, eff_contourPoly[i][1], 2.0, color, 2, 8);
	// }
	
	
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

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 0;



//use camshift to track the robot
std::vector<Eigen::VectorXd> camshiftTrack(Mat& frame)
{
	int64_t start = get_timestamp();   

	static int track_state = DETECTING;
	
    static Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

    Mat hsv, hue, mask, mask2, color_mask;

    static Mat hist, backproj, histimg = Mat::zeros(200, 320, CV_8UC3);
    static std::vector<Mat> hists;
    static vector<Rect> trackWindows;
    static vector<RotatedRect> robot_rects;

    static std::vector<Point> contour;
	std::vector< vector<Point> > contours;

	static int frame_cnt = 0;
    static int detect_cnt = 0;
    double diff = 0.0;
	
    vector< vector<Point> > contourPoly;
    vector<RobotFeature>  robot_features;

 
	char label[3];
	std::vector<Eigen::VectorXd> robotPose;//output

	if(frame.empty() ) 
	{
		return robotPose;
	}
	
	remap( frame, frame, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	robotSegment(frame, color_mask);//mask for red and green region 

	// imshow("mask", color_mask);

	if(track_state == DETECTING)
	{
		
		//cout << "detecting: " << " ";

		
		//waitKey(10);
		//contourPoly = findPattern(mask);

		contourPoly = findBox(color_mask);//find contours will destory the image 

		if(contourPoly.size() >= 1 && contourPoly.size() <= 10)
		{
			track_state = TRACKING;
			cvtColor(frame, hsv, COLOR_BGR2HSV);

			//initialize for tracking
			hists.clear();
			trackWindows.clear();

			for (int i = 0; i < contourPoly.size(); ++i)
			{
				Rect selection = boundingRect(contourPoly[i]); 

				//Rect selection = maxRect();


				cvtColor(frame, hsv, COLOR_BGR2HSV);


				if(selection.x < 0) selection.x = 0;  
				if(selection.y < 0) selection.y = 0;
				if(selection.x + selection.width > frame.cols) selection.width = frame.cols - selection.x - 1;
				if(selection.y + selection.height > frame.rows) selection.height = frame.rows - selection.y - 1;   

				            	
                int _vmin = 0; 
                int _vmax = 255;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(0, 256, MAX(_vmin, _vmax)), mask);

                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                //cout << "selection: " << selection << endl;
                
                Mat roi(hue, selection), maskroi(mask, selection);
                
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                
                normalize(hist, hist, 0, 255, NORM_MINMAX);
                trackWindow = selection;

                hists.push_back(hist);
                trackWindows.push_back(trackWindow);

				#if 0
                histimg = Scalar::all(0);
                int binW = histimg.cols / hsize;
                Mat buf(1, hsize, CV_8UC3);
                for( int i = 0; i < hsize; i++ )
                    buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                cvtColor(buf, buf, COLOR_HSV2BGR);
                for( int i = 0; i < hsize; i++ )
                {
                    int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                    rectangle( histimg, Point(i*binW,histimg.rows),
                               Point((i+1)*binW,histimg.rows - val),
                               Scalar(buf.at<Vec3b>(i)), -1, 8 );
                }
				#endif 
	            
			}
		

		}
	}else if(track_state == TRACKING)
	{
	 	
	 	// cout << "tracking: " << hists.size() << " robots" << endl; 
	 	
 	 	cvtColor(frame, hsv, COLOR_BGR2HSV);	
        
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);

        int success_track_cnt = 0;
        contours.clear();


        for (int i = 0; i < hists.size(); ++i)
        {
        	hists[i].copyTo(hist);


        	trackWindow = trackWindows[i]; 

        	//infomation from the past frame,  1 hist, 2, trackWindow
		 	calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
	        //backproj &= mask;
	       
	        RotatedRect trackBox = CamShift(backproj, trackWindow,
	                            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
	        

	        if( trackWindow.area() >= EFF_AREA_MIN_THRESHOLD && trackWindow.area() <= EFF_AREA_MAX_THRESHOLD)
	        {
	        	//track_state = DETECTING;
	        	success_track_cnt++;
	        	//ellipse( frame, trackBox, Scalar(0, 0, 255), 3, 8);
        		//drawrect(frame, trackBox, Scalar(0, 0, 255));

        		trackWindows[i] = trackWindow;

		        rect_to_contour(resize_rect(trackBox, 1.0), contour);
		        //cout << "contour: " << contour << endl;
		        contours.push_back(contour);
		        
	        }
	        

        }

        double track_rate = (double)success_track_cnt/hists.size();
        //if(track_rate < 0.5)
        if(success_track_cnt <= 0)
        {
        	track_state = DETECTING;
        }
       
	}
	

	std::vector<Point2f> mass_centers, shape_centers;
	//std::vector<Point2f> mass_centers = mass_center(contours, color_mask);
	//std::vector<Point2f> shape_centers = shape_center(contours);

	robot_center(contours, color_mask, shape_centers, mass_centers);
	std::vector<Point2f> dir_centers = robot_direction(contours, mass_centers, shape_centers);
	
	robotPose = normalized_robot_pose(shape_centers, dir_centers);
	
	//imshow("frame", frame);
	//waitKey(20);

	for (unsigned int i = 0; i < contours.size(); i++)
	{
		
		Scalar color = ColorTable[i%COLOR_CNT];//Scalar(rand() & 255, rand()& 255, rand() & 255);
		drawContours(frame, contours, i, color, 2, 8);
		
		circle(frame, shape_centers[i], 2.0, color, 2, 8);
		circle(frame, dir_centers[i], 2.0, color, 2, 8);
	}

	return robotPose;
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
std::vector<Eigen::VectorXd> robotCenter(vector< RobotFeature > & features, std::vector<int>& eff_ids)
{
	
	float radius = 0.0;
	std::vector<Eigen::VectorXd> normlizedCenter;
	cv::Mat src = Mat::zeros(1, 2, CV_32FC2);
	cv::Mat m = Mat::zeros(3, 1, CV_64FC1);
	std::vector<Point2f> p2;
	
	
	for(int i = 0; i < features.size(); i++)
	{
		 if(!eff_ids[i])
		 {
		 	continue;
		 }

		 Eigen::VectorXd cc(6);
		 m.at<double>(0) = features[i].shape_center.x;
		 m.at<double>(1) = features[i].shape_center.y;
		 m.at<double>(2) = 1.0;
		 m = K.inv()*m;
		 cc(0) = m.at<double>(0);
		 cc(1) = m.at<double>(1);
		 cc(2) = m.at<double>(2);
		 
		 //direction:
		 m.at<double>(0) = features[i].dir_center.x;
		 m.at<double>(1) = features[i].dir_center.y;
		 m.at<double>(2) = 1.0;
		 m = K.inv()*m;
		 cc(3) = m.at<double>(0);
		 cc(4) = m.at<double>(1);
		 cc(5) = m.at<double>(2);
		 
	 	normlizedCenter.push_back(cc);
	}
	return normlizedCenter;
}


//normlized position
std::vector<Eigen::VectorXd> normalized_robot_pose(
	std::vector<Point2f> shape_centers, 
	std::vector<Point2f> dir_centers)
{
	
	float radius = 0.0;
	std::vector<Eigen::VectorXd> normlizedCenter;
	cv::Mat src = Mat::zeros(1, 2, CV_32FC2);
	cv::Mat m = Mat::zeros(3, 1, CV_64FC1);
	std::vector<Point2f> p2;

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
		 m.at<double>(0) = dir_centers[i].x;
		 m.at<double>(1) = dir_centers[i].y;
		 m.at<double>(2) = 1.0;
		 m = K.inv()*m;
		 cc(3) = m.at<double>(0);
		 cc(4) = m.at<double>(1);
		 cc(5) = m.at<double>(2);
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


  //estimate direction of the robot
//use the mass center 

vector<Point2f> mass_center(std::vector< std::vector<Point> > contours, Mat& color_mask)
{
	cv::Mat mask = cv::Mat::zeros(color_mask.rows, color_mask.cols, CV_8UC1);
	vector<Point> contour;
	std::vector<Point2f> robotCenter;
	
	for(int i = 0; i < contours.size(); i++)
	{
		mask.setTo(Scalar(0));
		
		//cout << "contours[i]:" << contours[i] << endl;

		drawContours(mask, contours, i, Scalar(1), CV_FILLED);	
		
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


void robot_center(std::vector< std::vector<Point> > contours, 
				Mat& color_mask, 
				vector<Point2f>& mass_centers, 
				vector<Point2f>& shape_centers
				//vector<Point2f>& dir_centers,
				)
{
	cv::Mat mask = cv::Mat::zeros(color_mask.rows, color_mask.cols, CV_8UC1);
	vector<Point> contour;

	mass_centers.clear();
	shape_centers.clear();
	//dir_centers.clear();

	
	for(int i = 0; i < contours.size(); i++)
	{
		mask.setTo(Scalar(0));
		
		//cout << "contours[i]:" << contours[i] << endl;

		drawContours(mask, contours, i, Scalar(1), CV_FILLED);	
		
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
		mass_centers.push_back(center);


		sum_x = 0.0f;
		sum_y = 0.0f;
		count = 0;
		double dx, dy;
		r = 0, g = 0, b = 0;
		for (unsigned int j = 0; j < mask.rows; j++)
		{
			for (unsigned int k = 0; k < mask.cols; k++)
			{
				if(mask.at<unsigned char>(j, k))
				{
					dx = (k - center.x);
					dy = (j - center.y);
					sum_x += dx*dx*dx;
					sum_y += dy*dy*dy;
					count++;
				}				
			}
		}	
		double mean_x3 = pow(sum_x/count, 1.0/3.0);
		double mean_y3 = pow(sum_y/count, 1.0/3.0);

		shape_centers.push_back(Point2f(center.x + mean_x3, center.y + mean_y3) );
	}


}


vector<Point2f> shape_center(std::vector< std::vector<Point> >& contours)
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
     
   
