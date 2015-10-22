#ifndef __ROBOTFEATURE_H
#define __ROBOTFEATURE_H

typedef struct 
{
	double x, y, theta;
	//double vx, vy, w;
	double area;
	double r, g, b;//mean color
	
}RobotFeature;

double feature_dist(RobotFeature& r1, RobotFeature& r2);	

#endif 
