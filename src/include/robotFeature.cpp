#include "robotFeature.h"
#include "math.h"
#include "stdlib.h"


double feature_dist(RobotFeature& r1, RobotFeature& r2)
{
	double distance = 0.0;
	distance += 0.2 * abs(r1.x - r2.x);
	distance += 0.2 * abs(r1.y - r2.y);
	distance += abs(r1.theta - r2.theta)*30;//weighted parameter
	distance += abs(r1.area - r2.area);
	distance += abs(r1.r - r2.r);
	distance += abs(r1.g - r2.g);
	distance += abs(r1.b - r2.b);
	return distance;
}

