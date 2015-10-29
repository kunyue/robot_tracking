#include "robotFeature.h"
#include "math.h"
#include "stdlib.h"
#include "util.h"


double feature_dist(RobotFeature& r1, RobotFeature& r2)
{
	double dist = 0.0;
	dist += distance(r1.shape_center, r2.shape_center)*30;
	double ang1 = angle(r1.shape_center, r1.dir_center);
	double ang2 = angle(r2.shape_center, r2.dir_center);

	dist += normalize_angle_radian(ang1 - ang2) *100.0;
	//distance += abs(r1.theta - r2.theta)*30;//weighted parameter
	dist += abs(r1.area - r2.area);
	dist += abs(r1.r - r2.r)*20;
	dist += abs(r1.g - r2.g)*20;
	dist += abs(r1.b - r2.b)*20;
	return dist;
}

