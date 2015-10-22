#include "ros/ros.h"
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include "include/robotTracking.h"
using namespace std;
using namespace cv;
using namespace Eigen;

ros::Time tImage;
cv::Mat image, frame, dst;
bool image_ready = false;
void image_callback(const sensor_msgs::Image::ConstPtr &msg)
{
    image_ready = true;
    tImage = msg->header.stamp;
    image  = cv_bridge::toCvCopy(msg, string("bgr8"))->image;
}

//odometry related messages
Vector3d  pos_body;
Quaterniond att_body;
void odom_callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    pos_body.x() = msg->pose.pose.position.x;
    pos_body.y() = msg->pose.pose.position.y;
    pos_body.z() = msg->pose.pose.position.z;
    att_body.w() = msg->pose.pose.orientation.w;
    att_body.x() = msg->pose.pose.orientation.x;
    att_body.y() = msg->pose.pose.orientation.y;
    att_body.z() = msg->pose.pose.orientation.z;
}

// b -> body  c -> camera  w -> world i -> intermidiate (initial camera attitude)
Eigen::Matrix3d Rc2b, Rb2w, Ri2w, Rc2w, Rc2i;
Vector3d cam_off_b;
void init_rotation(void)
{
    Rc2b = Eigen::AngleAxisd(-0.5 * M_PI, Vector3d::UnitZ()) * Eigen::AngleAxisd(M_PI, Vector3d::UnitX());
    Ri2w = Rc2b;
    cam_off_b << 0.1, 0., -0.1;
}
Vector3d get_robot_position(Vector3d position)
{
    Rb2w = att_body.normalized().toRotationMatrix();
    Rc2w = Rb2w * Rc2b;
    Rc2i = Ri2w.inverse() * Rc2w;
    //pos_i is normlized coordinate; the POS_I is the real coordinate; both are in intermidiate frame
    Vector3d pos_i, POS_I, POS_W, CAM_W;
    pos_i = Rc2i * position;
    pos_i = pos_i / pos_i.z();
    POS_I = pos_body.z() * pos_i;
    CAM_W = pos_body + Rb2w * cam_off_b;
    POS_W = Ri2w * POS_I + CAM_W;
    return POS_W;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ball_track");
    ros::NodeHandle n("~");
    ros::Subscriber s2 = n.subscribe("image", 100, image_callback);
    ros::Subscriber s1 = n.subscribe("odom", 100, odom_callback);
    ros::Publisher  p1 = n.advertise<geometry_msgs::PoseStamped>("robot", 100);
    cout << "read camera info" << endl;
    char cm[] = "/home/ksu/Data/ros_ws/catkin_ws/src/robotTracking/config/camera.yml";
    char red[] = "/home/ksu/Data/ros_ws/catkin_ws/src/robotTracking/config/color_red_bluefox.yml";
    char green[] = "/home/ksu/Data/ros_ws/catkin_ws/src/robotTracking/config/color_green_bluefox.yml";
    init_rotation();
    int ret = robotTrackInit(cm, red, green);
    cout << "end" << endl;
    if (ret != 0)
        cout << "fail to read file" << endl;
    ros::Rate loop(60);
    std::vector<Eigen::Vector3d> robotPosition;//normalized position
    while (n.ok())
    {
        if (image_ready)
        {
            ros::Time t1 = ros::Time::now();
            image_ready = false;
            robotPosition = robotTrack(image);
            for (uint32_t i = 0; i < robotPosition.size(); i++)
            {
                cout << "robot " << i << endl;
                cout << "robot normalized: " << robotPosition[i].transpose() << endl;
                cout << "robot position: " << get_robot_position(robotPosition[i]).transpose() << endl;
            }
            imshow("frame", image);
            ros::Time t2 = ros::Time::now();
            cout << "time consumption: " << (t2-t1).toSec() << " \t hz: " << 1 / (t2-t1).toSec() << endl;
            char key = waitKey(30);
            if (key == 27)
            {
                break;
            }
        }
        loop.sleep();
        ros::spinOnce();
    }
}
