#include "ros/ros.h"
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
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
    cam_off_b << -0.08, 0.07, -0.10;
    cout << "Rc2b "<< endl << Rc2b << endl;
}
Vector3d get_robot_position(Vector3d position)
{
    Rb2w = att_body.normalized().toRotationMatrix();
    Rc2w = Rb2w * Rc2b;
    Rc2i = Ri2w.inverse() * Rc2w;
    //pos_i is normlized coordinate; the POS_I is the real coordinate; both are in intermidiate frame
    Vector3d pos_i, POS_I, POS_W, CAM_W;
    pos_i = Rc2i * position.normalized();
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
    char cm[1024];
    char red[1024];
    char green[1024];
    string ccm, rred, ggreen;
    bool image_view = false;
    n.getParam("cam_cal_file", ccm);
    n.getParam("red_thr_file", rred);
    n.getParam("green_thr_file", ggreen);
    n.getParam("show_image", image_view);
    cout << "cam_cal: " << ccm << endl;
    cout << "red_thr: " << rred << endl;
    cout << "green_thr: " << ggreen << endl;
    cout << "image_show: " << image_view << endl;
    std::strcpy(cm, ccm.c_str());
    std::strcpy(red, rred.c_str());
    std::strcpy(green, ggreen.c_str());
    int ret = robotTrackInit(cm, red, green);
    if (ret != 0)
        cout << "fail to read file" << endl;

    double invalid_pos_x, invalid_pos_y, invalid_pos_z;
    n.param("invalid_pos_x", invalid_pos_x, -1.0);
    n.param("invalid_pos_y", invalid_pos_y, -1.0);
    n.param("invalid_pos_z", invalid_pos_z, -1.0);


    image_transport::ImageTransport it(n);
    image_transport::Publisher vis_pub;
    
    if (image_view)
    {
        vis_pub = it.advertise("vis_img", 1);
    }
	
    init_rotation();
    ros::Rate loop(60);
    std::vector<Eigen::VectorXd> robotPosition;//normalized position
    while (n.ok())
    {
        if (image_ready)
        {
            image_ready = false;
            
            robotPosition = robotTrack(image);

            geometry_msgs::PoseStamped  robot;
            robot.header.stamp = tImage;
            robot.header.frame_id = "world";

            if (robotPosition.size() >= 1 && pos_body.z() > 0.3)
            {
                Eigen::Vector3d rob_pos = get_robot_position(robotPosition[0].segment(0, 3));
                robot.pose.position.x = rob_pos.x();
                robot.pose.position.y = rob_pos.y();
                robot.pose.position.z = rob_pos.z();
                Eigen::Vector3d head_pos = get_robot_position(robotPosition[0].segment(3, 3));
                Eigen::Vector3d head_ori = head_pos - rob_pos;
                double head_angle = atan2(head_ori.y(), head_ori.x());
                Eigen::Quaterniond robot_ori;
                robot_ori = AngleAxisd(head_angle, Eigen::Vector3d::UnitZ());
                robot.pose.orientation.x = robot_ori.x();
                robot.pose.orientation.y = robot_ori.y();
                robot.pose.orientation.z = robot_ori.z();
                robot.pose.orientation.w = robot_ori.w();
                cout << "sd: " << robotPosition[0].transpose() << endl; 
                cout << "robot pose: "<< rob_pos.x() << " \t " << rob_pos.y() << " \t " << rob_pos.z() << " \tori: " << 180 * head_angle / M_PI << " quad: " << pos_body.x() << " " << pos_body.y() << " " << pos_body.z() << endl;
            }
            else // publish invalid position
            {
                robot.pose.position.x = invalid_pos_x;
                robot.pose.position.y = invalid_pos_y;
                robot.pose.position.z = invalid_pos_z;
                robot.pose.orientation.x = 0;
                robot.pose.orientation.y = 0;
                robot.pose.orientation.z = 0;
                robot.pose.orientation.w = 1;
            }
            p1.publish(robot);

            if (image_view)
            {
                if (true)
                {
                    imshow("frame", image);
                    char key = waitKey(30);
                    if (key == 27)
                    {
                        break;
                    }
                }
                if (true)
                {
                    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(robot.header, "bgr8", image).toImageMsg();
                    vis_pub.publish(msg);
                }
            }
        }
        loop.sleep();
        ros::spinOnce();
    }
}
