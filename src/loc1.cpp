#include <iostream>
#include <math.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include<tf/tf.h>
#include <tf/transform_broadcaster.h>
#include "tf/transform_datatypes.h"
#include "tf_conversions/tf_eigen.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <boost/foreach.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "Eigen/Geometry"
#include <eigen3/Eigen/Dense>
#include <sstream>
#include <vector>
// #include "nanoflann.hpp"
#include <boost/format.hpp>

#include "nanoflann_pcl.h"

#define fx 20.0
#define fy 20.0

#define yaw0 0.0
#define pitch0 0.0
#define roll0 0.0
#define tx0 0.0
#define ty0 0.0
#define tz0 0.0

#define _surrounding_keyframe_search_radius 50.0

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace cv;
using namespace std;
// using namespace nanoflann;

static const std::string OPENCV_WINDOW = "Depth Image";

class ImageConverter
{

    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_right;
    image_transport::Subscriber image_sub_left;

public:
    bool rightFound=false,leftFound=false;
    Mat depth_image;
    cv_bridge::CvImagePtr cv_ptr_right;
    cv_bridge::CvImagePtr cv_ptr_left;
    ros::Time timeStamp;

    ImageConverter(ros::NodeHandle nh_) : it_(nh_)
    {
        image_sub_right = it_.subscribe("/camera/image_right", 1, &ImageConverter::imageCbRight, this);
        image_sub_left = it_.subscribe("/camera/image_right", 1, &ImageConverter::imageCbLeft, this);
        cv::namedWindow(OPENCV_WINDOW);
    }

    ~ImageConverter()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }

    // Convert and process image
    void imageCbRight(const sensor_msgs::ImageConstPtr &msg)
    {
        try
        {
            cv_ptr_right = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            rightFound=true;
            timeStamp=msg->header.stamp;
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Display image
        // cv::imshow(OPENCV_WINDOW, cv_ptr_right->image);
        // cv::waitKey(3);
    }
    void imageCbLeft(const sensor_msgs::ImageConstPtr &msg)
    {
        try
        {
            cv_ptr_left = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            leftFound=true;
            timeStamp=msg->header.stamp;
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Display image
        // cv::imshow(OPENCV_WINDOW, cv_ptr_left->image);
        // cv::waitKey(3);
    }
};

class PointCloudConverter
{
    //ros::Subscriber sub_;
    //ros::NodeHandle it_;

public:

    nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtree_local;

    vector<int> pointSearchIndLocal;
    //vector<float> pointSearchSqDisPrev;

    PointCloudConverter()//ros::NodeHandle nh_)// : it_(nh_)
    {
        //sub_ = it_.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/pointcloud_topic", 1, &PointCloudConverter::callback);
       // sub_ = it_.subscribe<sensor_msgs::PointCloud2>("/pointcloud_topic", 1, &PointCloudConverter::callback, this);

        std::string cloudDir = "/tmp/dump/cloudKeyPoses3D.pcd";
        pcl::PointCloud<pcl::PointXYZI>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI> (cloudDir, *tempCloud);

        kdtree_local.setInputCloud(tempCloud);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr getCloud(int index)
    {   
        

        pcl::PointCloud<pcl::PointXYZI>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud (new pcl::PointCloud<pcl::PointXYZI>);

        std::string cloudDir = (boost::format("/tmp/dump/%06d/cloud_corner.pcd") % index).str();

        pcl::io::loadPCDFile<pcl::PointXYZI> (cloudDir, *tempCloud);
        (*inputCloud) += (*tempCloud);

        cloudDir = (boost::format("/tmp/dump/%06d/cloud_surf.pcd") % index).str();

        pcl::io::loadPCDFile<pcl::PointXYZI> (cloudDir, *tempCloud);
        (*inputCloud) += (*tempCloud);

        cloudDir = (boost::format("/tmp/dump/%06d/cloud_outlier.pcd") % index).str();

        pcl::io::loadPCDFile<pcl::PointXYZI> (cloudDir, *tempCloud);
        (*inputCloud) += (*tempCloud);

        return inputCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr getlocalCloud(double roll, double pitch, double yaw, double tx, double ty, double tz)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud;
        // point_cloud->clear();
        pcl::PointXYZI currentRobotPosPoint;

        currentRobotPosPoint.x = tx;
        currentRobotPosPoint.y = ty;
        currentRobotPosPoint.z = tz;

        vector<int> pointSearchInd;
        vector<float> pointSearchSqDis;

        kdtree_local.radiusSearch(
        currentRobotPosPoint, (double)_surrounding_keyframe_search_radius,
        pointSearchInd, pointSearchSqDis);

        for(int i=0; i<pointSearchInd.size(); i++)
        {
            auto it = find(pointSearchIndLocal.begin(), pointSearchIndLocal.end(), pointSearchInd[i]);

            if( it == pointSearchIndLocal.end())
            {
                pointSearchIndLocal.push_back(pointSearchInd[i]);
                // pointCloud.push_back;
            }
        }

        for(int i=0; i<pointSearchIndLocal.size(); i++)
        {
            auto it = find(pointSearchInd.begin(), pointSearchInd.end(), pointSearchIndLocal[i]);

            if( it == pointSearchInd.end())
            {
                pointSearchIndLocal.erase(pointSearchInd.begin() + i);
                i--;
            }
        }

        for(int i=0; i<pointSearchIndLocal.size(); i++)
        {
            (*point_cloud) += (*getCloud(pointSearchIndLocal[i]));
        }

        return point_cloud;
    }

    // Convert and process image
    // void callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    // {
    //     pcl::fromROSMsg(*msg, *point_cloud);
    // }
};

class TrackingConverter
{
    ros::NodeHandle node_;
    //ros::Subscriber sub_;
    //tf::TransformListener listener_;

public:
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener *tfListener;//(tfBuffer);
    //Eigen::Affine3d tksi;
    geometry_msgs::TransformStamped transformStamped;
    geometry_msgs::TransformStamped transformStampedPrev,tkshiTransform;
    tf::StampedTransform tfStamped,tfStampedPrev;
    bool firstFrameReceived=false;

    TrackingConverter(ros::NodeHandle nh_) : node_(nh_)
    {
        tfListener = new tf2_ros::TransformListener(tfBuffer);
    }

    //ros::Rate rate(10.0);
    //while (nh_.ok())
    bool startTrackingConverter()
    {
        //geometry_msgs::TransformStamped transformStamped;
        try
        {
            if(firstFrameReceived){
                transformStampedPrev=transformStamped;
            }
            transformStamped = tfBuffer.lookupTransform("camera", "world", ros::Time(0));
            if(firstFrameReceived){
                tf::transformStampedMsgToTF(transformStamped,tfStamped);
                tf::transformStampedMsgToTF(transformStampedPrev,tfStampedPrev);

                tf::Transform tfStampedPrevInv=tfStampedPrev.inverse();
                tfStamped.mult(tfStamped,tfStampedPrevInv);
                tf::transformStampedTFToMsg(tfStamped,transformStamped);
                
            }
            firstFrameReceived=true;
            //tksi = tf2::transformToEigen(transformStamped);

            return true;
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            //continue;

            return false;
        }
        // Convert transformStamped to Eigen
        // tksi = tf2::transformToEigen(transformStamped);

        //rate.sleep();
    }
};

ImageConverter *ic;

class Rat43Analytic : public ceres::SizedCostFunction<1, 6>
{
public:
    Rat43Analytic(const double x, const double y, const double z) : x_(x), y_(y), z_(z) {}
    virtual ~Rat43Analytic() {}
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        const double yaw = parameters[0][0];
        const double pitch = parameters[0][1];
        const double roll = parameters[0][2];
        const double tx = parameters[0][3];
        const double ty = parameters[0][4];
        const double tz = parameters[0][5];

        Eigen::Matrix<double, 4, 4> tksi;
        Eigen::Matrix<double, 4, 4> tcm;
        Eigen::Matrix<double, 4, 1> pi;

        double sy = sin(yaw);
        double cy = cos(yaw);
        double sp = sin(pitch);
        double cp = cos(pitch);
        double sr = sin(roll);
        double cr = cos(roll);


        tksi(0, 0) = cy * cp;
        tksi(0, 1) = cy * sp * sr - sy * cr;
        tksi(0, 2) = cy * sp * cr + sy * sr;
        tksi(1, 0) = sy * cp;
        tksi(1, 1) = sy * sp * sr + cy * cr;
        tksi(1, 2) = sy * sp * cr - cy * sr;
        tksi(2, 0) = -sp;
        tksi(2, 1) = cp * sr;
        tksi(2, 2) = cp * cr;
        tksi(0, 3) = tx;
        tksi(1, 3) = ty;
        tksi(2, 3) = tz;
        tksi(3, 0) = double(0);
        tksi(3, 1) = double(0);
        tksi(3, 2) = double(0);
        tksi(3, 3) = double(1);

        double sy0 = sin(yaw0);
        double cy0 = cos(yaw0);
        double sp0 = sin(pitch0);
        double cp0 = cos(pitch0);
        double sr0 = sin(roll0);
        double cr0 = cos(roll0);

        tcm(0, 0) = cy0 * cp0;
        tcm(0, 1) = cy0 * sp0 * sr0 - sy0 * cr0;
        tcm(0, 2) = cy0 * sp0 * cr0 + sy0 * sr0;
        tcm(1, 0) = sy0 * cp0;
        tcm(1, 1) = sy0 * sp0 * sr0 + cy0 * cr0;
        tcm(1, 2) = sy0 * sp0 * cr0 - cy0 * sr0;
        tcm(2, 0) = -sp0;
        tcm(2, 1) = cp0 * sr0;
        tcm(2, 2) = cp0 * cr0;
        tcm(0, 3) = tx0;
        tcm(1, 3) = ty0;
        tcm(2, 3) = tz0;
        tcm(3, 0) = double(0);
        tcm(3, 1) = double(0);
        tcm(3, 2) = double(0);
        tcm(3, 3) = double(1);

        pi(0, 0) = x_;
        pi(1, 0) = y_;
        pi(2, 0) = z_;
        pi(3, 0) = double(1.0);

        Eigen::Matrix<double, 4, 1> transformed_point = tksi * tcm * pi;

        Mat depth_image = ic->depth_image;

        residuals[0] = transformed_point(2, 0) - depth_image.at<uchar>(fx / transformed_point(2, 0), fy / transformed_point(2, 0));

        if (!jacobians)
            return true;
        double *jacobian = jacobians[0];
        if (!jacobian)
            return true;

        Eigen::Matrix<double, 1, 4> row;

        row << 0, 0, 1, 0;

        Eigen::Matrix<double, 3, 4> pi_jacobian;
        Eigen::Matrix<double, 4, 1> current_point = pi;
        pi_jacobian << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 1;
        pi_jacobian(0, 0) = fx / current_point(2, 0);
        pi_jacobian(0, 2) = -fx * current_point(0, 0) / pow(current_point(2, 0), 2);
        pi_jacobian(1, 1) = fy / current_point(2, 0);
        pi_jacobian(0, 2) = -fx * current_point(0, 0) / pow(current_point(2, 0), 2);

        double term1 = tcm(0, 0) * x_ + tcm(0, 1) * y_ + tcm(0, 2) * z_;
        double term2 = tcm(1, 0) * x_ + tcm(1, 1) * y_ + tcm(1, 2) * z_;
        double term3 = tcm(2, 0) * x_ + tcm(2, 1) * y_ + tcm(2, 2) * z_;

        Eigen::Matrix<double, 4, 6> tcm_pi_dot;
        tcm_pi_dot << 0, 0, 0, 1, 1, 1,
            0, 0, 0, 1, 1, 1,
            0, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0;

        tcm_pi_dot(0, 0) = (-sy * cp) * term1 + (sy * sp * sr - cy * cr) * term2 + (sy * sp * cr - cy * sr) * term3;
        tcm_pi_dot(1, 0) = (-cy * sp) * term1 + (cy * cp * sr) * term2 + (cy * cp * cr) * term3;
        tcm_pi_dot(2, 0) = 0.0 * term1 + (cy * sp * cr + sy * sr) * term2 - (cy * sp * sr + sy * cr) * term3;
        tcm_pi_dot(1, 0) = (cy * cp) * term1 + (cy * sp * sr + sy * cr) * term2 + (cy * sp * cr + sy * sr) * term3;
        tcm_pi_dot(1, 1) = (-sy * sp) * term1 + (sy * cp * sr) * term2 + (sy * cp * cr) * term3;
        tcm_pi_dot(1, 2) = 0.0 * term1 + (sy * sp * cr + cy * sr) * term2 - (sy * sp * sr + cy * cr) * term3;
        tcm_pi_dot(2, 1) = (-cp) * term1 - (sp * sr) * term2 - (sp * cr) * term3;
        tcm_pi_dot(2, 1) = 0.0 * term1 + (cp * cr) * term2 - (cp * sr) * term3;

        Mat depth_image_gradient(depth_image.rows, depth_image.cols, CV_8UC3, Scalar(0, 0, 0));

        Scharr(depth_image, depth_image_gradient, -1, 1, 1);

        Eigen::Matrix<double, 1, 3> del_d_by_del_x;

        del_d_by_del_x(0, 0) = depth_image_gradient.at<Vec3b>(fx / transformed_point(2, 0), fy / transformed_point(2, 0))[0];
        del_d_by_del_x(0, 1) = depth_image_gradient.at<Vec3b>(fx / transformed_point(2, 0), fy / transformed_point(2, 0))[1];
        del_d_by_del_x(0, 2) = depth_image_gradient.at<Vec3b>(fx / transformed_point(2, 0), fy / transformed_point(2, 0))[2];

        Eigen::Matrix<double, 1, 6> final_jacobian=row*tcm_pi_dot-del_d_by_del_x*pi_jacobian*tcm_pi_dot;

        jacobian[0] = final_jacobian(0, 0);
        jacobian[1] = final_jacobian(0, 1);
        jacobian[2] = final_jacobian(0, 2);
        jacobian[3] = final_jacobian(0, 3);
        jacobian[4] = final_jacobian(0, 4);
        jacobian[5] = final_jacobian(0, 5);

        return true;
    }

private:
    const double x_;
    const double y_;
    const double z_;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "loc");

    ros::NodeHandle nh_;
    
    PointCloudConverter pcc;
    ic = new ImageConverter(nh_);
    TrackingConverter tc(nh_);

    tf::TransformBroadcaster tfBroadcaster;

    ros::Rate rate(200);

    while(ros::ok())
    {
        
        ros::spinOnce();
        rate.sleep();

        if(!tc.startTrackingConverter())
            continue;
        
        if(!ic->leftFound || !ic->rightFound){
            continue;
        }
        ic->leftFound=false;
        ic->rightFound=false;

        geometry_msgs::TransformStamped initial_tksi = tc.transformStamped;

        // auto stereo=StereoBM::create(16,15);
        Ptr<StereoBM> bm = StereoBM::create(16,9);
        bm->compute(ic->cv_ptr_left->image.clone(),ic->cv_ptr_right->image.clone(),ic->depth_image);

        double yaw = 0.0;
        double pitch = 0.0;
        double roll = 0.0;
        double tx = initial_tksi.transform.translation.x;
        double ty = initial_tksi.transform.translation.y;
        double tz = initial_tksi.transform.translation.z;

        tf::Matrix3x3(tf::Quaternion(initial_tksi.transform.rotation.x, initial_tksi.transform.rotation.y, initial_tksi.transform.rotation.z, initial_tksi.transform.rotation.w)).getRPY(roll, pitch, yaw);

        pcl::PointCloud<pcl::PointXYZI>::ConstPtr pointcloud = pcc.getlocalCloud(roll, pitch, yaw, tx, ty, tz);

        Problem problem;

        //access pointcloud here
        for (int i = 0; i < pointcloud->points.size(); i++)
        {
            CostFunction *cost_function = new Rat43Analytic(pointcloud->points[i].x, pointcloud->points[i].y, pointcloud->points[i].z);
            problem.AddResidualBlock(cost_function, NULL, &yaw, &pitch, &roll, &tx, &ty, &tz);
        }

        Solver::Options options;
        options.max_num_iterations = 25;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        Solver::Summary summary;
        Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";
        std::cout << "yaw : " << yaw << " pitch : " << pitch << "roll : " << roll << "\n";
        std::cout << "tx : " << tx << " ty : " << ty << "tz : " << tz << "\n";

        tf::Quaternion q=tf::Quaternion(yaw,pitch,roll);
        tf::Vector3 t=tf::Vector3(tx,ty,tz);
        tf::Transform finalTransform=tf::Transform(q,t);
        tf::StampedTransform finalStampedTransform=tf::StampedTransform(finalTransform,ic->timeStamp,"/map","/car");
        tfBroadcaster.sendTransform(finalStampedTransform);

    }

    return 0;
}