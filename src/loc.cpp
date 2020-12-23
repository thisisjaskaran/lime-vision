#include <iostream>
#include <math.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
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

#define fx 20
#define fy 20

static const std::string OPENCV_WINDOW = "Depth Image";

// Subscribe to ros image, convert to OpenCV image, and process
class ImageConverter
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;

public:
    cv_bridge::CvImagePtr cv_ptr;

    ImageConverter() : it_(nh_)
    {
        image_sub_ = it_.subscribe("/camera/image_raw", 1, &ImageConverter::imageCb, this);
        cv::namedWindow(OPENCV_WINDOW);
    }

    ~ImageConverter()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }

    // Convert and process image
    void imageCb(const sensor_msgs::ImageConstPtr &msg)
    {

        try
        {
            this.cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Display image
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3);
    }
};

class PointCloudConverter
{
    ros::NodeHandle nh_;
    ros::Subscriber sub_;

public:
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr point_cloud;
    PointCloudConverter() : it_(nh_)
    {
        sub = it_.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/pointcloud_topic", 1, &PointCloudConverter::callback);
    }

    // Convert and process image
    void callback(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &msg)
    {
        this.point_cloud = msg;
    }
};

class TrackingConverter
{
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    tf::TransformListener listener_;

public:
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    ros::Rate rate(10.0);
    while (nh_.ok())
    {
        geometry_msgs::TransformStamped transformStamped;
        try
        {
            transformStamped = tfBuffer.lookupTransform("target_frame", "source_frame", ros::Time(0));
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }

        // Convert transformStamped to Eigen
        doTransform(const tf2::Stamped<Eigen::Isometry3d> &t_in, tf2::Stamped<Eigen::Isometry3d> &t_out, const geometry_msgs::TransformStamped &transform)
            tf2::Stamped<Eigen::Isometry3d>

                rate.sleep();
    }
};

void ComputeDepthIntoProjectionAndJacobian(Eigen::Matrix<double, 4, 1> transformed_point, Eigen::Matrix<double, 4, 1> current_point, double *value, Eigen::Matrix<double, 1, 4> *jacobian)
{
    ImageConverter ic;
    Mat depth_image = ic.cv_ptr->image.clone();
    *value = depth_image.at<uchar>(f * transformed_point(0, 0) / transformed_point(2, 0), f * transformed_point(1, 0) / transformed_point(2, 0));

    //access depth image
    Mat depth_gradient(depth_image.rows, depth_image.cols, CV_8UC3, Scalar(0, 0, 0));
    Mat depth_gradient_x = depth_gradient.clone();
    Mat depth_gradient_y = depth_gradient.clone();

    Scharr(depth_image, depth_gradient_x, -1, 1, 0, 1 / sqrt(2));
    Scharr(depth_image, depth_gradient_y, -1, 0, 1, 1 / sqrt(2));

    for (int i = 0; i < depth_gradient.rows; i++)
    {
        for (int j = 0; j < depth_gradient.cols; j++)
        {
            depth_gradient.at<Vec3b>(i, j)[0] = sqrt(pow(depth_gradient_x.at<Vec3b>(i, j)[0], 2) + pow(depth_gradient_y.at<Vec3b>(i, j)[0], 2));

            depth_gradient.at<Vec3b>(i, j)[1] = sqrt(pow(depth_gradient_x.at<Vec3b>(i, j)[1], 2) + pow(depth_gradient_y.at<Vec3b>(i, j)[1], 2));

            depth_gradient.at<Vec3b>(i, j)[2] = sqrt(pow(depth_gradient_x.at<Vec3b>(i, j)[2], 2) + pow(depth_gradient_y.at<Vec3b>(i, j)[2], 2));
        }
    }

    Eigen::Matrix<double, 1, 3> depth_jacobian;
    Eigen::Matrix<double, 3, 4> pi_jacobian;

    depth_jacobian(0, 0) = depth_gradient.at<Vec3b>(fx * transformed_point(0, 0) / transformed_point(2, 0), fy * transformed_point(1, 0) / transformed_point(2, 0))[0];
    depth_jacobian(1, 0) = depth_gradient.at<Vec3b>(fx * transformed_point(0, 0) / transformed_point(2, 0), fy * transformed_point(1, 0) / transformed_point(2, 0))[1];
    depth_jacobian(2, 0) = depth_gradient.at<Vec3b>(fx * transformed_point(0, 0) / transformed_point(2, 0), fy * transformed_point(1, 0) / transformed_point(2, 0))[2];

    pi_jacobian << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1;
    pi_jacobian(0, 0) = fx / current_point(2, 0);
    pi_jacobian(0, 2) = -fx * current_point(0, 0) / pow(current_point(2, 0), 2);
    pi_jacobian(1, 1) = fy / current_point(2, 0);
    pi_jacobian(0, 2) = -fx * current_point(0, 0) / pow(current_point(2, 0), 2);

    *jacobian = depth_jacobian * pi_jacobian;
}

class ComputeResidualFunction : public ceres::SizedCostFunction<1, 1>
{
public:
    virtual bool Evaluate(Eigen::Matrix<double, 4, 1> *transformed_point, Eigen::Matrix<double, 4, 1> *current_point, double *residuals, Eigen::Matrix<double, 1, 4> **jacobians) const
    {
        ComputeDepthIntoProjectionAndJacobian(transformed_point[0], current_point[0], residuals, jacobians[0]);
        return true;
    }
};

// This returns the residual
struct Affine2DWithDistortion
{
    Affine2DWithDistortion(const double x_in, const double y_in, const double z_in)
    {
        x_ = x_in;
        y_ = y_in;
        z_ = z_in;
        compute_residual.reset(
            new ceres::CostFunctionToFunctor<1, 1, 1>(new ComputeResidualFunction));
    }

    template <typename T>
    bool operator()(const T *yaw,
                    const T *pitch,
                    const T *roll,
                    const T *tx,
                    const T *ty,
                    const T *tz,
                    T *residual) const
    {

        Eigen::Matrix<T, 4, 4> tcm;
        Eigen::Matrix<T, 4, 1> transformed_point;
        Eigen::Matrix<T, 4, 1> current_point;

        const T tcm(0, 0) = cos(yaw[0]) * cos(pitch[0]);
        const T tcm(0, 1) = cos(yaw[0]) * sin(pitch[0]) * sin(roll[0]) - sin(yaw[0]) * cos(roll[0]);
        const T tcm(0, 2) = cos(yaw[0]) * sin(pitch[0]) * cos(roll[0]) + sin(yaw[0]) * sin(roll[0]);
        const T tcm(1, 0) = sin(yaw[0]) * cos(pitch[0]);
        const T tcm(1, 1) = sin(yaw[0]) * sin(pitch[0]) * sin(roll[0]) + cos(yaw[0]) * cos(roll[0]);
        const T tcm(1, 2) = sin(yaw[0]) * sin(pitch[0]) * cos(roll[0]) - cos(yaw[0]) * sin(roll[0]);
        const T tcm(2, 0) = -1 * sin(pitch[0]);
        const T tcm(2, 1) = cos(pitch[0]) * sin(roll[0]);
        const T tcm(2, 2) = cos(pitch[0]) * cos(roll[0]);

        const T tcm(0, 3) = tx[0];
        const T tcm(1, 3) = ty[0];
        const T tcm(2, 3) = tz[0];

        const T tcm(3, 0) = T(0.0);
        const T tcm(3, 1) = T(0.0);
        const T tcm(3, 2) = T(0.0);
        const T tcm(3, 3) = T(1.0);

        //remember to convert tksi to eigen

        T error;

        current_point(0, 0) = T(x_);
        current_point(1, 0) = T(y_);
        current_point(2, 0) = T(z_);
        current_point(3, 0) = T(1.0);

        transformed_point = tksi * tcm * current_point;

        double d_pi;
        (*compute_residual)(&transformed_point, &current_point, &d_pi);

        error = transformed_point(2, 0) - T(d_pi);

        // We need to minimize this residual function after taking its huber norm, via calculating J=0
        residual[0] += pow(error, 2);

        return true;
    }

    double x_;
    double y_;
    double z_;
    std::unique_ptr<ceres::CostFunctionToFunctor<1, 1, 1>> compute_residual;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "loc");

    PointCloudConverter pcc;
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr pointcloud = pcc.point_cloud;
    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;

    Problem problem;

    //access pointcloud here
    for (int i = 0; i < pointcloud->points.size(); i++)
    {
        CostFunction *cost_function = new AutoDiffCostFunction<calculateResidual, 1, 1, 1, 1, 1, 1, 1>(new calculateResidual(pointcloud->points[i].x, pointcloud->points[i].y, pointcloud->points[i].z));
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

    ros::spin();
    return 0;
}