#ifndef __LIDAR_SLAM_SYSTEM_H__
#define __LIDAR_SLAM_SYSTEM_H__
#include <iostream>
#include <ostream>
#include <filesystem>
#include <vector>
#include <deque>
#include <json.hpp>
#include <pcl/io/ply_io.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

using namespace nlohmann;

namespace LIDAR_SLAM
{
    class Camera
    {
    public:
        using Ptr = std::shared_ptr<Camera>;
        Camera();
        ~Camera();
        void setCameraInfo(cv::Mat cameraMatrix, cv::Mat distCoeffs)
        {
            this->cameraMatrix_ = cameraMatrix;
            this->distCoeffs_ = distCoeffs;
        }
        void setCameraInfo(cv::Mat cameraMatrix);
        {
            this->cameraMatrix_ = cameraMatrix;
            this->distCoeffs_ = cv::Mat::zeros(4, 1, CV_64F);
        }
        void setCameraImg(cv::Mat img)
        {
            this->img_raw_ = img;
        }
        void setUndistortedImg(cv::Mat img);
        {
            this->img_raw_ = img;
            this->undistorted_img_ = img;
        }
        void undistortImage()
        {
            cv::undistort(this->img_raw_, this->undistorted_img_, this->cameraMatrix_, this->distCoeffs_);
        }
        cv::Mat getCameraImg()
        {
            return this->img_raw_;
        }
        cv::Mat cameraMatrix()
        {
            return this->cameraMatrix_;
        }

    private:
        cv::Mat img_raw_, undistorted_img_;
        cv::Mat cameraMatrix_, distCoeffs_;
        int width_, height_;
        
    }

    class Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Ptr = std::shared_ptr<Frame>;
        Frame();
        ~Frame();

    private:
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;
        Eigen::Affine3f pose_;
        double timestamp_;
        std::vector<Camera::Ptr> camera_;
        std::vector<Eigen::Affine3f> camera_trans_;

    };

    class System
    {
    public:
        // File type
        enum FileType
        {
            TEXT_FILE = 0,
            BINARY_FILE = 1,
        };

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        std::vector<Frame::Ptr> frames_;
    };

} // namespace LIDAR_SLAM
#endif