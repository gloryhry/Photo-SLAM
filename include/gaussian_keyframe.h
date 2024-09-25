/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <memory>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "types.h"
#include "camera.h"
#include "point2d.h"
#include "general_utils.h"
#include "graphics_utils.h"
#include "tensor_utils.h"

class GaussianKeyframe
{
public:
    GaussianKeyframe() {}

    GaussianKeyframe(std::size_t fid, int creation_iter = 0)
        : fid_(fid), creation_iter_(creation_iter) {}

    /**
     * 设置姿态（四元数和位移向量）
     * @param qw 四元数的w分量
     * @param qx 四元数的x分量
     * @param qy 四元数的y分量
     * @param qz 四元数的z分量
     * @param tx 位移向量的x分量
     * @param ty 位移向量的y分量
     * @param tz 位移向量的z分量
     */
    void setPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz);
    
    /**
     * 设置姿态（四元数和位移向量）
     * @param q 四元数
     * @param t 位移向量
     */
    void setPose(
        const Eigen::Quaterniond& q,
        const Eigen::Vector3d& t);

    /**
     * 获取姿态（SE3变换矩阵）
     * @return SE3变换矩阵
     */
    Sophus::SE3d getPose();

    /**
     * 获取姿态（SE3变换矩阵，单精度）
     * @return SE3变换矩阵（单精度）
     */
    Sophus::SE3f getPosef();

    /**
     * 设置相机参数
     * @param camera 相机对象
     */
    void setCameraParams(const Camera& camera);

    /**
     * 设置2D点集合
     * @param points2D 2D点集合
     */
    void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);

    /**
     * 为2D点设置对应的3D点索引
     * @param point2D_idx 2D点索引
     * @param point3D_id 3D点ID
     */
    void setPoint3DIdxForPoint2D(
        const point2D_idx_t point2D_idx,
        const point3D_id_t point3D_id);

    /**
     * 计算变换张量
     */
    void computeTransformTensors();

    /**
     * 获取世界坐标系到视图坐标系的变换矩阵
     * @param trans 平移向量
     * @param scale 缩放因子
     * @return 变换矩阵
     */
    Eigen::Matrix4f getWorld2View2(
        const Eigen::Vector3f& trans = {0.0f, 0.0f, 0.0f},
        float scale = 1.0f);

    /**
     * 获取投影矩阵
     * @param znear 近裁剪面
     * @param zfar 远裁剪面
     * @param fovX 水平视场角
     * @param fovY 垂直视场角
     * @param device_type 设备类型（默认CUDA）
     * @return 投影矩阵
     */
    torch::Tensor getProjectionMatrix(
        float znear,
        float zfar,
        float fovX,
        float fovY,
        torch::DeviceType device_type = torch::kCUDA);

    /**
     * 获取当前高斯金字塔层级
     * @return 当前高斯金字塔层级
     */
    int getCurrentGausPyramidLevel();

public:
    std::size_t fid_; // 帧ID
    int creation_iter_; // 创建迭代次数
    int remaining_times_of_use_ = 0; // 剩余使用次数

    bool set_camera_ = false; // 是否设置了相机参数

    camera_id_t camera_id_; // 相机ID
    int camera_model_id_ = 0; // 相机模型ID

    std::string img_filename_; // 图像文件名
    cv::Mat img_undist_, img_auxiliary_undist_; // 去畸变图像和辅助去畸变图像
    torch::Tensor original_image_; ///< 原始图像
    int image_width_; ///< 图像宽度
    int image_height_; ///< 图像高度

    int num_gaus_pyramid_sub_levels_; // 高斯金字塔子层数
    std::vector<int> gaus_pyramid_times_of_use_; // 高斯金字塔使用次数
    std::vector<std::size_t> gaus_pyramid_width_; ///< 高斯金字塔图像宽度
    std::vector<std::size_t> gaus_pyramid_height_; ///< 高斯金字塔图像高度
    std::vector<torch::Tensor> gaus_pyramid_original_image_; ///< 高斯金字塔原始图像
    // Tensor gt_alpha_mask_;

    std::vector<float> intr_; ///< 相机内参

    float FoVx_; ///< 相机水平视场角 相机内参
    float FoVy_; ///< 相机垂直视场角 相机内参

    bool set_pose_ = false; // 是否设置了姿态
    bool set_projection_matrix_ = false; // 是否设置了投影矩阵

    Eigen::Quaterniond R_quaternion_; ///< 旋转四元数（外参）
    Eigen::Vector3d t_; ///< 平移向量（外参）
    Sophus::SE3d Tcw_; ///< 相机到世界坐标系的变换矩阵（外参）

    torch::Tensor R_tensor_; ///< 旋转矩阵张量（外参）
    torch::Tensor t_tensor_; ///< 平移向量张量（外参）

    float zfar_ = 100.0f; // 远裁剪面
    float znear_ = 0.01f; // 近裁剪面

    Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f}; // 平移向量
    float scale_ = 1.0f; // 缩放因子

    torch::Tensor world_view_transform_; ///< 世界到视图的变换矩阵 tensors
    torch::Tensor projection_matrix_; ///< 投影矩阵 tensors
    torch::Tensor full_proj_transform_; ///< 完整投影变换矩阵 tensors
    torch::Tensor camera_center_; ///< 相机中心 tensors

    std::vector<Point2D> points2D_; // 2D点集合
    std::vector<float> kps_pixel_; // 像素关键点
    std::vector<float> kps_point_local_; // 局部坐标关键点

    bool done_inactive_geo_densify_ = false; // 是否完成了非活动几何密集化
};;
