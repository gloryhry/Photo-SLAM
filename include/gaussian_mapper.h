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

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <map>
#include <random>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

#include <jsoncpp/json/json.h>

// #include "ORB-SLAM3/include/System.h"
// Sophus库
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
// TODO: 需要替换为Lidar_SLAM相关头文件
#include "Lidar_SLAM/include/System.h"

#include "operate_points.h"
#include "stereo_vision.h"
#include "tensor_utils.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"
#include "gaussian_trainer.h"

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());
struct UndistortParams // 去畸变参数结构体
{
    UndistortParams(
        const cv::Size& old_size, // 旧图像尺寸
        cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f)) // 畸变系数
        : old_size_(old_size)
    {
        dist_coeff.copyTo(dist_coeff_); // 复制畸变系数
    }

    cv::Size old_size_; // 旧图像尺寸
    cv::Mat dist_coeff_; // 畸变系数
};

enum SystemSensorType // 系统传感器类型枚举
{
    INVALID = 0, // 无效
    MONOCULAR = 1, // 单目
    STEREO = 2, // 双目
    RGBD = 3 // RGBD
};

struct VariableParameters // 变量参数结构体
{
    float position_lr_init; // 初始位置学习率
    float feature_lr; // 特征学习率
    float opacity_lr; // 不透明度学习率
    float scaling_lr; // 缩放学习率
    float rotation_lr; // 旋转学习率
    float percent_dense; // 密集百分比
    float lambda_dssim; // DSSIM损失权重
    int opacity_reset_interval; // 不透明度重置间隔
    float densify_grad_th; // 密集化梯度阈值
    int densify_interval; // 密集化间隔
    int new_kf_times_of_use; // 新关键帧使用次数
    int stable_num_iter_existence; // 稳定存在的迭代次数（回环闭合校正）

    bool keep_training; // 是否继续训练
    bool do_gaus_pyramid_training; // 是否进行高斯金字塔训练
    bool do_inactive_geo_densify; // 是否进行非活动几何体密集化
};

class GaussianMapper
{
public:
    GaussianMapper(
        std::shared_ptr<Lidar_SLAM::System> pSLAM, 
        std::filesystem::path gaussian_config_file_path, 
        std::filesystem::path result_dir = "", 
        int seed = 0, 
        torch::DeviceType device_type = torch::kCUDA);

    void readConfigFromFile(std::filesystem::path cfg_path); // 从文件读取配置

    void run(); // 运行
    void trainColmap(); // 训练Colmap
    void trainForOneIteration(); // 进行一次迭代训练

    bool isStopped(); // 是否已停止
    void signalStop(const bool going_to_stop = true); // 发送停止信号

    cv::Mat renderFromPose(
        const Sophus::SE3f &Tcw, // 相机位姿
        const int width, // 图像宽度
        const int height, // 图像高度
        const bool main_vision = false); // 是否为主视觉

    int getIteration(); // 获取迭代次数
    void increaseIteration(const int inc = 1); // 增加迭代次数

    float positionLearningRateInit(); // 获取初始位置学习率
    float featureLearningRate(); // 获取特征学习率
    float opacityLearningRate(); // 获取不透明度学习率
    float scalingLearningRate(); // 获取缩放学习率
    float rotationLearningRate(); // 获取旋转学习率
    float percentDense(); // 获取密集百分比
    float lambdaDssim(); // 获取DSSIM损失权重
    int opacityResetInterval(); // 获取不透明度重置间隔
    float densifyGradThreshold(); // 获取密集化梯度阈值
    int densifyInterval(); // 获取密集化间隔
    int newKeyframeTimesOfUse(); // 获取新关键帧使用次数
    int stableNumIterExistence(); // 获取稳定存在的迭代次数
    bool isKeepingTraining(); // 是否继续训练
    bool isdoingGausPyramidTraining(); // 是否进行高斯金字塔训练
    bool isdoingInactiveGeoDensify(); // 是否进行非活动几何体密集化

    void setPositionLearningRateInit(const float lr); // 设置初始位置学习率
    void setFeatureLearningRate(const float lr); // 设置特征学习率
    void setOpacityLearningRate(const float lr); // 设置不透明度学习率
    void setScalingLearningRate(const float lr); // 设置缩放学习率
    void setRotationLearningRate(const float lr); // 设置旋转学习率
    void setPercentDense(const float percent_dense); // 设置密集百分比
    void setLambdaDssim(const float lambda_dssim); // 设置DSSIM损失权重
    void setOpacityResetInterval(const int interval); // 设置不透明度重置间隔
    void setDensifyGradThreshold(const float th); // 设置密集化梯度阈值
    void setDensifyInterval(const int interval); // 设置密集化间隔
    void setNewKeyframeTimesOfUse(const int times); // 设置新关键帧使用次数
    void setStableNumIterExistence(const int niter); // 设置稳定存在的迭代次数
    void setKeepTraining(const bool keep); // 设置是否继续训练
    void setDoGausPyramidTraining(const bool gaus_pyramid); // 设置是否进行高斯金字塔训练
    void setDoInactiveGeoDensify(const bool inactive_geo_densify); // 设置是否进行非活动几何体密集化

    VariableParameters getVaribleParameters(); // 获取变量参数
    void setVaribleParameters(const VariableParameters &params); // 设置变量参数

    GaussianModelParams& getGaussianModelParams() { return this->model_params_; } // 获取高斯模型参数
    void setColmapDataPath(std::filesystem::path colmap_path) { this->model_params_.source_path_ = colmap_path; } // 设置Colmap数据路径
    void setSensorType(SystemSensorType sensor_type) { this->sensor_type_ = sensor_type; } // 设置传感器类型

    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = ""); // 加载PLY文件

protected:
    bool hasMetInitialMappingConditions(); // 是否满足初始映射条件
    bool hasMetIncrementalMappingConditions(); // 是否满足增量映射条件

    void combineMappingOperations(); // 合并映射操作

    void handleNewKeyframe(std::tuple<unsigned long,
                                      unsigned long,
                                      Sophus::SE3f,
                                      cv::Mat,
                                      bool,
                                      cv::Mat,
                                      std::vector<float>,
                                      std::vector<float>,
                                      std::string> &kf); // 处理新关键帧
    void generateKfidRandomShuffle(); // 生成关键帧ID随机洗牌
    std::shared_ptr<GaussianKeyframe> useOneRandomSlidingWindowKeyframe(); // 使用一个随机滑动窗口关键帧
    std::shared_ptr<GaussianKeyframe> useOneRandomKeyframe(); // 使用一个随机关键帧
    void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> pkf, int times); // 增加关键帧使用次数
    void cullKeyframes(); // 剔除关键帧

    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<GaussianKeyframe> pkf); // 通过关键帧增加非活动几何体密集化

    // bool needInterruptTraining(); // 是否需要中断训练
    // void setInterruptTraining(const bool interrupt_training); // 设置中断训练

    void recordKeyframeRendered(
        torch::Tensor &rendered, // 渲染图像
        torch::Tensor &ground_truth, // 真实图像
        unsigned long kfid, // 关键帧ID
        std::filesystem::path result_img_dir, // 结果图像目录
        std::filesystem::path result_gt_dir, // 结果真实图像目录
        std::filesystem::path result_loss_dir, // 结果损失图像目录
        std::string name_suffix = ""); // 文件名后缀
    void renderAndRecordKeyframe(
        std::shared_ptr<GaussianKeyframe> pkf, // 关键帧
        float &dssim, // DSSIM损失
        float &psnr, // PSNR
        float &psnr_gs, // 高斯PSNR
        double &render_time, // 渲染时间
        std::filesystem::path result_img_dir, // 结果图像目录
        std::filesystem::path result_gt_dir, // 结果真实图像目录
        std::filesystem::path result_loss_dir, // 结果损失图像目录
        std::string name_suffix = ""); // 文件名后缀
    void renderAndRecordAllKeyframes(
        std::string name_suffix = ""); // 渲染并记录所有关键帧

    void savePly(std::filesystem::path result_dir); // 保存PLY文件
    void keyframesToJson(std::filesystem::path result_dir); // 将关键帧保存为JSON文件
    void saveModelParams(std::filesystem::path result_dir); // 保存模型参数
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = ""); // 写入关键帧使用次数

public:
    // Parameters
    std::filesystem::path config_file_path_; // 配置文件路径

    // Model
    std::shared_ptr<GaussianModel> gaussians_; // 高斯模型
    std::shared_ptr<GaussianScene> scene_; // 场景

    // SLAM system
    std::shared_ptr<Lidar_SLAM::System> pSLAM_; // SLAM系统指针

    // Settings
    torch::DeviceType device_type_; // 设备类型
    int num_gaus_pyramid_sub_levels_ = 0; // 高斯金字塔子层级数量
    std::vector<int> kf_gaus_pyramid_times_of_use_; // 关键帧高斯金字塔使用次数
    std::vector<float> kf_gaus_pyramid_factors_; // 关键帧高斯金字塔因子

    bool viewer_camera_id_set_ = false; // 查看器相机ID是否设置
    std::uint32_t viewer_camera_id_ = 0; // 查看器相机ID
    float rendered_image_viewer_scale_ = 1.0f; // 渲染图像查看器缩放比例
    float rendered_image_viewer_scale_main_ = 1.0f; // 主渲染图像查看器缩放比例

    float z_near_ = 0.01f; // 近裁剪面
    float z_far_ = 100.0f; // 远裁剪面

    // Data
    bool kfid_shuffled_ = false; 
    std::map<camera_id_t, torch::Tensor> undistort_mask_; // 去畸变
    std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_; // 查看器主去畸变
    std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_; // 查看器子去畸变

protected:
    // Parameters
    GaussianModelParams model_params_; // 高斯模型参数
    GaussianOptimizationParams opt_params_; // 优化参数
    GaussianPipelineParams pipe_params_; // 管道参数

    // Data
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> viewpoint_sliding_window_; // 视点滑动窗口
    std::vector<std::size_t> kfid_shuffle_; // 关键帧ID随机洗牌
    std::size_t kfid_shuffle_idx_ = 0; // 关键帧ID随机洗牌索引
    std::map<std::size_t, int> kfs_used_times_; // 关键帧使用次数

    // Status
    bool initial_mapped_; // 是否完成初始映射
    bool interrupt_training_; // 是否中断训练
    bool stopped_; // 是否停止
    int iteration_; // 迭代次数
    float ema_loss_for_log_; // 用于日志的指数移动平均损失
    bool SLAM_ended_; // SLAM是否结束
    bool loop_closure_iteration_; // 是否为回环闭合迭代
    bool keep_training_ = false; // 是否继续训练
    int default_sh_ = 0; // 默认SH系数数量

    // Settings
    SystemSensorType sensor_type_; // 传感器类型

    float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0; // 单目模式下非活动几何体密集化最大像素距离
    float stereo_baseline_length_ = 0.0f; // 立体基线长度
    int stereo_min_disparity_ = 0; // 立体最小视差
    int stereo_num_disparity_ = 128; // 立体视差数量
    cv::Mat stereo_Q_; // 立体Q矩阵
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_; // 立体SGM算法
    float RGBD_min_depth_ = 0.0f; // RGBD最小深度
    float RGBD_max_depth_ = 100.0f; // RGBD最大深度

    bool inactive_geo_densify_ = true; // 是否进行非活动几何体密集化
    int depth_cached_ = 0; // 缓存的深度图数量
    int max_depth_cached_ = 1; // 最大缓存的深度图数量
    torch::Tensor depth_cache_points_; // 深度缓存点
    torch::Tensor depth_cache_colors_; // 深度缓存颜色

    unsigned long min_num_initial_map_kfs_; // 初始映射所需的最小关键帧数量
    torch::Tensor background_; // 背景
    float large_rot_th_; // 大旋转阈值
    float large_trans_th_; // 大平移阈值
    torch::Tensor override_color_; // 覆盖颜色

    int new_keyframe_times_of_use_; // 新关键帧使用次数
    int local_BA_increased_times_of_use_; // 局部BA增加的使用次数
    int loop_closure_increased_times_of_use_; // 回环闭合增加的使用次数

    bool cull_keyframes_; // 是否剔除关键帧
    int stable_num_iter_existence_; // 稳定存在的迭代次数

    bool do_gaus_pyramid_training_; // 是否进行高斯金字塔训练

    std::filesystem::path result_dir_; // 结果目录
    int keyframe_record_interval_; // 关键帧记录间隔
    int all_keyframes_record_interval_; // 所有关键帧记录间隔
    bool record_rendered_image_; // 是否记录渲染图像
    bool record_ground_truth_image_; // 是否记录真实图像
    bool record_loss_image_; // 是否记录损失图像

    int training_report_interval_; // 训练报告间隔
    bool record_loop_ply_; // 是否记录回环PLY文件

    int prune_big_point_after_iter_; // 在迭代后修剪大点的次数
    float densify_min_opacity_ = 20; // 密集化最小不透明度

    // Tools
    std::random_device rd_; // 随机设备

    // Mutex
    std::mutex mutex_status_; // 状态互斥锁
    std::mutex mutex_settings_; // 设置互斥锁
    std::mutex mutex_render_; // 渲染互斥锁（模型应为只读）
};