/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <string>
#include <filesystem>

class GaussianModelParams
{
public:
    GaussianModelParams(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false);

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;
};

class GaussianPipelineParams
{
public:
    GaussianPipelineParams(
        bool convert_SHs = false,
        bool compute_cov3D = false);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
};

class GaussianOptimizationParams
{
public:
    GaussianOptimizationParams(
        int iterations = 30'000,
        float position_lr_init = 0.00016f,
        float position_lr_final = 0.0000016f,
        float position_lr_delay_mult = 0.01f,
        int position_lr_max_steps = 30'000,
        float feature_lr = 0.0025f,
        float opacity_lr = 0.05f,
        float scaling_lr = 0.005f,
        float rotation_lr = 0.001f,
        float percent_dense = 0.01f,
        float lambda_dssim = 0.2f,
        int densification_interval = 100,
        int opacity_reset_interval = 3000,
        int densify_from_iter = 500,
        int densify_until_iter = 15'000,
        float densify_grad_threshold = 0.0002f);

public:
    int iterations_; // 迭代次数
    float position_lr_init_; // 初始位置学习率
    float position_lr_final_; // 最终位置学习率
    float position_lr_delay_mult_; // 位置学习率延迟乘数
    int position_lr_max_steps_; // 位置学习率最大步数
    float feature_lr_; // 特征学习率
    float opacity_lr_; // 不透明度学习率
    float scaling_lr_; // 缩放学习率
    float rotation_lr_; // 旋转学习率
    float percent_dense_; // 密集百分比
    float lambda_dssim_; // DSSIM损失权重
    int densification_interval_; // 密集化间隔
    int opacity_reset_interval_; // 不透明度重置间隔
    int densify_from_iter_; // 从第几次迭代开始密集化
    int densify_until_iter_; // 到第几次迭代结束密集化
    float densify_grad_threshold_; // 密集化梯度阈值
};
