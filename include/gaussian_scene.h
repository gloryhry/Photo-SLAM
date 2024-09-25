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

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <tuple>
#include <filesystem>

#include "types.h"
#include "camera.h"
#include "point3d.h"
#include "point2d.h"
#include "gaussian_parameters.h"
#include "gaussian_model.h"
#include "gaussian_keyframe.h"

class GaussianScene
{
public:
    /**
     * 构造函数
     * @param args 高斯模型参数
     * @param load_iteration 加载的迭代次数
     * @param shuffle 是否打乱
     * @param resolution_scales 分辨率缩放比例
     */
    GaussianScene(
        GaussianModelParams& args,
        int load_iteration = 0,
        bool shuffle = true,
        std::vector<float> resolution_scales = {1.0f});

public:
    /**
     * 添加相机
     * @param camera 相机对象
     */
    void addCamera(Camera& camera);

    /**
     * 获取相机
     * @param cameraId 相机ID
     * @return 相机对象
     */
    Camera& getCamera(camera_id_t cameraId);

    /**
     * 添加关键帧
     * @param new_kf 新的关键帧
     * @param shuffled 是否被打乱
     */
    void addKeyframe(std::shared_ptr<GaussianKeyframe> new_kf, bool* shuffled);

    /**
     * 获取关键帧
     * @param fid 帧ID
     * @return 关键帧对象
     */
    std::shared_ptr<GaussianKeyframe> getKeyframe(std::size_t fid);

    /**
     * 获取所有关键帧
     * @return 关键帧映射
     */
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes();

    /**
     * 获取所有关键帧的副本
     * @return 关键帧映射的副本
     */
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> getAllKeyframes();

    /**
     * 缓存3D点
     * @param point3D_id 3D点ID
     * @param point3d 3D点对象
     */
    void cachePoint3D(point3D_id_t point3D_id, Point3D& point3d);

    /**
     * 获取缓存的3D点
     * @param point3DId 3D点ID
     * @return 3D点对象
     */
    Point3D& getPoint3D(point3D_id_t point3DId);

    /**
     * 清除缓存的3D点
     */
    void clearCachedPoint3D();

    /**
     * 应用缩放变换
     * @param s 缩放因子
     * @param T 变换矩阵
     */
    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    /**
     * 获取Nerfpp的归一化信息
     * @return 归一化向量和半径
     */
    std::tuple<Eigen::Vector3f, float> getNerfppNorm();

    /**
     * 分割训练和测试关键帧
     * @param test_ratio 测试集比例
     * @return 训练集和测试集的关键帧映射
     */
    std::tuple<std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>,
               std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>>
        splitTrainAndTestKeyframes(const float test_ratio);

public:
    float cameras_extent_; ///< 场景信息中的Nerf归一化半径 scene_info.nerf_normalization["radius"]

    int loaded_iter_;

    std::map<camera_id_t, Camera> cameras_;
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes_;
    std::map<point3D_id_t, Point3D> cached_point_cloud_;

protected:
    std::mutex mutex_kfs_;
};
