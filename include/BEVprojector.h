#include <common_lib.h>
#include "ros/ros.h"

#pragma once

class BEVProjector {
  public:
    BEVProjector(ros::NodeHandle nh);

    void voxelDownSample(BEVFrame& frame);

    double resolution() const {return resolution_;}

    V3D turn_pixel_to_point(const V3D& pixel_uv);
    
    void getBEV(BEVFrame& frame);

    int rows() const {return y_num_;}
    int cols() const {return x_num_;}

  private:
    void loadParameters(ros::NodeHandle nh);

    double resolution_; // 分辨率，单位为米
    double max_x_;      // 最大 x 坐标
    double min_x_;      // 最小 x 坐标
    double max_y_;      // 最大 y 坐标
    double min_y_;      // 最小 y 坐标

    double voxel_size_; // 体素大小
    bool downsample_;
    bool normalize_to_255_;
    bool use_dense_;
    int x_min_ind_;
    int x_max_ind_;
    int y_min_ind_;
    int y_max_ind_;

    int x_num_;
    int y_num_;

};