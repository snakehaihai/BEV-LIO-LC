#include <string>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include "BEVprojector.h"
#include <common_lib.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "REM.hpp"
#include <chrono>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <opencv2/xfeatures2d.hpp>

#pragma once

class BEVFeatureManager {
  public:
    BEVFeatureManager(ros::NodeHandle& nh, std::shared_ptr<BEVProjector> projector);

    void detectBEVFeatures(BEVFrame& frame);

    double resolution() const { return projector_->resolution(); }

    std::vector<cv::DMatch> matchFeatures(const BEVFrame& frame,const BEVFrame& frame_prev,cv::Mat& img_matches);

    const std::vector<cv::DMatch>& matches() const { return matches_; }

    void getFAST(BEVFrame& frame, bool draw_feature);
    
    void getORB(BEVFrame& frame, bool draw_feature);

    void getSIFT(BEVFrame& frame, bool draw_feature);

    void getSURF(BEVFrame& frame, bool draw_feature);

  private:
    void loadmodel();

    void loadParameters(ros::NodeHandle& nh);

    torch::Tensor cvMatToTensor(const cv::Mat& img);

    std::shared_ptr<BEVProjector> projector_;

    int visualize_en_;
    int draw_keypoints_;

    double min_range_;
    double max_range_;

    double ransac_threshold_;

    string matcher_type_;
    string match_mode_;
    float downsample_ratio_;
    int down_sample_matches_;

    std::vector<cv::DMatch> matches_;

    int fast_threshold_;
    float ratio_thresh_;
    // model
    string model_path_;
    torch::jit::Module model_;

    cv::Ptr<cv::FastFeatureDetector> fast_detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
    c10::cuda::CUDAStream stream_;

    cv::Ptr<cv::ORB> orb_extractor_;
    cv::Ptr<cv::xfeatures2d::SURF> surf_extractor_;    // SURF from xfeatures2d
    cv::Ptr<cv::xfeatures2d::SIFT> sift_extractor_; 
};