#include "image_processing.h"
#include <stdexcept>

ImageProcessor::ImageProcessor(ros::NodeHandle nh, std::shared_ptr<BEVProjector> projector,std::shared_ptr<BEVFeatureManager> manager) : projector_(projector) {
    try {
        loadParameters(nh);
    } catch (const std::runtime_error& e) {
        ROS_ERROR_STREAM(e.what());
        exit(1);
    }
};

void ImageProcessor::loadParameters(ros::NodeHandle nh) {

    nh.param<bool>("image/brightness_filter", brightness_filter_, true);
    nh.param<bool>("image/blur", blur_, true);
    nh.param<bool>("image/dyn_scale", dyn_scale_, false);
}

void ImageProcessor::filterBrightness(cv::Mat& img) {
    // Create brightness map
    cv::Mat brightness;
    cv::blur(img, brightness, cv::Size(10,10));
    brightness += 1;
    // Normalize and scale image
    cv::Mat normalized_img = (140.*img / brightness); 
    img = normalized_img;
}

void ImageProcessor::createImages(BEVFrame& frame) {
    projector_->getBEV(frame); 

    if (brightness_filter_) {
        filterBrightness(frame.img_dense);
    }
    
    if (blur_) {
        cv::Mat img_blur;
        cv::GaussianBlur(frame.img_dense, img_blur, cv::Size(3,3), 0);
        frame.img_dense = img_blur;
    }

    if (dyn_scale_){
        double max_val;
        cv::minMaxLoc(frame.img_dense, nullptr, &max_val);
        if (max_val > 1) {
            frame.img_dense.convertTo(frame.img_dense, CV_32FC1, 255.0 / max_val);
        }
    }
    else{// 超过255设置为255
        cv::threshold(frame.img_dense, frame.img_dense, 255, 255, cv::THRESH_TRUNC);
    }

    // Convert to 8 bit for visualization
    frame.img_dense.convertTo(frame.img_photo_u8, CV_8UC1, 1);
}
