#include <common_lib.h>
#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include "bev_feature.h"
#include "BEVprojector.h"

class ImageProcessor {
  public:
    ImageProcessor(ros::NodeHandle nh, std::shared_ptr<BEVProjector> projector, std::shared_ptr<BEVFeatureManager> manager);

    void createImages(BEVFrame& frame);

  private:
    void loadParameters(ros::NodeHandle nh);
    void filterBrightness(cv::Mat& img);
    std::shared_ptr<BEVProjector> projector_;

    bool brightness_filter_;
    bool blur_;

    bool dyn_scale_;
};