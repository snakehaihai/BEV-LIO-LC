#include "bev_feature.h"

BEVFeatureManager::BEVFeatureManager(ros::NodeHandle& nh, std::shared_ptr<BEVProjector> projector) : projector_(projector) ,stream_(c10::cuda::getStreamFromPool()) {
    loadParameters(nh);
    
    fast_detector_ = cv::FastFeatureDetector::create(fast_threshold_, true, cv::FastFeatureDetector::TYPE_9_16);
    if (matcher_type_ == "BF")
    {
        matcher_ = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);
    }
    else if (matcher_type_ == "BruteForce-Hamming")
    {
        matcher_ = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING);
    }
    else
    {
        ROS_ERROR("Invalid matcher type, setting to BF");
        matcher_ = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);
    }
    // initialize feature extractors
    orb_extractor_ = cv::ORB::create(
        500,              // 最大特征点数量 (nfeatures)
        1.00f,            // 尺度因子 (scale_factor)
        1,                // 金字塔层数 (n_levels)
        31,               // 边缘阈值 (edge_threshold)
        0,                // 初始层级 (first_level)
        2,                // WTA_K
        cv::ORB::HARRIS_SCORE, // 使用 Harris 角点评分 (score_type)
        31,               // 补丁大小 (patch_size)
        35                // 快速阈值 (fast_threshold)
    );

    // SURF 初始化（需 xfeatures2d）
    surf_extractor_ = cv::xfeatures2d::SURF::create(
        1000,             // Hessian 阈值（示例值，可调整）
        4,                 // 金字塔层数
        3,                // 扩展描述子标志
        true,             // 使用 U-SURF 加速
        false             // 不计算方向（若需方向设为 true）
    );

    // SIFT 初始化（主仓库或 xfeatures2d）
    sift_extractor_ = cv::xfeatures2d::SIFT::create(
        500,              // 最大特征点数量
        3,                // 金字塔层数
        0.04,             // 对比度阈值
        10.0,             // 边缘阈值
        1.6               // sigma
    );

    loadmodel();
}

void BEVFeatureManager::loadParameters(ros::NodeHandle& nh) {
    nh.param<int>("visualize_en_", visualize_en_, 0);
    nh.param<int>("draw_keypoints_", draw_keypoints_, 1);
    nh.param<float>("downsample_ratio_", downsample_ratio_, 0.7f);
    nh.param<int>("down_sample_matches_", down_sample_matches_, 1);
    nh.param<double>("image/ransac_threshold", ransac_threshold_, 4.0);
    nh.param<float>("image/ratio_thresh", ratio_thresh_, 0.90f);
    nh.param<int>("image/fast_threshold", fast_threshold_, 10);
    // change it to your path here
    nh.param<string>("model_path", model_path_, "/home/chx/bev-lio-lc/src/BEV_LIO/models/gpu.pt");
    nh.param<string>("image/matcher",matcher_type_,"BF");
    nh.param<string>("image/match_mode", match_mode_, "knn");
}

void BEVFeatureManager::loadmodel() {
    try {
        std::cout << "=> loading GPU checkpoint from '" << model_path_ << "'" << std::endl;

        int device_count = torch::cuda::device_count();
        std::cout << "CUDA devices available: " << device_count << std::endl;
        // load model
        model_ = torch::jit::load(model_path_);
        model_.to(torch::Device(torch::kCUDA, 0));
        model_.eval();
        
        stream_ = c10::cuda::getStreamFromPool();
        c10::cuda::setCurrentCUDAStream(stream_);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the checkpoint: " << e.what() << std::endl;
        ROS_ERROR("Error loading the model");
    }
}

torch::Tensor BEVFeatureManager::cvMatToTensor(const cv::Mat& img) {
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 256);
    if (img_float.channels() == 1) {
        cv::Mat img_rgb;
        cv::cvtColor(img_float, img_rgb, cv::COLOR_GRAY2RGB); // convert gray to RGB
        img_float = img_rgb;
    } else {
        // convert BGR to RGB
        cv::cvtColor(img_float, img_float, cv::COLOR_BGR2RGB);
    }
    // convert Mat to Tensor, add batch channel，it's {1, H, W, C} now
    torch::Tensor tensor_image = torch::from_blob(img_float.data, {1, img_float.rows, img_float.cols, 3}, torch::kFloat);
    return tensor_image.permute({0, 3, 1, 2});
}

void BEVFeatureManager::detectBEVFeatures(BEVFrame& frame) {
    c10::cuda::setCurrentCUDAStream(stream_);

    // convert cv::Mat to torch::Tensor
    torch::Tensor img_tensor = cvMatToTensor(frame.img_photo_u8);
    torch::NoGradGuard no_grad;

    // create c10::IValue vector to hold the input tensors
    std::vector<at::Tensor> inputs = { img_tensor }; 

    std::vector<c10::IValue> ivalue_inputs;
    ivalue_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
         ivalue_inputs.push_back(input.to(torch::Device(torch::kCUDA, 0)));
    }

    auto tuple_output = model_.forward(ivalue_inputs).toTuple();

    // torch::Tensor output1 = tuple_output->elements()[0].toTensor();
    torch::Tensor output2 = tuple_output->elements()[1].toTensor(); // local_feats
    torch::Tensor output3 = tuple_output->elements()[2].toTensor(); // global_desc
    
    // Asynchronously copy local features to CPU on the non-default stream
    {
        c10::cuda::CUDAStreamGuard guard(stream_);
        frame.local_feats = output2.to(at::kCPU, false); // async copy
        // frame.global_desc = output3.to(at::kFloat, false); // async copy
    }

    // Synchronize the non-default stream with the default stream
    at::cuda::CUDAEvent event;
    event.record(stream_);
    event.synchronize();

    // Flatten the global descriptor and copy it to the CPU
    output3 = output3.view(-1); // conver to 8192
    auto output3_data = output3.detach().cpu(); // global_desc: numpy torch.Size([1, 8192])

    frame.global_desc.resize(output3_data.numel()); // Resize to 8192
    std::memcpy(frame.global_desc.data(), output3_data.data_ptr<float>(), output3_data.numel() * sizeof(float));
}

void BEVFeatureManager::getFAST(BEVFrame& frame, bool draw_feature) {
    fast_detector_->detect(frame.img_photo_u8, frame.keypoints);

    std::vector<torch::Tensor> descriptors;

    for (const auto& kp : frame.keypoints) {
        int u = static_cast<int>(kp.pt.x);
        int v = static_cast<int>(kp.pt.y);

        // from CWH to WHC
        torch::Tensor descriptor = frame.local_feats.index({0, torch::indexing::Slice(), v, u});
        descriptors.push_back(descriptor);
    }

    //  concat descriptors into a single tensor
    if (!descriptors.empty()) {
        int descriptor_size = descriptors[0].size(0);
        cv::Mat concatenated_descriptors(descriptors.size(), descriptor_size, CV_32F);
        for (size_t i = 0; i < descriptors.size(); ++i) {
            memcpy(concatenated_descriptors.ptr<float>(i), descriptors[i].data_ptr<float>(), descriptor_size * sizeof(float));
        }
        frame.query_descriptors = concatenated_descriptors;
    }
    
    if(draw_feature){
        cv::Mat inverted_img;
        cv::bitwise_not(frame.img_photo_u8, inverted_img);
        cv::cvtColor(inverted_img, frame.img_with_keypoints, cv::COLOR_GRAY2BGR); // gray to BGR
        // red markers
        for (const auto& kp : frame.keypoints) {
            cv::drawMarker(frame.img_with_keypoints, kp.pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 5, 1);
        }
    }
}

void BEVFeatureManager::getORB(BEVFrame& frame, bool draw_feature) {
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;

    orb_extractor_->detectAndCompute(frame.img_photo_u8, cv::noArray(), orb_keypoints, orb_descriptors);

    frame.keypoints = orb_keypoints;
    frame.query_descriptors = orb_descriptors;

    if (draw_feature) {
        cv::cvtColor(frame.img_photo_u8, frame.img_with_keypoints, cv::COLOR_GRAY2BGR); 
        for (const auto& kp : frame.keypoints) {
            cv::drawMarker(frame.img_with_keypoints, kp.pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 5, 1);
        }
    }
}

void BEVFeatureManager::getSIFT(BEVFrame& frame, bool draw_feature) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    
    sift_extractor_->detectAndCompute(frame.img_photo_u8, cv::noArray(),
                                    keypoints, descriptors);
    
    frame.keypoints = keypoints;
    frame.query_descriptors = descriptors;

    if (draw_feature) {
        cv::cvtColor(frame.img_photo_u8, frame.img_with_keypoints, cv::COLOR_GRAY2BGR);
        for (const auto& kp : frame.keypoints) {
            cv::drawMarker(frame.img_with_keypoints, kp.pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 5, 1);
        }
    }
}

void BEVFeatureManager::getSURF(BEVFrame& frame, bool draw_feature) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    
    surf_extractor_->detectAndCompute(frame.img_photo_u8, cv::noArray(), 
                                    keypoints, descriptors);
    
    frame.keypoints = keypoints;
    frame.query_descriptors = descriptors;
    
    if (draw_feature) {
        cv::cvtColor(frame.img_photo_u8, frame.img_with_keypoints, cv::COLOR_GRAY2BGR);
        for (const auto& kp : keypoints) {
            cv::drawMarker(frame.img_with_keypoints, kp.pt, cv::Scalar(0,0,255),
                         cv::MARKER_CROSS, 5, 1);
        }
    }
}

std::vector<cv::DMatch> BEVFeatureManager::matchFeatures(const BEVFrame& frame,const BEVFrame& frame_prev,cv::Mat& img_matches) {
    thread_local cv::Ptr<cv::BFMatcher> local_matcher;
    local_matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::DMatch> knn_matches_one;

    if (frame.keypoints.empty() || frame_prev.keypoints.empty()) {
        std::cerr << "No keypoints to match" << std::endl;
        return good_matches;
    }

    if (frame.query_descriptors.type() != frame_prev.query_descriptors.type()) {
        frame.query_descriptors.convertTo(frame.query_descriptors, frame_prev.query_descriptors.type());
        std::cout << "not same type" << std::endl;
    }
    
    cv::Mat descriptors1 = frame.query_descriptors.clone();
    cv::Mat descriptors2 = frame_prev.query_descriptors.clone();
    local_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    // local_matcher->knnMatch(frame.query_descriptors, frame_prev.query_descriptors,knn_matches, 2);

    // #pragma omp parallel for
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh_ * knn_matches[i][1].distance) {
            // #pragma omp critical
            good_matches.push_back(knn_matches[i][0]);
        }
        // #pragma omp critical 
        knn_matches_one.push_back(knn_matches[i][0]);
    }

    // RANSAC to find inliers
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++) {
        obj.push_back(frame.keypoints[good_matches[i].queryIdx].pt);
        scene.push_back(frame_prev.keypoints[good_matches[i].trainIdx].pt);
    }
    std::vector<uchar> inliers;
    if (obj.size() < 4 || scene.size() < 4) {
        std::cout << "Not enough points for RANSAC" << std::endl;
    }
    else{
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC, ransac_threshold_, inliers);
    }

    std::vector<cv::DMatch> ransac_matches;
    for (size_t i =0; i < inliers.size(); i++) {
        if (inliers[i]) {
            ransac_matches.push_back(good_matches[i]);
        }
    }

    std::vector<cv::DMatch> sampled_matches;

    // downsample
    if (down_sample_matches_ == 1)
    {
        int downsample_step = static_cast<int>(1.0f / downsample_ratio_);
        downsample_step = std::max(downsample_step, 1);
        for (size_t i = 0; i < ransac_matches.size(); i += downsample_step) {
            sampled_matches.push_back(ransac_matches[i]);
        }
        std::cout << "ransac_matches.size(): " << ransac_matches.size() << std::endl;
        std::cout << "sampled_matches.size(): " << sampled_matches.size() << std::endl;
    }
    else
    {
        sampled_matches = ransac_matches;
    }

    if (visualize_en_ == 1)
    {
        cv::Mat inverted_img1;
        cv::bitwise_not(frame.img_photo_u8, inverted_img1);
        cv::Mat inverted_img2;
        cv::bitwise_not(frame_prev.img_photo_u8, inverted_img2);
        cv::Mat img_bgr_1;
        cv::Mat img_bgr_2;
        cv::cvtColor(inverted_img1, img_bgr_1, CV_GRAY2RGB);
        cv::cvtColor(inverted_img2, img_bgr_2, CV_GRAY2RGB);
        cv::drawMatches(img_bgr_1, frame.keypoints,
                img_bgr_2, frame_prev.keypoints,
                sampled_matches, img_matches,
                cv::Scalar::all(-1),
                // cv::Scalar(0, 255, 0),
                cv::Scalar(0, 0, 255),
                //  cv::Scalar::all(-1),
                std::vector<char>() // use std::vector<std::vector<char>>
                // cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                );

        if (draw_keypoints_ == 1)
        {
            cv::Mat inverted_img1;
            cv::bitwise_not(frame.img_photo_u8, inverted_img1);
            cv::Mat inverted_img2;
            cv::bitwise_not(frame_prev.img_photo_u8, inverted_img2);
            cv::Mat img_bgr_1;
            cv::Mat img_bgr_2;
            cv::cvtColor(inverted_img1, img_bgr_1, CV_GRAY2RGB);
            cv::cvtColor(inverted_img2, img_bgr_2, CV_GRAY2RGB);
            cv::drawMatches(img_bgr_1, frame.keypoints,
                    img_bgr_2, frame_prev.keypoints,
                    sampled_matches, img_matches,
                    cv::Scalar::all(-1),
                    // cv::Scalar(0, 255, 0),
                    cv::Scalar(0, 0, 255),
                    //  cv::Scalar::all(-1),
                    std::vector<char>(), // use std::vector<std::vector<char>>
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                    );
        }
    }
    else
    {
        cv::Mat img_bgr_1;
        cv::Mat img_bgr_2;
        cv::cvtColor(frame.img_photo_u8, img_bgr_1, CV_GRAY2RGB);
        cv::cvtColor(frame_prev.img_photo_u8, img_bgr_2, CV_GRAY2RGB);
        cv::drawMatches(img_bgr_1, frame.keypoints,
                        img_bgr_2, frame_prev.keypoints,
                        sampled_matches, img_matches,
                        cv::Scalar::all(-1),
                        // cv::Scalar(0, 255, 0),
                        cv::Scalar(0, 0, 255),
                        //  cv::Scalar::all(-1),
                        std::vector<char>() // use std::vector<std::vector<char>>
                        // cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                        );
    }

    if (match_mode_ == "knn") {
        matches_ = knn_matches_one;
        return knn_matches_one;
    }
    else if (match_mode_ == "ransac")
    {
        matches_ = ransac_matches;
        return ransac_matches;
    }
    else if (match_mode_ == "good")
    {
        matches_ = good_matches;
        return good_matches;   
    }
}

