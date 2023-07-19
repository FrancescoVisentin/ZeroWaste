#ifndef metrics_h
#define metrics_h

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <algorithm>

namespace zw {

    struct BoundingBox {
        int left, top, right, bottom;
    };




    double averagePrecision(const std::vector<std::vector<std::pair<cv::Rect,int>>>& detectedItemsPerTray, std::string resPath);
    double IoU(const std::vector<std::vector<std::pair<cv::Rect,int>>>& detectedItemsPerTray, std::string resPath);
    double leftoverRatio(const std::vector<cv::Mat>& foodMasks, std::string resPath);
}


#endif