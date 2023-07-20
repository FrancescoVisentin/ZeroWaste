#ifndef metrics_h
#define metrics_h

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

namespace zw {
    // Functions used to compute the required metrics to evaluate the segmentation and detection performances of the code
    void mAP(const std::vector<std::vector<std::pair<cv::Rect,int>>>& detectedItemsPerTray, std::string trayPath);
    void mIoU(const std::vector<std::vector<std::pair<cv::Rect,int>>>& detectedItemsPerTray, std::string trayPath);
    void leftoverRatio(const std::vector<cv::Mat>& foodMasks, std::string trayPath);
}


#endif
