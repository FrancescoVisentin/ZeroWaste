#ifndef segmentation_h
#define segmentation_h

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <common.hpp>
#include <iostream>
#include <string>

namespace zw {
    // Constant values/thresholds used inside the code
    constexpr int SAT_THRESH_SALAD = 185;
    constexpr int MIN_AREA_SALAD = 10000;

    //Function used to filter the circles found
    void filterCircles(const std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& filtered);

    // Functions used to detect the position of foods inside the input image
    void getPlatesROI(const cv::Mat& gray, cv::Mat& mask, std::vector<cv::Rect>& platesROI); //Lorenzo
    void getSaladROI(const cv::Mat& gray, cv::Mat& mask, std::vector<cv::Rect>& saladROI);  //Lorenzo
    void getBreadROI(const cv::Mat& src, std::vector<cv::Rect>& breadROI); //Alberto

    // Functions that, given the ROIs, segment the regions and label the detected foods
    void segmentAndDetectPlates(cv::Mat src, std::vector<cv::Rect>& platesROI, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems); //Da fare da zero
    void segmentAndDetectSalad(cv::Mat& src, std::vector<cv::Rect>& saladROI, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);  //fatto
    void segmentAndDetectBread(cv::Mat& src, std::vector<cv::Rect>& breadROI, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems); //Da fare da zero
}



#endif