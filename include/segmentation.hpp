#ifndef segmentation_h
#define segmentation_h

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <classifier.hpp>
#include <common.hpp>
#include <iostream>
#include <string>

namespace zw {
    // Constant values/thresholds used inside the code
    constexpr int MIN_AREA_SALAD = 1000;
    constexpr int MIN_AREA_PLATES = 1000;
    constexpr int MIN_AREA_BREAD = 1000;

    const std::array<std::pair<int,int>, 14> saturationRange = {
        std::pair<int, int> (  0,   0),    // Background
        std::pair<int, int> (190, 255),    // Pasta pesto
        std::pair<int, int> (190, 255),    // Pasta tomato
        std::pair<int, int> (190, 255),    // Pasta meat sauce
        std::pair<int, int> (190, 255),    // Pasta clams and mussels
        std::pair<int, int> (190, 255),    // Rice peppers and peas
        std::pair<int, int> ( 69,  86),    // Pork cutlet
        std::pair<int, int> (166, 170),    // Fish cutlet
        std::pair<int, int> (204, 225),    // Rabbit
        std::pair<int, int> (  0, 255),    // Seafood salas
        std::pair<int, int> (172, 230),    // Beans
        std::pair<int, int> ( 60,  98),    // Basil potato
        std::pair<int, int> (182, 211),    // Salad
        std::pair<int, int> ( 60,  98)     // Bread
    };

    // Functions used to detect the position of foods inside the input image
    void getPlatesROI(const cv::Mat& gray, cv::Mat& roiMask, std::vector<cv::Rect>& platesROI, std::vector<cv::Mat>& platesMask);
    void getSaladROI(const cv::Mat& gray, cv::Mat& roiMask, std::vector<cv::Rect>& saladROI, std::vector<cv::Mat>& saladMask);
    void getBreadROI(const cv::Mat& src, std::vector<cv::Rect>& breadROI);

    // Functions that, given the ROIs, segment the regions and label the detected foods
    void segmentAndDetectPlates(cv::Mat& src, const std::vector<cv::Rect>& platesROI, const std::vector<cv::Mat>& platesMask, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);
    void segmentAndDetectSalad(cv::Mat& src, const std::vector<cv::Rect>& saladROI, const std::vector<cv::Mat>& saladMask, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);
    void segmentAndDetectBread(cv::Mat& src, const std::vector<cv::Rect>& breadROI, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);
}

#endif