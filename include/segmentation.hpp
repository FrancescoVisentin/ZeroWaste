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
    constexpr int MIN_DETECTED_AREA = 500;
    constexpr int MIN_AREA_SALAD = 1000;
    constexpr int MIN_AREA_PLATES = 3500;
    constexpr int MIN_AREA_BREAD = 7000;
    constexpr int BREAD_COLOR_THRESHOLD = 150;
    constexpr float BREAD_AREA_THRESHOLD = 0.6;

    const std::array<std::pair<int,int>, 14> saturationRange = {
        std::pair<int, int> (  0,   0),    // Background
        std::pair<int, int> (200, 235),    // Pasta pesto
        std::pair<int, int> (145, 175),    // Pasta tomato
        std::pair<int, int> (200, 245),    // Pasta meat sauce
        std::pair<int, int> (190, 240),    // Pasta clams and mussels
        std::pair<int, int> (125, 160),    // Rice peppers and peas
        std::pair<int, int> ( 69, 100),    // Pork cutlet 69-86
        std::pair<int, int> (166, 170),    // Fish cutlet
        std::pair<int, int> (156, 166),    // Rabbit
        std::pair<int, int> (197, 202),    // Seafood salas 197-202 close 15
        std::pair<int, int> (172, 177),    // Beans 172-230
        std::pair<int, int> ( 60,  98),    // Basil potato
        std::pair<int, int> (182, 211),    // Salad
        std::pair<int, int> ( 90, 129)     // Bread
    };

    // Functions used to detect the position of foods inside the input image
    void getPlatesROI(const cv::Mat& gray, cv::Mat& roiMask, std::vector<cv::Rect>& platesROI, std::vector<cv::Mat>& platesMask);
    void getSaladROI(const cv::Mat& gray, cv::Mat& roiMask, std::vector<cv::Rect>& saladROI, std::vector<cv::Mat>& saladMask);

    // Functions that, given the ROIs, segment the regions and label the detected foods
    void segmentAndDetectPlates(cv::Mat& src, const std::vector<cv::Rect>& platesROI, const std::vector<cv::Mat>& platesMask, zw::Classifier& cf, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);
    void segmentAndDetectBread(cv::Mat& src, const cv::Mat& roiMask, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);
    void segmentSalad(cv::Mat& src, const std::vector<cv::Rect>& saladROI, const std::vector<cv::Mat>& saladMask, cv::Mat& foodsMask, std::vector<std::pair<cv::Rect,int>>& trayItems);
}

#endif