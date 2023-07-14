#ifndef segmentation_h
#define segmentation_h

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <common.hpp>
#include <iostream>
#include <string>

namespace zw {

    void getPlatesROI(const cv::Mat& gray, cv::Mat& mask, std::vector<cv::Rect>& platesROI); //Lorenzo
    void getSaladROI(const cv::Mat& gray, cv::Mat& mask, std::vector<cv::Rect>& saladROI);  //Lorenzo
    void getBreadROI(const cv::Mat& src, std::vector<cv::Rect>& breadROI); //Alberto

    void segmentAndDetectPlates(cv::Mat src, std::vector<cv::Rect>& platesROI); //Da fare da zero
    void segmentAndDetectSalad(cv::Mat& src, std::vector<cv::Rect>& saladROI);  //fatto
    void segmentAndDetectBread(cv::Mat& src, std::vector<cv::Rect>& breadROI); //Da fare da zero
}



#endif