#ifndef segmentation_utils_h
#define segmentation_utils_h

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

namespace zw_utils{

    //Function used to filter spurious circles
    void filterCircles(const std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& filtered);

    //Function used to select bounding box
    bool filterBox(cv::Rect& r);
}

#endif