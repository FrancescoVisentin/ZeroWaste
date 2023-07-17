#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <segmentation_utils.hpp>

using namespace std;
using namespace cv;


// Function used to filter the circles found by Hough circles
void zw_utils::filterCircles(const std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& filtered) {
    for(int i = 0; i < circles.size(); i++)
        //Filter by radius size
        if(circles[i][2] >= 186 && circles[i][2] <= 240) filtered.push_back(circles[i]);
    
    //If more than one salad plate is found then remove one
    if(filtered.size() > 1) filtered.pop_back();
}


 //Function used to select bounding box
bool zw_utils::filterBox(Rect& r){

    if(r.area() > 30000){
        if(r.width >= r.height)  //Filters out long and thin boxes
            
            return (r.width/r.height < 2.5) ? true : false; 

        else //Filters out high and thin boxes
            
            return (r.height/r.width < 1.8) ? true : false;
    }

    return false;
}
