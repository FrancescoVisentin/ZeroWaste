#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Vec3f> circles;
vector<Mat> plates_roi;
const char* window_name1 = "Thresh: ";
int lower = 40;
int upper = 60;
int tMax = 255;

int NUM = 4;
int OFFSET = 0;

int iter = 0;

static void thresh(int, void*) {
    for(int i = NUM*OFFSET; i < plates_roi.size() && i <NUM+NUM*OFFSET; i++) {
        Mat satMask;
        GaussianBlur(plates_roi[i], satMask, Size(5,5), 1);
        cvtColor(satMask, satMask, COLOR_BGR2HSV);
        extractChannel(satMask, satMask, 1);

        // Threshold on the saturation channel to detect main components
        //threshold(satMask, satMask, thresh, 255, THRESH_BINARY);
        inRange(satMask, lower, upper, satMask);

        // Correct the thresholded image
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20,20));
        morphologyEx(satMask, satMask, MORPH_CLOSE, element);
        element = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
        morphologyEx(satMask, satMask, MORPH_OPEN, element);

        // Find contours in the thesholded image and compute a bounding box for each one
        vector<Vec4i> hierarchy;
        vector<vector<Point>> contours;
        findContours(satMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        vector<Rect> contoursRect;
        for (int j = 0; j < contours.size(); j++) {
            vector<Point> contoursPoly;
            approxPolyDP( Mat(contours[j]), contoursPoly, 3, true );
            
            contoursRect.push_back(boundingRect(contoursPoly));
        }

        if (contoursRect.size() > 0) {
            // Takes the largest bounding box
            Mat dst = plates_roi[i].clone();
            for(size_t j = 0; j < contours.size(); j++) {
                double area = contourArea(contours[j]); 
                
                if (area >= 1000 && hierarchy[j][3] == -1) {
                    drawContours(dst, contours, j, Scalar(0,255,0), 1);
                    rectangle(dst, contoursRect[j], 255);
                    rectangle(satMask, contoursRect[j], 255);
                }
            }


            resize(dst, dst, dst.size()/2);
            resize(satMask, satMask, satMask.size()/2);
            imshow("Gamma pre thresh "+ to_string(i+1), dst);
            if (iter < NUM) moveWindow("Gamma pre thresh "+ to_string(i+1), 620*(i-NUM*OFFSET), 800);
                
            // Correct the thresholded image
            imshow("Binary Image "+to_string(i+1), satMask);
            if (iter++ < NUM) moveWindow("Binary Image "+to_string(i+1), 620*(i-NUM*OFFSET), 10);
        }
    }
}

#include <opencv2/core/utils/filesystem.hpp>

int main(int argc, char **argv){
    vector<string> paths;
    string p = argv[1];
    glob(p+"*.jpg", paths, true);

    for (int i = 0; i < paths.size(); i++) {
        cout<<paths[i]<<"\n";
    }

    //OFFSET = atoi(argv[2]);
   
    for(int j = 0; j < paths.size(); j++) {
        Mat src = imread(paths[j]);
        if(!src.data){  cout << "An error occured while loading the image." << endl; return -1; }

        //add roi to the vector
        plates_roi.push_back(src);
    }

    namedWindow(window_name1, WINDOW_NORMAL);
    createTrackbar("Lower:", window_name1, &lower, tMax, thresh);
    createTrackbar("Upper:", window_name1, &upper, tMax, thresh);

    waitKey(0);

    return 0;
}