#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Vec3f> circles;
vector<Mat> plates_roi;
const char* window_name1 = "Thresh: ";
int tVal = 40;
int tMax = 255;

int NUM = 5;
int OFFSET = 0;

static void thresh(int, void*) {
    for(int i = NUM*OFFSET; i < plates_roi.size() && i <NUM+NUM*OFFSET; i++) {
        Mat bw;
        cvtColor(plates_roi[i], bw, COLOR_BGR2HSV);
        extractChannel(bw, bw, 1);
        imshow("Gamma pre thresh "+ to_string(i+1), bw);
        moveWindow("Gamma pre thresh "+ to_string(i+1), 440*(i-NUM*OFFSET), 500);
    
        threshold(bw, bw, tVal, 255, THRESH_BINARY); //Better threshold here to discard empty plates
        imshow("Binary Image "+to_string(i+1), bw);
        moveWindow("Binary Image "+to_string(i+1), 440*(i-NUM*OFFSET), 10);
    }
}

#include <opencv2/core/utils/filesystem.hpp>

int main(int argc, char **argv){
    vector<string> paths;
    glob("/home/francesco/Scaricati/Food_leftover_dataset/*.jpg", paths, true);

    for (int i = 0; i < paths.size(); i++) {
        cout<<paths[i]<<"\n";
    }
   
    for(int j = 0; j < paths.size(); j++) {
        Mat src = imread(paths[j]);
        if(!src.data){  cout << "An error occured while loading the image." << endl; return -1; }
        Mat src_gray;
        cvtColor(src, src_gray, COLOR_BGR2GRAY);

        //Gaussian Blur for nois removal
        GaussianBlur(src_gray, src_gray, Size(3,3), 1);

        Mat dst = src.clone();
        HoughCircles(src_gray, circles, HOUGH_GRADIENT_ALT, 1, src_gray.rows/16, 300, 0.9, 200);

        //Compute regions of interest
        //A center is a triplet of [x,y,radius]
        for(size_t i = 0; i < circles.size(); i++){
            //Get the rect containing the circle
            Point center(circles[i][0], circles[i][1]);
            int radius = round(circles[i][2]);

            int a = (center.x > radius) ? center.x-radius : 0;
            int b = (center.y > radius) ? center.y-radius : 0;
            int width = (a+radius*2 < src.cols) ? radius*2 : src.cols-a; 
            int heigth = (b+radius*2 < src.rows) ? radius*2 : src.rows-b;
            Rect r = Rect(a, b, width, heigth);
           
            //obtain ROI
            Mat roi(src, r);
            Mat mask(roi.size(), roi.type(), Scalar::all(255));
            // with a white, filled circle in it:
            circle(mask, Point(0.75*radius,0.75*radius), 0.75*radius, Scalar::all(255), -1);

            // combine roi & mask:
            Mat plate = roi & mask;

            //add roi to the vector
            plates_roi.push_back(plate);
        }
    }

    namedWindow(window_name1, WINDOW_AUTOSIZE);
    createTrackbar("Threshold:", window_name1, &tVal, tMax, thresh);

    waitKey(0);

    return 0;
}