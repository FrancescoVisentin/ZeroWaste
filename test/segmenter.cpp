#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../include/common.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace zw;


double detectMainComponents(const Mat& src, int thresh, Rect& boundRect){
    Mat satMask;
    GaussianBlur(src, satMask, Size(5,5), 0.7);
    cvtColor(satMask, satMask, COLOR_BGR2HSV);
    extractChannel(satMask, satMask, 1);

    // Threshold on the saturation channel to detect main components
    threshold(satMask, satMask, thresh, 255, THRESH_BINARY);

    // Correct the thresholded image
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20,20));
    morphologyEx(satMask, satMask, MORPH_CLOSE, element);

    // Find contours in the thesholded image and compute a bounding box for each one
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(satMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    vector<Rect> contoursRect;
    for (int i = 0; i < contours.size(); i++) {
        vector<Point> contoursPoly;
        approxPolyDP( Mat(contours[i]), contoursPoly, 3, true );
        
        contoursRect.push_back(boundingRect(contoursPoly));
    }

    if (contoursRect.size() > 0) {
        // Takes the largest bounding box
        boundRect = contoursRect[0];
        for (int i = 1; i < contoursRect.size(); i++) {
            if (contoursRect[i].area() > boundRect.area()) boundRect = contoursRect[i];
        }

        Mat dst = src.clone();
        boundRect = contoursRect[0];
        double area = contourArea(contours[0]);
        for(size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) >= 80)
                if ( hierarchy[i][3] == -1) 
                    drawContours(dst, contours, i, Scalar(0,255,0), 1);

            if (contourArea(contours[i]) > area) {
                area = contourArea(contours[i]);
                boundRect = contoursRect[i];
            }
        }   
        
        ///*                                                                  //TODO: remove
        rectangle(dst, boundRect, 255);
        imshow("mask", dst);
        imshow("threshold", satMask);
        //cout<<boundRect.area()<<"\n";
        //waitKey(0);
        //*/

        return area;

    }   

    return 0;
}

void grabCutSeg(const Mat& src, int id, Mat& mask) {
    Mat bgdModel, fgdModel;
    grabCut(src, mask, Rect(), bgdModel, fgdModel, 5, GC_INIT_WITH_MASK);
   
    // Assigns to the pixels in the foreground the given food id
    static_cast<u_char>(id);
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<u_char>(i,j) == GC_FGD || mask.at<u_char>(i,j) == GC_PR_FGD) mask.at<u_char>(i,j) = id;
            else mask.at<u_char>(i,j) = 0;
        }
    }
}


int main(int argc, char const *argv[]) {
    vector<string> path;

    string p = argv[1];
    glob(p+"*.jpg", path, false);

    for (int i = 0; i < path.size(); i++) {
        Mat roi = imread(path[i]);

        Mat tmp = roi.clone();
       
        // Computes a minimum size bounding box around the salad inside the bowl
        Rect boundBox;
        double area = detectMainComponents(tmp, atoi(argv[2]), boundBox);

        Mat mask = Mat::zeros(roi.size(), CV_8U);
        if (area > 5000) {
            rectangle(mask, boundBox, GC_PR_FGD, -1);
            grabCutSeg(tmp, 1, mask);
            drawMask(roi, mask);

            threshold(mask, mask, 0.5, 255, THRESH_BINARY);
            Mat m2;
            cvtColor(mask, m2, COLOR_GRAY2BGR);
            tmp -= m2;


            imshow("m", roi);
            waitKey(0);
        }
        

        area = detectMainComponents(tmp, atoi(argv[3]), boundBox);
        if (area < 5000) {
            cout<<"skipped!\n";
            waitKey(0);
            continue;
        }

        Mat m3 = Mat::zeros(roi.size(), CV_8U);
        rectangle(m3, boundBox, GC_PR_FGD, -1);
        m3 -= mask;
        grabCutSeg(tmp, 2, m3);
        drawMask(roi, m3);

        imshow("m3",m3*30);
        imshow("m", roi);
        waitKey(0);
    }

    return 0;
}

