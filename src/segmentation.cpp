#include <segmentation.hpp>

using namespace std;
using namespace cv;
using namespace zw;

/*************************************************************************************/
/*                      Utility functions used by the segmentation                   */
/*                        and detection functions defined below                      */
/*                                                                                   */
/*************************************************************************************/

// Using the saturation channel, extracts the main areas of interest from the
// input image and computes a bounding rectangle around them
void detectMainComponents(const Mat& src, int thresh, Rect& boundRect){
    Mat satMask;
    GaussianBlur(src, satMask, Size(5,5), 1);
    cvtColor(satMask, satMask, COLOR_BGR2HSV);
    extractChannel(satMask, satMask, 1);

    // Threshold on the saturation channel to detect main components
    threshold(satMask, satMask, thresh, 255, THRESH_BINARY);

    // Correct the thresholded image
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
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

    // Takes the largest bounding box
    boundRect = contoursRect[0];
    for (int i = 1; i < contoursRect.size(); i++) {
        if (contoursRect[i].area() > boundRect.area()) boundRect = contoursRect[i];
    }

    Mat dst = src.clone();
    for(size_t i = 0; i < contours.size(); i++)
        if (contourArea(contours[i]) >= 80)
            if ( hierarchy[i][3] == -1) 
                drawContours(dst, contours, i, Scalar(0,255,0), 1);
    
    /*                                                                  //TODO: remove
    rectangle(dst, boundRect, 255);
    imshow("mask", dst);
    imshow("threshold", satMask);
    cout<<boundRect.area()<<"\n";
    waitKey(0);
    */
}

// Applies the cv::grubCut segmentation algorithm inside the area defined by the rect given in input 
void grabCutSeg(const Mat&src, const Rect& rect, int id, Mat& mask){
    Mat bgdModel, fgdModel;
    grabCut(src, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);
   
    // Assigns to the pixels in the foreground the given food id
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i,j) == GC_FGD || mask.at<uchar>(i,j) == GC_PR_FGD) mask.at<uchar>(i,j) = id;
            else mask.at<uchar>(i,j) = 0;
        }
    }
}




/*************************************************************************************/
/*                     Definitions of the functions declared                         */
/*                             in segmentation.hpp                                   */
/*                                                                                   */
/*************************************************************************************/
void zw::getPlatesROI(const Mat& gray, Mat& mask, vector<Rect>& platesROI) {
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 1, 200, 400, 0.7, 230);

    for (int i = 0; i < circles.size(); i++) {
        Point center = Point(circles[i][0], circles[i][1]);
        int radius = circles[i][2];

        int x = (center.x-radius > 0) ? center.x-radius : 0;
        int y = (center.y-radius > 0) ? center.y-radius : 0;
        int width = (x+2*radius < gray.cols) ? 2*radius : gray.cols-x;
        int height = (y+2*radius < gray.rows) ? 2*radius : gray.rows-y;
        Rect roi = Rect(x, y, width, height);

        circle(mask, center, radius, Scalar::all(255), -1);
        platesROI.push_back(roi);
    }
}


void zw::getSaladROI(const Mat& gray, Mat& mask, vector<Rect>& saladROI) {
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 1, 400, 250, 0.88, 140, 230);

    for (int i = 0; i < circles.size() & i < 1; i++) {
        Point center = Point(circles[i][0], circles[i][1]);
        int radius = round(circles[i][2]);

        int x = (center.x-radius > 0) ? center.x-radius : 0;
        int y = (center.y-radius > 0) ? center.y-radius : 0;
        int width = (x+2*radius < gray.cols) ? 2*radius : gray.cols-x;
        int height = (y+2*radius < gray.rows) ? 2*radius : gray.rows-y;
        Rect roi = Rect(x, y, width, height);

        circle(mask, center, radius, Scalar::all(255), -1);
        saladROI.push_back(roi);
    }
}


void zw::getBreadROI(const Mat& src, vector<Rect>& breadROI) {
   
}


void zw::segmentAndDetectPlates(Mat src, vector<Rect>& platesROI, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {

}


void zw::segmentAndDetectSalad(Mat& src, vector<Rect>& saladROI, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {
    for (int i = 0; i < saladROI.size(); i++) {
        Mat roi = Mat(src, saladROI[i]);

        // Computes a minimum size bounding box around the salad inside the bowl
        Rect boundBox;
        detectMainComponents(roi, SAT_THRESH_SALAD, boundBox);

        // Checks if the size of the returned bounding box is big enough
        if (boundBox.area() < MIN_AREA_SALAD) return;

        Mat mask;
        grabCutSeg(roi, boundBox, SALAD, mask);

        // Add the detected segmented region to the tray mask
        Mat(foodsMask, saladROI[i]) += mask;

        // Save the bounding box of the above detected region 
        Rect finalBox = boundingRect(mask);
        boundBox.x += saladROI[i].x;
        boundBox.y += saladROI[i].y;
        trayItems.push_back(pair(boundBox, SALAD));

        // Draws the overlay for showing the results
        drawMask(roi, mask);

        /*                                                                  //TODO: remove
        imshow("m", roi);
        waitKey(0);
        */
    }
}


void zw::segmentAndDetectBread(Mat& src, vector<Rect>& breadROI, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {

}
    