#include <segmentation.hpp>

using namespace std;
using namespace cv;
using namespace zw;

/*************************************************************************************/
/*                      Utility functions used by the segmentation                   */
/*                        and detection functions defined below                      */
/*                                                                                   */
/*************************************************************************************/
void getCleaningMask(const Mat& src, Rect& boundRect){
    Mat mask;
    GaussianBlur(src, mask, Size(5,5), 1);
    cvtColor(mask, mask, COLOR_BGR2HSV);
    extractChannel(mask, mask, 1);

    //Mask for cleaning up the segmented area
    threshold(mask, mask, 195, 255, THRESH_BINARY);
    //imshow("Mask thresh", mask);

    //Correct the thresholded image
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(mask, mask, MORPH_CLOSE, element);

    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(mask, contours,hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> bound_Rect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        bound_Rect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( contours_poly[i], center[i], radius[i] );
    }

    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    for(size_t i = 0; i < contours.size(); i++)
        if(contourArea(contours[i]) >= 80)
            if ( hierarchy[i][3] == -1) 
                drawContours(dst, contours, i, 255, 1);

    vector<Rect>::iterator maxEl = max_element(bound_Rect.begin(), bound_Rect.end(), 
                                    [](Rect const &r1, Rect const& r2){
                                        return r1.area() < r2.area();
                                    });

    boundRect = *maxEl;
}

void grabCutSeg(const Mat&src,const Rect& rect, Mat& mask){
    Mat bgdModel, fgdModel;
    grabCut(src, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);
   
    inRange(mask, 3, 3, mask);
}




/*************************************************************************************/
/*                      Definitions of the functions defined                         */
/*                              in segmentation.hpp                                  */
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


void zw::segmentAndDetectPlates(Mat src, vector<Rect>& platesROI) {

}


void zw::segmentAndDetectSalad(Mat& src, vector<Rect>& saladROI) {
    for (int i = 0; i < saladROI.size(); i++) {
        Mat roi = Mat(src, saladROI[i]);

        Rect boundBox;
        getCleaningMask(roi, boundBox);

        Mat mask;
        grabCutSeg(roi, boundBox, mask);

        drawMask(roi, mask, SALAD);
    }
}


void zw::segmentAndDetectBread(Mat& src, vector<Rect>& breadROI) {

}
    