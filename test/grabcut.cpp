#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Vec3f> circles;

void drawCircles(Mat& dst){    
    for( size_t i = 0; i < circles.size(); i++ )
    {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // draw the circle center
         circle( dst, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // draw the circle outline
         circle( dst, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
    //imshow("Result", dst);
}

void saturationChannel(Mat& src, Mat& dst){
    Mat tmp;
    cvtColor(src, tmp, COLOR_BGR2HSV);
    Mat hsv[3];
    split(tmp, hsv);

    dst = hsv[1].clone();
}

Mat getCleaningMask(Mat& src, Rect& bound_rect){

    Mat image = src.clone();
    GaussianBlur(image, image, Size(5,5),1);

    //Mask for cleaning up the segmented area
    Mat plate_thresh;
    Mat plate_sat;
    saturationChannel(image, plate_sat);
    threshold(plate_sat, plate_thresh,0,255, THRESH_OTSU);

    imshow("Mask thresh", plate_thresh);

    //Correct the thresholded image
    Mat corrected;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(plate_thresh, corrected, MORPH_CLOSE, element);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat contourOutput = corrected.clone();
    findContours(contourOutput, contours,hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( contours_poly[i], center[i], radius[i] );
    }

    Mat dst(corrected.size(), CV_8UC1);
    for(size_t i = 0; i < contours.size(); i++)
        if(contourArea(contours[i]) >= 80)
            if ( hierarchy[i][3] == -1) 
                drawContours(dst, contours, i, 255, 1);

    vector<Rect>::iterator maxEl = max_element(boundRect.begin(), boundRect.end(), 
                                    [](Rect const &r1, Rect const& r2){
                                        return r1.area() < r2.area();
                                    });

    bound_rect = *maxEl;

    
    /*for(size_t i = 0; i < contours.size(); i++)
        if(boundRect[i].area() >= 10000)
            if ( hierarchy[i][3] == -1){
                rectangle( dst, boundRect[i].tl(), boundRect[i].br(), 255, 2);
                //cout << "Area: " << boundRect[i].area() << endl;
            }*/
                
    return dst;
}

void grabCutSeg(Mat&src, Rect& rect){

    Mat img = src.clone();

    Mat mask;
    Mat bgdModel;
    Mat fgdModel;
    
    grabCut(img, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);
    
    // draw foreground
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (mask.ptr<uchar>(i, j)[0] == 0 || mask.ptr<uchar>(i, j)[0] == 2) {
                img.ptr<uchar>(i, j)[0] = 0;
                img.ptr<uchar>(i, j)[1] = 0;
                img.ptr<uchar>(i, j)[2] = 0;
            }
        }
    }
    imshow("canvasOutput", img);
}

int main(int argc, char **argv){

    if(argc < 2){
        cout << "Usage: task1.cpp <image>" << endl;
        return -1;
    }

    Mat src = imread(argv[1]);
    if(!src.data){  cout << "An error occured while loading the image." << endl; return -1; }
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    
    //Gaussian Blur for nois removal
    GaussianBlur(src_gray, src_gray, Size(3,3), 1);

    Mat dst = src.clone();
    HoughCircles(src_gray, circles, HOUGH_GRADIENT_ALT, 1, src_gray.rows/16, 300, 0.9, 200);
    drawCircles(dst);

    //Computes the saturation channel of the source image
    Mat sat_ch;
    saturationChannel(src, sat_ch);
    imshow("Saturation channel", sat_ch);

    Mat sat_thresh;
    threshold(sat_ch, sat_thresh, 130, 255, THRESH_BINARY);
    //imshow("Saturation Thresh", sat_thresh);

    //Compute regions of interest
    //A center is a triplet of [x,y,radius]
    vector<Mat> plates_roi;

    for(size_t i = 0; i < circles.size(); i++){
        //Get the rect containing the circle
        Point center(circles[i][0], circles[i][1]);
        int radius = round(circles[i][2]);

        double offset = 1.3; //offset to enlarge the ROI

        double w = 2 * cos(0.79)*radius *offset;
        double h = 2 * sin(0.79)*radius;

        Rect r(center.x - cos(0.79)*radius*offset, center.y - sin(0.79)*radius, w, h);

        //obtain ROI
        Mat roi(src, r);
        Mat mask(roi.size(), roi.type(), Scalar::all(255));

        // combine roi & mask:
        Mat plate = roi & mask;

        //add roi to the vector
        plates_roi.push_back(plate);
    }

    for (int i=0; i < plates_roi.size(); i++) {

        imshow("Original plate", plates_roi[i]);

        Rect b_rect;
        Mat mask = getCleaningMask(plates_roi[i], b_rect);
        imshow("Cleaning mask", mask);

        grabCutSeg(plates_roi[i], b_rect);
        
        waitKey(0);
    }

    return 0;
}
