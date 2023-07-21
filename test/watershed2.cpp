#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Vec3f> circles;

void gammaCorrection(Mat& image, Mat& dst, double gamma=1){
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    Mat gamma_corr = image.clone();
    LUT(image, lookUpTable, gamma_corr);

    cvtColor(gamma_corr, gamma_corr, COLOR_BGR2HSV);
    imshow("gamma", gamma_corr);

    extractChannel(gamma_corr, dst, 1);

    GaussianBlur(dst, dst, Size(7,7), 0);
}

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
    imshow("Result", dst);
}

void water(Mat& img, Mat& dst, int width, int height) {
    blur(img, img, Size(7,7));

    Mat bw;
    gammaCorrection(img, bw);
    imshow("Gamma pre OTSU", bw);
    moveWindow("Gamma pre OTSU", width*2, 10);

    Mat i;
    cvtColor(bw,i,COLOR_GRAY2BGR);

    // Create binary image from source image
    //cvtColor(tmp, bw, COLOR_BGR2GRAY);
    //cout<<threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU)<<"\n"; //Better threshold here to discard empty plates
    threshold(bw, bw, 70, 255, THRESH_BINARY);
    imshow("Binary Image", bw);
    moveWindow("Binary Image", width*3, 10);
    
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    Mat dd = dist.clone();
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    moveWindow("Distance Transform Image", 0, 100+height);
    
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    moveWindow("Peaks", width, 100+height);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    vector<pair<Point, float>> p;
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);

        double max, min;
        Point maxPos, minPos;
        Mat tmp = Mat::zeros(dist.size(), CV_8U);
        drawContours(tmp, contours, static_cast<int>(i), 255, -1);
        minMaxLoc(dd, &min, &max, &minPos, &maxPos, tmp);
        p.push_back(pair(maxPos,max));
    }

    sort(p.begin(), p.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });

    int NUM = 4;

    Mat myMarkers = Mat::zeros(dist.size(), CV_32S);
    circle(myMarkers, Point(100,100), 3, Scalar(255), -1);
    circle(myMarkers, Point(myMarkers.cols-100,100), 3, Scalar(255), -1);
    circle(myMarkers, Point(100,myMarkers.rows-100), 3, Scalar(255), -1);
    circle(myMarkers, Point(myMarkers.cols-100,myMarkers.rows-100), 3, Scalar(255), -1);
    for (int i = 0; i < p.size() && i < NUM; i++) {
        circle(myMarkers, p[i].first, 3, Scalar(i+1), -1);
    }
    Mat myM8u;
    myMarkers.convertTo(myM8u, CV_8U);
    imshow("myM", myM8u*50);


    // Draw the background marker
    circle(markers, Point(100,100), 3, Scalar(255), -1);
    circle(markers, Point(markers.rows-100,100), 3, Scalar(255), -1);
    circle(markers, Point(100,markers.cols-100), 3, Scalar(255), -1);
    circle(markers, Point(markers.rows-100,markers.cols-100), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers", markers8u);
    moveWindow("Markers", width*2, 100+height);

    watershed(i, myMarkers);
    Mat myMark;
    myMarkers.convertTo(myMark, CV_8U);
    bitwise_not(myMark, myMark);
    imshow("MyMarkers_v2", myMark); // uncomment this if you want to see how the mark image looks like at that point
    
    // Generate random colors
    vector<Vec3b> myColors;
    for (size_t i = 0; i < NUM; i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        myColors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat myDst = Mat::zeros(myMarkers.size(), CV_8UC3);
    //Fill labeled objects with random colors
    for (int i = 0; i < myMarkers.rows; i++) {
        for (int j = 0; j < myMarkers.cols; j++) {
            int index = myMarkers.at<int>(i,j);
            if (index > 0 && index <= NUM) {
                myDst.at<Vec3b>(i,j) = myColors[index-1];
            }
        }
    }
    imshow("Res", myDst);


    // Perform the watershed algorithm
    watershed(i, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark image looks like at that point
    moveWindow("Markers_v2", width*3, 100+height);
    
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
        }
    }
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

    GaussianBlur(src_gray, src_gray, Size(3,3), 1);

    
    Mat dst = src.clone();
    HoughCircles(src_gray, circles, HOUGH_GRADIENT_ALT, 1, src_gray.rows/16, 300, 0.9, 200);
    drawCircles(dst);

    //Compute regions of interest
    //A center is a triplet of [x,y,radius]
    vector<Mat> plates_roi;
    for(size_t i = 0; i < circles.size(); i++){
        //Get the rect containing the circle
        Point center(circles[i][0], circles[i][1]);
        int radius = round(circles[i][2]);

        int x = (center.x > radius) ? center.x-radius: 0;
        int y = (center.y > radius) ? center.y-radius: 0;

        Rect r = Rect(x, y, radius*2, radius*2);

        cout<<r<<"\n";

        //obtain ROI
        Mat roi= Mat(src, r);
        Mat mask = Mat::zeros(roi.size(), roi.type());
        // with a white, filled circle in it:
        circle(mask, Point(radius,radius), radius, Scalar::all(255), -1);

        // combine roi & mask:
        Mat plate = roi & mask;

        //add roi to the vector
        plates_roi.push_back(plate);
    }

    for (int i=0; i < plates_roi.size(); i++) {        
        Mat d;
        water(plates_roi[i], d, plates_roi[i].cols, plates_roi[i].rows);
        imshow("Final plate ", d);
        waitKey(0);
    }

    return 0;
}