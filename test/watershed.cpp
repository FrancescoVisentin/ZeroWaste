#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Vec3f> circles;


void kmeans_segmentation(Mat& src, Mat& dst, int K) {
    Mat features = Mat(src.rows*src.cols, 3, CV_32F);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            for (int z = 0; z < 3; z++)
                features.at<float>(i*src.cols + j, z) = src.at<Vec3b>(i,j)[z];

    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER, 15, 1.0);

    Mat labels,centers;
    kmeans(features, K, labels, criteria, 5, KMEANS_PP_CENTERS, centers);
    
    centers.convertTo(centers, CV_8UC3);
    dst = Mat(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int l = labels.at<int>(i*src.cols + j, 0);
            dst.at<Vec3b>(i,j) = centers.at<Vec3b>(l);
        }
    }
}

void gammaCorrection(Mat& image, Mat& dst, double gamma=0.5){
    GaussianBlur(image, image, Size(5,5), 1);

    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    Mat gamma_corr = image.clone();
    LUT(image, lookUpTable, gamma_corr);

    Mat tmp;
    cvtColor(gamma_corr, tmp, COLOR_BGR2HSV);
    Mat hsv[3];
    split(tmp, hsv);

    dst = hsv[1].clone();
    GaussianBlur(dst, dst, Size(5,5),1);
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
    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(img, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    img.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    imshow( "Laplace Filtered Image", imgLaplacian );
    moveWindow("Laplace Filtered Image", width, 10);
    imshow( "New Sharped Image", imgResult );
    moveWindow("New Sharped Image", width*2, 10);

    Mat bw;
    gammaCorrection(imgResult, bw);
    imshow("Gamma pre OTSU", bw);

    // Create binary image from source image
    //cvtColor(tmp, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);
    moveWindow("Binary Image", width*3, 10);
    
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

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
    
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }
    
    // Draw the background marker
    circle(markers, Point(50,50), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers", markers8u);
    moveWindow("Markers", width*2, 100+height);

    
    // Perform the watershed algorithm
    watershed(imgResult, markers);
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
    
    //Gaussian Blur for nois removal
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

        Rect r (center.x - round(0.75*radius), center.y - round(0.75*radius), 0.75*radius*2,  0.75*radius*2);

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

    for (int i=0; i < plates_roi.size(); i++) {
        Mat m;

        int K = (argc >= 3) ? atoi(argv[2]) : 0;
        
        if (K > 0) kmeans_segmentation(plates_roi[i], m, K);
        else       pyrMeanShiftFiltering(plates_roi[i], m, 30, 35, 2);

        cout<<plates_roi[i].size()<<"\n";
        imshow("Input clustering plate", m);
        moveWindow("Input clustering plate", 0, 10);

        Mat d;
        water(m, d, plates_roi[i].cols, plates_roi[i].rows);
        imshow("Final plate ", d);
        waitKey(0);
    }

    return 0;
}