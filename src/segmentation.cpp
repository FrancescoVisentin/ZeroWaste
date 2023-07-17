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
// input image and initializes a cv::grubCut mask with possible foreground/background
void detectMainComponents(const Mat& src, int lower, int upper, int minArea, Mat& mask) {
    Mat satMask;
    GaussianBlur(src, satMask, Size(5,5), 1);
    cvtColor(satMask, satMask, COLOR_BGR2HSV);
    extractChannel(satMask, satMask, 1);

    // Extract a saturation range to extract the desired item
    inRange(satMask, lower, upper, satMask);

    // Correct the obtained binary image
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20,20));
    morphologyEx(satMask, satMask, MORPH_CLOSE, element);
    element = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
    morphologyEx(satMask, satMask, MORPH_OPEN, element);

    // Find contours in the image and compute a bounding box for each one
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(satMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    vector<Rect> contoursRect;
    for (int i = 0; i < contours.size(); i++) {
        vector<Point> contoursPoly;
        approxPolyDP( Mat(contours[i]), contoursPoly, 3, true );
        
        contoursRect.push_back(boundingRect(contoursPoly));
    }

    mask = Mat::zeros(src.size(), CV_8U);
    if (contoursRect.size() > 0) {
        // Selects the main areas of interest and initialize the mask
        Mat dst = src.clone();
        for (int i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) >= minArea && hierarchy[i][3] == -1) {
                drawContours(dst, contours, i, Scalar(0,255,0), 1);
                rectangle(dst, contoursRect[i], 255);
                rectangle(mask, contoursRect[i], GC_PR_BGD, -1);
            }
        }

        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                if (satMask.at<u_char>(i,j) && mask.at<u_char>(i,j) == GC_PR_BGD) mask.at<u_char>(i,j) = GC_PR_FGD; 
            }
        }

        ///*                                                                  //TODO: remove
        imshow("Contours", dst);
        imshow("Mask", mask*30);
        //*/
    }   
}

// Applies the cv::grubCut segmentation algorithm inside the area defined by the rect given in input 
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

// Function used to filter the circles found by Hough circles
void filterCircles(const vector<Vec3f>& circles, vector<Vec3f>& filtered) {
    for(int i = 0; i < circles.size(); i++)
        //Filter by radius size
        if(circles[i][2] >= 186 && circles[i][2] <= 240) filtered.push_back(circles[i]);
    
    //If more than one salad plate is found then remove one
    if(filtered.size() > 1) filtered.pop_back();
}



/*************************************************************************************/
/*                     Definitions of the functions declared                         */
/*                             in segmentation.hpp                                   */
/*                                                                                   */
/*************************************************************************************/
void zw::getPlatesROI(const Mat& gray, Mat& roiMask, vector<Rect>& platesROI, vector<Mat>& platesMask) {
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 23, 20, 387, 0.75, 238);

    for (int i = 0; i < circles.size(); i++) {
        Point center = Point(circles[i][0], circles[i][1]);
        int radius = circles[i][2];

        int x = (center.x-radius > 0) ? center.x-radius : 0;
        int y = (center.y-radius > 0) ? center.y-radius : 0;
        int width = (x+2*radius < gray.cols) ? 2*radius : gray.cols-x;
        int height = (y+2*radius < gray.rows) ? 2*radius : gray.rows-y;
        Rect roi = Rect(x, y, width, height);

        circle(roiMask, center, radius, Scalar::all(255), -1);
        platesROI.push_back(roi);

        Mat mask = Mat::zeros(roi.size(), CV_8UC3);
        circle(mask, Point(center.x-x, center.y-y), radius, Scalar::all(255), -1);
        platesMask.push_back(mask);
    }
}


void zw::getSaladROI(const Mat& gray, Mat& roiMask, vector<Rect>& saladROI, vector<Mat>& saladMask) {
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 1.5, gray.rows/16, 310, 0.36, 150, 300);

    vector<Vec3f> filtered_circles;
    filterCircles(circles, filtered_circles);

    for (int i = 0; i < filtered_circles.size(); i++) {
        Point center = Point(filtered_circles[i][0], filtered_circles[i][1]);
        int radius = round(filtered_circles[i][2]);

        int x = (center.x-radius > 0) ? center.x-radius : 0;
        int y = (center.y-radius > 0) ? center.y-radius : 0;
        int width = (x+2*radius < gray.cols) ? 2*radius : gray.cols-x;
        int height = (y+2*radius < gray.rows) ? 2*radius : gray.rows-y;
        Rect roi = Rect(x, y, width, height);

        circle(roiMask, center, radius, Scalar::all(255), -1);
        saladROI.push_back(roi);

        Mat mask = Mat::zeros(roi.size(), CV_8UC3);
        circle(mask, Point(center.x-x, center.y-y), radius, Scalar::all(255), -1);
        saladMask.push_back(mask);
    }
}


void zw::getBreadROI(const Mat& src, vector<Rect>& breadROI) {
      
}


void zw::segmentAndDetectPlates(Mat& src, const vector<Rect>& platesROI, const vector<Mat>& platesMask, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {
    for (int i = 0; i < platesROI.size(); i++) {
        Mat roi = Mat(src, platesROI[i]).clone();
        roi &= platesMask[i];

        vector<int> plateItemsIDs;
        detect(roi, plateItemsIDs);

        for (int j = 0; j < plateItemsIDs.size(); j ++) {
            // Get the saturation range for the current plate item
            pair<int, int> satRange = saturationRange[plateItemsIDs[j]]; 

            // Computes a mask of probable foreground/background for the current food 
            Mat mask;
            detectMainComponents(roi, satRange.first, satRange.second, MIN_AREA_PLATES, mask);
            
            if (countNonZero(mask) > 0) {
                // Segment the food starting from the mask
                grabCutSeg(roi, plateItemsIDs[j], mask);

                // Add the detected segmented region to the tray mask
                Mat(foodsMask, platesROI[i]) += mask;

                // Save the bounding box of the above detected region 
                Rect finalBox = boundingRect(mask);
                finalBox.x += platesROI[i].x;
                finalBox.y += platesROI[i].y;
                trayItems.push_back(pair<Rect,int>(finalBox, plateItemsIDs[j]));

                // Draws the overlay on the src image for showing the results
                Mat srcROI = Mat(src, platesROI[i]);
                drawMask(srcROI, mask);

                // Removes the region segmented in this iteration
                threshold(mask, mask, 0.5, 255, THRESH_BINARY);
                cvtColor(mask, mask, COLOR_GRAY2BGR);
                roi -= mask;
            
                ///*                                                                  //TODO: remove
                imshow("tmp", roi);
                imshow("res", srcROI);
                waitKey(0);
                //*/
            }
        }
    }
}


void zw::segmentAndDetectSalad(Mat& src, const vector<Rect>& saladROI, const vector<Mat>& saladMask, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {
    for (int i = 0; i < saladROI.size(); i++) {
        Mat roi = Mat(src, saladROI[i]).clone();
        roi &= saladMask[i];

        // Get the salad saturation range
        pair<int, int> satRange = saturationRange[SALAD]; 

        // Computes a mask of probable foreground/background for the bowl 
        Mat mask;
        detectMainComponents(roi, satRange.first, satRange.second, MIN_AREA_SALAD, mask);
        
        if (countNonZero(mask) > 0) {
            // Segment the food starting from the mask
            grabCutSeg(roi, SALAD, mask);

            // Add the detected segmented region to the tray mask
            Mat(foodsMask, saladROI[i]) += mask;

            // Save the bounding box of the above detected region 
            Rect finalBox = boundingRect(mask);
            finalBox.x += saladROI[i].x;
            finalBox.y += saladROI[i].y;
            trayItems.push_back(pair<Rect,int>(finalBox, SALAD));

            // Draws the overlay for showing the results
            Mat srcROI = Mat(src, saladROI[i]);
            drawMask(srcROI, mask);
        
            ///*                                                                  //TODO: remove
            imshow("res", srcROI);
            waitKey(0);
            //*/
        }
    }    

}


void zw::segmentAndDetectBread(const Mat& src, const vector<Rect>& breadROI, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {

}
    