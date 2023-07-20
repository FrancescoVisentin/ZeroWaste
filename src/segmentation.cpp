#include <segmentation.hpp>
#include <segmentation_utils.hpp>

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
        approxPolyDP(Mat(contours[i]), contoursPoly, 3, true);
        
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
            else {
                // Remove contour from satMask
                drawContours(satMask, contours, i, 0, -1);
            }
        }

        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                if (satMask.at<u_char>(i,j) && mask.at<u_char>(i,j) == GC_PR_BGD) mask.at<u_char>(i,j) = GC_PR_FGD; 
            }
        }

        // Uncomment to view the mask and the contours
        //imshow("Contours", dst);
        //imshow("Mask", mask*30);
    }   
}


// Applies the cv::grubCut segmentation algorithm inside the area defined by the rect given in input 
void grabCutSeg(const Mat& src, int id, Mat& mask) {
    Mat bgdModel, fgdModel;
    grabCut(src, mask, Rect(), bgdModel, fgdModel, 5, GC_INIT_WITH_MASK);
   
    // Assigns to the pixels in the foreground the given food id
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<u_char>(i,j) == GC_FGD || mask.at<u_char>(i,j) == GC_PR_FGD) mask.at<u_char>(i,j) = id;
            else mask.at<u_char>(i,j) = 0;
        }
    }
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
    zw_utils::filterCircles(circles, filtered_circles);

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


void zw::segmentAndDetectPlates(Mat& src, const vector<Rect>& platesROI, const vector<Mat>& platesMask, Classifier& cf, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {
    vector<vector<int>> itemPerPlate;
    cf.classifyAndUpdate(src, platesROI, platesMask, itemPerPlate);

    for (int i = 0; i < platesROI.size(); i++) {
        Mat roi = Mat(src, platesROI[i]).clone();
        roi &= platesMask[i];

        Mat segmentedRegions = Mat::zeros(roi.size(), CV_8U);
        vector<int> plateItemsIDs = itemPerPlate[i];
        for (int j = 0; j < plateItemsIDs.size(); j ++) {
            // Get the saturation range for the current plate item
            pair<int, int> satRange = saturationRange[plateItemsIDs[j]];

            // Computes a mask of probable foreground/background for the current food 
            Mat mask;
            detectMainComponents(roi, satRange.first, satRange.second, MIN_AREA_PLATES, mask);
            
            // Avoids overlaps with already segmented regions
            mask -= segmentedRegions;
            
            int maskArea = countNonZero(mask);
            if (maskArea > 0 && maskArea < roi.rows*roi.cols) {
                // Segment the food starting from the mask
                grabCutSeg(roi, plateItemsIDs[j], mask);

                int detectedArea = countNonZero(mask);
                if (detectedArea > MIN_DETECTED_AREA) {
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
                    segmentedRegions += mask;
                    cvtColor(mask, mask, COLOR_GRAY2BGR);
                    roi -= mask;
                
                    // Uncomment to view the plate item segmentation
                    //imshow("tmp", roi);
                    //imshow("res", srcROI);
                    //waitKey(0);
                }
            }
        }
    }
}


void zw::segmentSalad(Mat& src, const vector<Rect>& saladROI, const vector<Mat>& saladMask, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {
    for (int i = 0; i < saladROI.size(); i++) {
        Mat roi = Mat(src, saladROI[i]).clone();
        roi &= saladMask[i];

        // Get the salad saturation range
        pair<int, int> satRange = saturationRange[SALAD]; 

        // Computes a mask of probable foreground/background for the bowl 
        Mat mask;
        detectMainComponents(roi, satRange.first, satRange.second, MIN_AREA_SALAD, mask);
        
        int maskArea = countNonZero(mask);
        if (maskArea > 0 && maskArea < roi.rows*roi.cols) {
            // Segment the food starting from the mask
            grabCutSeg(roi, SALAD, mask);

            int detectedArea = countNonZero(mask);
            if (detectedArea > MIN_DETECTED_AREA) {
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
            
                // Uncomment to view the salad segmentation
                //imshow("res", srcROI);
                //waitKey(0);
            }
        }
    }    

}


void zw::segmentAndDetectBread(Mat& src, const Mat& roiMask, Mat& foodsMask, vector<pair<Rect,int>>& trayItems) {
    Mat blurred = src-roiMask;
    blur(blurred, blurred, Size(11,11));

    // First filtering based on a saturation range, will identify some false positives
    Mat mask;
    pair<int, int> satRange = saturationRange[BREAD];
    detectMainComponents(blurred, satRange.first, satRange.second, MIN_AREA_BREAD, mask);

    vector<vector<Point>> contours;
    findContours(mask, contours, noArray(), RETR_TREE, CHAIN_APPROX_NONE);

    // Region of interest of the zones that may contain the bread
    vector<Rect> contoursRect;
    for (int i = 0; i < contours.size(); i++) {
        vector<Point> contoursPoly;
        approxPolyDP(Mat(contours[i]), contoursPoly, 3, true);
        
        // Filters the detected zones based on their shape (bread shape is usually similar to the a square)
        Rect r = boundingRect(contoursPoly);
        if (r.width > r.height  && 1.*r.height/r.width > BREAD_AREA_THRESHOLD) contoursRect.push_back(r);
        if (r.width <= r.height  && 1.*r.width/r.height > BREAD_AREA_THRESHOLD) contoursRect.push_back(r);
    }

    // Filter the zones considering their dominant colors extracted with kmeans and build a segmentation mask accordingly
    Mat segMask = Mat::zeros(src.size(), CV_8U);
    for (int k = 0; k < contoursRect.size(); k++) {
        Mat roi = Mat(blurred, contoursRect[k]);

        Mat features = Mat(roi.rows*roi.cols, 3, CV_32F);
        for (int i = 0; i < roi.rows; i++)
            for (int j = 0; j < roi.cols; j++)
                for (int z = 0; z < 3; z++)
                    features.at<float>(i*roi.cols + j, z) = roi.at<Vec3b>(i,j)[z];

        // Find the 3 most dominant colors in the image by clustering with kmeans
        Mat labels, colors;
        TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER, 15, 1.0);
        kmeans(features, 3, labels, criteria, 5, KMEANS_PP_CENTERS, colors);
        colors.convertTo(colors, CV_8UC3);
        
        // Bread reference color
        array<Vec3b,3> ref = dishRefColors[10];
        
        // Distance with respect to the current triplet of ref colors 
        double colorDist = 0;
        for (int j = 0; j < 3; j++) {
            Vec3b c = colors.at<Vec3b>(j);
            // Given a dominant color consider only its distance from the closest ref color for the current dish type 
            double minDist = 10000;
            for (int k = 0; k < 3; k++) {
                double tmp = norm(c, ref[k], NORM_L2);
                if (tmp < minDist) minDist = tmp;
            }
            colorDist += minDist;
        }
        
        if (colorDist < BREAD_COLOR_THRESHOLD) {
            Mat(mask, contoursRect[k]).copyTo(segMask(contoursRect[k]));
        }
    }
    
    int maskArea = countNonZero(segMask);
    if (maskArea > 0 && maskArea < src.rows*src.cols) {
        // Segment the food starting from the mask
        grabCutSeg(blurred, BREAD, segMask);
        
        int detectedArea = countNonZero(segMask);
        if (detectedArea > MIN_DETECTED_AREA) {
            // Add the detected segmented region to the tray mask
            foodsMask += segMask;

            // Save the bounding box of the above detected region 
            Rect finalBox = boundingRect(segMask);
            trayItems.push_back(pair<Rect,int>(finalBox, BREAD));

            // Draws the overlay for showing the results
            drawMask(src, segMask);
        
            
            // Uncomment to view the bread segmentation
            //imshow("res", src);
            //waitKey(0);
        }
    }
}
    