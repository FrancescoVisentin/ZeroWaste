#include <classifier.hpp>

using namespace std;
using namespace cv;
using namespace zw;

/*************************************************************************************/
/*                      Utility functions used by the segmentation                   */
/*                        and detection functions defined below                      */
/*                                                                                   */
/*************************************************************************************/

// Computes an histogram on the BGR and HSV color space for the input image considering only the
// values different from zero in the given input mask
void computeHist(const Mat& src, const Mat& mask, Mat& hist) {
    // BGR histograms
    Mat h1;
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 };
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);

        h1.push_back(tmp);
    }

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // HSV histograms
    Mat h2;
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            float range[]{0, 180}; // Hue has range 0-179
            const float* histRange[] = { range };
            int histSize = 180;
            
            Mat tmp;
            calcHist(&hsv, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        else {
            float range[]{0, 256}; //the upper boundary is exclusive
            const float* histRange[] = { range };
            int histSize = 256;
            
            Mat tmp;
            calcHist(&hsv, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
    }

    hist.push_back(h1);
    hist.push_back(h2);

    hist = hist.reshape(1,1);
    normalize(hist, hist, 1, NORM_MINMAX);
}

// For each plate detects if its a first or a second course
// This information is used to help int the plate classification 
bool isFirstCourse(const Mat& src) {
    Mat sat;
    cvtColor(src, sat, COLOR_BGR2HSV);
    extractChannel(sat, sat, 1);

    Mat satMask;
    threshold(sat, satMask, FIRST_COURSE_SAT, 255, THRESH_BINARY);

    vector<Mat> cornersHistograms;
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            Rect roi = Rect(a*src.cols/2, b*src.rows/2, src.cols/2, src.rows/2);
            Mat tmp = Mat(src, roi);
            Mat mask = Mat(satMask, roi);

            Mat hist;
            computeHist(tmp, mask, hist);
            cornersHistograms.push_back(hist);
        }
    }

    vector<double> cornerAVG;
    for (int i = 0; i < 4; i++) {
        double avg = 0;
        for (int j = 0; j < 4; j++) if (i != j) avg += norm(cornersHistograms[i], cornersHistograms[j], NORM_L2);

        cornerAVG.push_back(avg/3);
    }

    double maxAVG = *max_element(cornerAVG.begin(), cornerAVG.end());
    cvtColor(satMask, satMask, COLOR_GRAY2BGR);
    cout<<"\n"<<maxAVG<<"\n";
    imshow("satMask", src&satMask);
    imshow("sas", satMask);
    
    return maxAVG < FIRST_COURSE_MAX_AVG;
}

// Given a plate return an ID describing its content.
// The IDs are consistent with the one defined in classifier.hpp
int classifyPlate(const cv::Mat& src, bool isFirstCourse) {
    string model = (isFirstCourse) ? "../model/modelKnn1.yml" : "../model/modelKnn2.yml";
    Ptr<ml::KNearest> knn = ml::KNearest::load(model);

    Mat hist;
    computeHist(src, Mat(), hist);

    return knn->predict(hist);
}



/*************************************************************************************/
/*                     Definitions of the functions declared                         */
/*                              in classifier.hpp                                    */
/*                                                                                   */
/*************************************************************************************/
void zw::Classifier::classifyAndUpdate(const Mat& src, const vector<Rect>& platesROI, const vector<Mat>& platesMask, vector<vector<int>>& itemPerPlate) {
    if (!initialized) {
        for (int i = 0; i < platesROI.size(); i++) {
            Mat roi = Mat(src, platesROI[i]).clone();
            roi &= platesMask[i];

            int plateID = (isFirstCourse(roi)) ? classifyPlate(roi, true) : classifyPlate(roi, false);
                
            Mat sat;
            cvtColor(roi, sat, COLOR_BGR2HSV);
            extractChannel(sat, sat, 1);

            Mat satMask;
            threshold(sat, satMask, 0, 255, THRESH_BINARY | THRESH_OTSU);

            Mat hist;
            computeHist(roi, satMask, hist);

            trayRefHist.push_back(pair<Mat, int>(hist, plateID));
            itemPerPlate.push_back(platesContent[plateID]);

            initialized = true;
        }
        return;
    }

    // Computes the histograms of the plates in the tray
    vector<Mat> histograms;
    for (int i = 0; i < platesROI.size(); i++) {
        Mat roi = Mat(src, platesROI[i]).clone();
        roi &= platesMask[i];
        
        Mat sat;
        cvtColor(roi, sat, COLOR_BGR2HSV);
        extractChannel(sat, sat, 1);

        Mat satMask;
        threshold(sat, satMask, 0, 255, THRESH_BINARY | THRESH_OTSU);

        Mat hist;
        computeHist(roi, satMask, hist);

        histograms.push_back(hist);
    }

    // For each plate computes the closest reference histogram from the one computed in the previous tray image and assign the same ID to the plate
    // EX: leftover2.jpg will compare the histograms of its plates to the one computed in leftover1.jpg and match each to the closest one to assign the labels
    vector<pair<double, int>> histMinDist;
    for (int i = 0;  i < histograms.size(); i++) {
        vector<double> histDist;
        for (int j = 0; j < trayRefHist.size(); j++) {
            histDist.push_back(norm(histograms[i], trayRefHist[j].first, NORM_L2));
        }

        auto min = min_element(histDist.begin(), histDist.end());
        histMinDist.push_back(pair<double, int>(*min, distance(histDist.begin(), min)));
    }

    // We assume to have at most 2 main corses per tray
    if (histograms.size() == 1) {
        // Update the reference histograms
        int id1 = histMinDist[0].second;
        trayRefHist[id1].first = histograms[0];

        // Assigns a label to the plate
        itemPerPlate.push_back(platesContent[trayRefHist[id1].second]);
    }
    else if (histograms.size() == 2) {
        int id1 = histMinDist[0].second;
        int id2 = histMinDist[1].second;
        double d1 = histMinDist[0].first; 
        double d2 = histMinDist[1].first;
        // Udjust the indices if both plates match to the same hist
        if (id1 == id2) {
            if (d1 < d2) id2 = (id1 == 0) ? 1 : 0;
            else         id1 = (id2 == 0) ? 1 : 0;
        }
        // Update the reference histograms
        trayRefHist[id1].first = histograms[0];
        trayRefHist[id2].first = histograms[1];

        // Assigns a label to each plate
        itemPerPlate.push_back(platesContent[trayRefHist[id1].second]);
        itemPerPlate.push_back(platesContent[trayRefHist[id2].second]);
    }
}