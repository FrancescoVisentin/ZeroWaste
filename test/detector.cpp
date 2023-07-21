#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void getColHist(Mat& src, Mat& hist, Mat& mask) {
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);

        hist.push_back(tmp);
    }
    hist = hist.reshape(1,1);
}


void getSatHist(Mat& src, Mat& hist, Mat& mask) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    extractChannel(hsv, hsv, 1);

    imshow("sat", hsv);
    waitKey(0);
    
    float range[] = { 30, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    int histSize = 226;
    
    int channel = 0;
    calcHist(&hsv, 1, &channel, mask, hist, 1, &histSize, histRange, true, true);

    hist = hist.reshape(1,1);
    //normalize(hist, hist, 1, NORM_MINMAX);
    //cout<<hist.size()<<" "<<hist<<"\n";
}

void getHist(Mat& src, Mat& hist, Mat& mask) {
    Mat h1;
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);

        h1.push_back(tmp);
    }

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Mat h2;
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            float range[]{0, 180}; //the upper boundary is exclusive
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
}




int main(int argc, char **argv){
    vector<string> imgPaths;
    cv::utils::fs::glob("/home/francesco/Scaricati/hist/first/", "*.jpg", imgPaths, false, false);
    
    for (int i = 0; i < imgPaths.size(); i++) {
        Mat src = imread(imgPaths[i]);
        Mat m2;
        inRange(src, Scalar(0,0,0), Scalar(0,0,0), m2);
        Mat mask = Mat(src.size(), CV_8U, 255);
        mask -= m2;
        
        
        vector<Mat> histograms;
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                Rect roi = Rect(a*src.cols/2, b*src.rows/2, src.cols/2, src.rows/2);
                Mat tmp = Mat(src, roi);
                Mat m = Mat(mask, roi);

                Mat a;
                Mat hist;
                getSatHist(tmp, hist, m);

                normalize(hist, hist, 1, NORM_MINMAX);

                histograms.push_back(hist);

                //imshow("tmp", tmp);
                //waitKey(0);
            }
        }

        cout << "img: " + imgPaths[i] << "\n";
        
        vector<double> avgs;
        vector<double> maxPerZone;
        for (int j = 0; j < 4; j++) {
            double avg = 0;
            vector<double> max;
            for (int k = 0; k < 4; k++) {
                if (j != k) avg += norm(histograms[j], histograms[k], NORM_L2);
                if (j != k) max.push_back(norm(histograms[j], histograms[k], NORM_L2));
            }
            //cout << "\tavg of " << j << ": " << avg/3 << "\n";
            double zoneMax = *max_element(max.begin(), max.end());
            cout << "\tmax of " << j << ": " << zoneMax << "\n";
            avgs.push_back(avg/3);
            maxPerZone.push_back(zoneMax);
        }

        cout << "\tMax MAX "<<*max_element(maxPerZone.begin(), maxPerZone.end())<<"\n";

        double maxAVG = *max_element(avgs.begin(), avgs.end());
        if (maxAVG > 0.5) cout << "\tSECONDO! Max avg: " <<  maxAVG << "\n\n";
        else              cout << "\tPRIMO!   Max avg: " <<  maxAVG << "\n\n";


        for (int j = 0; j < 4; j++) {
            double max = 0;
            for (int k = 0; k < 4; k++) {
                double d = norm(histograms[j], histograms[k], NORM_L2);
                if (d > max) max = d;
            }
            //cout << "\tmax of " << j << ": " << max << "\n";
        }

    }

    
    return 0;
}