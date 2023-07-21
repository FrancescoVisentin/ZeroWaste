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
        vector<Mat> histograms;

        Mat m2;
        inRange(src, Scalar(0,0,0), Scalar(0,0,0), m2);
        
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                Rect roi = Rect(a*src.cols/2, b*src.rows/2, src.cols/2, src.rows/2);
                Mat tmp = Mat(src, roi);

                for (int c = 0; c < 2; c++) {
                    Mat mask = Mat::zeros(src.size(), CV_8U);
                    Mat m = Mat(mask, roi);
                    vector<Point> poly;
                    if (a == b) {
                        poly.push_back(Point(0,0));
                        poly.push_back(Point(m.cols,m.rows));
                        if (c == 0) poly.push_back(Point(m.cols,0));
                        else        poly.push_back(Point(0,m.rows));
                    }
                    else {
                        poly.push_back(Point(0,m.rows));
                        poly.push_back(Point(m.cols,0));
                        if (c == 0) poly.push_back(Point(0, 0));
                        else        poly.push_back(Point(m.cols,m.rows));
                    }

                    fillPoly(m, poly, 255);
                    mask -= m2;
                    
                    Mat hist;
                    getHist(tmp, hist, m);

                    normalize(hist, hist, 1, NORM_MINMAX);

                    histograms.push_back(hist);
                }
            }
        }

        cout << "img: " + imgPaths[i] << "\n";
        
        vector<double> dist;
        vector<double> avgs;
        for (int j = 0; j < histograms.size(); j++) {
            double avg = 0;
            for (int k = 0; k < histograms.size(); k++) {
                if (j != k) avg += norm(histograms[j], histograms[k], NORM_L2);
            }
            //cout << "\tavg of " << j << ": " << avg/7 << "\n";
            avgs.push_back(avg/7);
        }
        double avgAVG = 0;
        for (int j = 0; j < avgs.size(); j++) avgAVG += avgs[j];
        cout << "\tavgAVG: " << avgAVG/7 << "\n";


        double maxAVG = *max_element(avgs.begin(), avgs.end());
        if (maxAVG > 0.5) cout << "\tSECONDO! Max avg: " <<  maxAVG << "\n\n";
        else              cout << "\tPRIMO!   Max avg: " <<  maxAVG << "\n\n";


        for (int j = 0; j < 8; j++) {
            double max = 0;
            for (int k = 0; k < 8; k++) {
                double d = norm(histograms[j], histograms[k], NORM_L2);
                if (d > max) max = d;
            }
            //cout << "\tmax of " << j << ": " << max << "\n";
        }

    }

    
    return 0;
}