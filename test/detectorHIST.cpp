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

    //imshow("sat", hsv);
    //waitKey(0);
    
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    int histSize = 256;
    
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



vector<int> match(vector<Mat>& hists, vector<Mat>& labels) {
    vector<int> ids;
    if (labels.size() == 0) {
        for (int i = 0; i < hists.size(); i++) {
            labels.push_back(hists[i]);
            ids.push_back(i);
        }

        return ids;
    }

    vector<pair<double,int>> dist;
    for (int i = 0;  i < hists.size(); i++) {
        
        vector<double> histDist;
        for (int j = 0; j < labels.size(); j++) {
            histDist.push_back(norm(hists[i], labels[j], NORM_L2));
        }

        auto min = min_element(histDist.begin(), histDist.end());
        dist.push_back(pair<double, int>(*min, distance(histDist.begin(), min)));
    }

    if (dist.size() == 1) {
        cout<<"size 1\n";
        labels[dist[0].second] = dist[0].first;
        ids.push_back(dist[0].second);
        
        return ids;
    }

    if (dist[0].second != dist[1].second) {
        cout<<"buono\n";
        labels[dist[0].second] = hists[dist[0].second];
        labels[dist[1].second] = hists[dist[1].second];

        ids.push_back(dist[0].second);
        ids.push_back(dist[1].second);
    }
    else if (dist[0].first < dist[1].first) {
        if (dist[0].second == 0) {
            cout<<"0 < 1 and id = 0\n";
            labels[0] = hists[0];
            labels[1] = hists[1];

            ids.push_back(0);
            ids.push_back(1);
        }
        else {
            cout<<"0 < 1 and id = 1\n";
            labels[0] = hists[1];
            labels[1] = hists[0];

            ids.push_back(1);
            ids.push_back(0);
        }
    }
    else { // dist[1].first < dist[0].first
        if (dist[1].second == 1) {
            cout<<"1 < 0 and id = 1\n";
            labels[0] = hists[0];
            labels[1] = hists[1];

            ids.push_back(0);
            ids.push_back(1);
        }
        else {
            cout<<"1 < 0 and id = 0\n";
            labels[0] = hists[1];
            labels[1] = hists[0];

            ids.push_back(1);
            ids.push_back(0);
        }
    }

    return ids;

}



int main(int argc, char **argv){
    vector<int> ids;
    vector<Mat> labels;
    for (int i = 0; i < 4; i++) {
        vector<string> imgPaths;
        string p = "/home/francesco/Scaricati/detector/first/tray"+to_string(atoi(argv[1]))+"/";
        cv::utils::fs::glob(p, "tray?_img"+to_string(i)+"*.jpg", imgPaths, false, false);

        vector<Mat> hists;
        for (int j = 0; j < imgPaths.size(); j++) {
            Mat src = imread(imgPaths[j]);
            
            Mat sat;
            cvtColor(src, sat, COLOR_BGR2HSV);
            extractChannel(sat, sat, 1);

            Mat satMask;
            threshold(sat, satMask, 120, 255, THRESH_BINARY | THRESH_OTSU);

            Mat dif;
            threshold(sat, dif, 120, 255, THRESH_BINARY_INV | THRESH_OTSU);
            cvtColor(dif, dif, COLOR_GRAY2BGR);

            Mat hist;
            getHist(src, hist, satMask);
            normalize(hist, hist, 1, NORM_MINMAX);

            hists.push_back(hist);

            //imshow("src", src);
            //imshow("mask", src-dif);
            //waitKey(0);
        }

        vector<int> res = match(hists, labels);
        for (int j = 0; j < hists.size(); j++) {
            cout << "Img: " << imgPaths[j] << " label: " << res[j] << "\n";
        }

    }

    
    return 0;
}