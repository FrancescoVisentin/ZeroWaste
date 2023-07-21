#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void getHistOLD(Mat& src, Mat& hist) {
    Mat sat;
    cvtColor(src, sat, COLOR_BGR2HSV);
    extractChannel(sat, sat, 1);

    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    int histSize = 256;
    bool uniform = true, accumulate = false;
    calcHist(&sat, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
    hist = hist.reshape(1,1);

    for (int i = 0; i < 3; i++) {
        Mat tmp;
        calcHist(&src, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, uniform, accumulate);
        tmp = tmp.reshape(1, 1);
       
        hist.push_back(tmp);
    }
    hist = hist.reshape(1,1);
}

void getHist(Mat& src, Mat& hist) {
    Mat h1;
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, true, true);
    
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
            calcHist(&hsv, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        else {
            float range[]{0, 256}; //the upper boundary is exclusive
            const float* histRange[] = { range };
            int histSize = 256;
            
            Mat tmp;
            calcHist(&hsv, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        
    }

    hist.push_back(h1);
    hist.push_back(h2);

    hist = hist.reshape(1,1);
}


#include <opencv2/core/utils/filesystem.hpp>


int main(int argc, char **argv){
    vector<string> dirPaths;
    cv::utils::fs::glob("/home/francesco/Scaricati/hist_main/second_course/", "*_*", dirPaths, false, true);
    
    std::sort(dirPaths.begin(), dirPaths.end());

    Mat train, labels;
    for (int j = 0; j < dirPaths.size(); j++) {
        cout<<dirPaths[j]<<"\n";

        vector<string> paths;
        glob(dirPaths[j]+"/*.jpg", paths, true);

        for (int i = 0; i < paths.size(); i++) {
            cout<<"\t"+paths[i]<<"\n";
        }
    
        Mat tot = Mat::zeros(1, 1460, CV_32FC1);
        for(int i = 0; i < paths.size(); i++) {
            Mat src = imread(paths[i]);

            Mat hist;
            getHist(src, hist);

            normalize(hist, hist, 1, NORM_MINMAX);

            tot += hist;
        }
        tot /= paths.size();


        labels.push_back(Mat(1,1,CV_32SC1, j+6));
        train.push_back(tot);
    }

    Ptr<KNearest> knn = KNearest::create();

    knn->setIsClassifier(true);
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->setDefaultK(1);
    Ptr<TrainData> td = TrainData::create(train, ROW_SAMPLE, labels);
    
    knn->train(td);
    
    knn->save("/home/francesco/Scaricati/hist_main/second_course/modelKnn2.yml");

    return 0;
}