#include <classifier.hpp>

using namespace std;
using namespace cv;
using namespace zw;

const Ptr<ml::KNearest> knn = ml::KNearest::load("../model/modelKnn.yml");

Mat getHist(const Mat& src) {
    Mat hist;
    for (int i = 0; i < 3; i++) {
        Mat tmp;
        calcHist(src, vector<int>{i}, noArray(), tmp, vector<int>{256}, vector<float>{0,256});

        hist.push_back(tmp);
    }

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Mat h2;
    for (int i = 0; i < 3; i++) {
        Mat tmp;
        if (i == 0) calcHist(src, vector<int>{i}, noArray(), tmp, vector<int>{180}, vector<float>{0,180}); //hue has a range in 0-179
        else        calcHist(src, vector<int>{i}, noArray(), tmp, vector<int>{256}, vector<float>{0,256});
        
        hist.push_back(tmp);
    }

    hist = hist.reshape(1,1);
    normalize(hist, hist, 1, NORM_MINMAX);

    return hist;
}



 void zw::detect(const Mat& src, vector<int>& cat) {
    Mat hist = getHist(src);

    int label = knn->predict(hist);

    cat = platesID[label];
}