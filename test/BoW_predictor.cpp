#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

Ptr<SIFT> sift = SIFT::create();
Ptr<BFMatcher> matcher = BFMatcher::create();
Ptr<SVM> svm = SVM::load("model/model.yml");

void plotHist(const Mat& hist) {
    int histWidth = hist.cols*6;
    int histHeight = 400;
    int bin_w = cvRound((double) histWidth/hist.cols);
    Mat histImage = cv::Mat(histHeight, histWidth, CV_8UC3, cv::Scalar(0,0,0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX);
    
    for( int i = 0; i < hist.cols; i++) {
        cv::line( histImage, 
                  cv::Point(bin_w*(i), histHeight),
                  cv::Point(bin_w*(i), histHeight-cvRound(hist.at<float>(0,i))),
                  cv::Scalar(0,0,255),
                  2,
                  cv::LineTypes::LINE_8,
                  0);
    }

    imshow("hist", histImage);
}

Mat getHistogram(const Mat& desc, const Mat& codewords) {
    vector<DMatch> matches;
    matcher->match(desc, codewords, matches);

    Mat hist = Mat::zeros(1, codewords.rows, CV_32F);
    for (int i = 0; i < matches.size(); i++) {
        hist.at<float>(0, matches[i].trainIdx)++;
    }

    return hist;
}

int predict(string img_path, const Mat& codewords) {
    Mat desc;
    vector<KeyPoint> key;
    sift->detectAndCompute(imread(img_path), noArray(), key, desc);

    Mat hist = getHistogram(desc, codewords);

    normalize(hist, hist, 0, 500, NORM_MINMAX);

    return svm->predict(hist);
}

void predictAndShow(string path, int label, const vector<String> categories, const Mat& codewords) {
    Mat img = imread(path);
    Mat desc;
    vector<KeyPoint> key;
    sift->detectAndCompute(img, noArray(), key, desc);

    vector<DMatch> matches;
    matcher->match(desc, codewords, matches);

    vector<KeyPoint> keyMatched;
    Mat hist = Mat::zeros(1, codewords.rows, CV_32F);
    for (int i = 0; i < matches.size(); i++) {
        hist.at<float>(0, matches[i].trainIdx)++;
        keyMatched.push_back(key[matches[i].queryIdx]);
    }

    normalize(hist, hist, 0, 500, NORM_MINMAX);

    int p = svm->predict(hist);

    drawKeypoints(img, key, img, Scalar(0, 255, 0));
    drawKeypoints(img, keyMatched, img, Scalar(0, 0, 255));

    plotHist(hist);

    imshow("Prediction", img);
    moveWindow("Prediction", 10, 10);
    cout<<"True label: "<<label<<"="<<categories[label]<<" Predicted label: "<<p<<"="<<categories[p]<<"\n";

    waitKey(0);
}

void predictBatch(const vector<string>& testImgsPaths, const vector<int>& testImgsLabels, Mat& codewords,const vector<String> categories) {
    cout<<"Evaluating model predictions on "<<testImgsPaths.size()<<" images...\n";
    int count = 0;
    for (int i = 0; i < testImgsPaths.size(); i++) {
        int label = predict(testImgsPaths[i], codewords);

        if (label != testImgsLabels[i]) count++; 

        cout<<"True label: "<<testImgsLabels[i]<<"="<<categories[testImgsLabels[i]]<<" Predicted label: "<<label<<"="<<categories[label]<<"\n";

    }
    cout<<"Wrong predictions: "<<count<<"/"<<testImgsPaths.size()<<"\n";
}

int main(int argc, char** argv) {
    vector<int> testImgsLabels;
    vector<string> testImgsPaths, categories;
    FileStorage file("model/full_testset.yml", FileStorage::READ);
    file["testPaths"]>>testImgsPaths;
    file["testLabels"]>>testImgsLabels;
    file["categories"]>>categories;

    Mat codewords;
    file = FileStorage("model/codewords.yml", FileStorage::READ);
    file["codewords"]>>codewords;
    file.release();

    if (argc > 1)   predictAndShow(testImgsPaths[atoi(argv[1])], testImgsLabels[atoi(argv[1])], categories, codewords);
    else            predictBatch(testImgsPaths, testImgsLabels, codewords, categories);

    return 0;
}
