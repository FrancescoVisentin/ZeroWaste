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


Mat getHistogram(const Mat& desc, const Mat& codewords) {
    vector<vector<DMatch>> matches;
    matcher->knnMatch(desc, codewords, matches, 2);

    vector<DMatch> goodMatches;
    for (vector<DMatch> m : matches) {
        if (m[0].distance < 0.75*m[1].distance)
            goodMatches.push_back(m[0]);
    }

    Mat hist = Mat::zeros(1, codewords.rows, CV_32F);
    for (int i = 0; i < goodMatches.size(); i++) {
        hist.at<float>(0, goodMatches[i].trainIdx)++;
    }

    return hist;
}

int predict(string img_path, const Mat& codewords) {
    Mat desc;
    vector<KeyPoint> key;
    sift->detectAndCompute(imread(img_path), noArray(), key, desc);

    Mat hist = getHistogram(desc, codewords);

    return svm->predict(hist);
}

void predictAndShow(string path, int label, const vector<String> categories, const Mat& codewords) {
    Mat img = imread(path);
    Mat desc;
    vector<KeyPoint> key;
    sift->detectAndCompute(img, noArray(), key, desc);

    vector<vector<DMatch>> matches;
    matcher->knnMatch(desc, codewords, matches, 2);

    vector<DMatch> goodMatches;
    for (vector<DMatch> m : matches) {
        if (m[0].distance < 0.75*m[1].distance)
            goodMatches.push_back(m[0]);
    }

    Mat hist = Mat::zeros(1, codewords.rows, CV_32F);
    for (int i = 0; i < goodMatches.size(); i++) {
        hist.at<float>(0, goodMatches[i].trainIdx)++;
    }

    int p = svm->predict(hist);

    vector<KeyPoint> goodKey;
    for (DMatch m : goodMatches) {
        goodKey.push_back(key[m.queryIdx]);
    }

    drawKeypoints(img, key, img, Scalar(0, 255, 0));
    drawKeypoints(img, goodKey, img, Scalar(0, 0, 255));

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
    FileStorage file("model/testset.yml", FileStorage::READ);
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
