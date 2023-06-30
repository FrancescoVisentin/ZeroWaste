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
    int histWidth = hist.cols*3;
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

    cv::line(histImage, Point(bin_w*200, histHeight), Point(bin_w*200, 0), Scalar(255,255,255));
    cv::line(histImage, Point(bin_w*400, histHeight), Point(bin_w*400, 0), Scalar(255,255,255));

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

int predict(string img_path, const array<Mat,3> codewords) {
    Mat bgr[3];
    split(imread(img_path), bgr);
    
    Mat hist;
    for (int i = 0; i < 3; i++) {
        Mat desc;
        vector<KeyPoint> key;
        sift->detectAndCompute(bgr[i], noArray(), key, desc);

        Mat channelHist = getHistogram(desc, codewords[i]);
        hist.push_back(channelHist);
    }
    hist = hist.reshape(1, 1);
    normalize(hist, hist, 0, 500, NORM_MINMAX);

    return svm->predict(hist);
}

void predictAndShow(string path, int label, const vector<String> categories, const array<Mat,3> codewords) {
    Mat img = imread(path);
    
    Mat bgr[3];
    split(img, bgr);

    vector<Scalar> colors = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255)}; 

    Mat hist;
    for (int i = 0; i < 3; i++) {
        Mat desc;
        vector<KeyPoint> key;
        sift->detectAndCompute(bgr[i], noArray(), key, desc);

        vector<DMatch> matches;
        matcher->match(desc, codewords[i], matches);

        vector<KeyPoint> keyMatched;
        Mat channelHist = Mat::zeros(1, codewords[i].rows, CV_32F);
        for (int i = 0; i < matches.size(); i++) {
            channelHist.at<float>(0, matches[i].trainIdx)++;
            keyMatched.push_back(key[matches[i].queryIdx]);
        }

        drawKeypoints(img, keyMatched, img, colors[i]);

        hist.push_back(channelHist);
    }
    hist = hist.reshape(1, 1);
    normalize(hist, hist, 0, 500, NORM_MINMAX);

    int p = svm->predict(hist);

    plotHist(hist);

    imshow("Prediction", img);
    moveWindow("Prediction", 10, 10);
    cout<<"True label: "<<label<<"="<<categories[label]<<" Predicted label: "<<p<<"="<<categories[p]<<"\n";

    waitKey(0);
}

void predictBatch(const vector<string>& testImgsPaths, const vector<int>& testImgsLabels, array<Mat,3> codewords,const vector<String> categories) {
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
    /*
    Mat pesto = imread("data/1_pasta_pesto/tray_1_food_image.jpg");
    Mat tomato = imread("data/2_pasta_tomato/tray_2_food_image.jpg");

    Mat bgr1[3];
    Mat bgr2[3];
    split(pesto, bgr1);
    split(tomato, bgr2);

    imshow("pesto", imread("data/1_pasta_pesto/tray_1_food_image.jpg", IMREAD_GRAYSCALE));
    imshow("tomato", imread("data/2_pasta_tomato/tray_2_food_image.jpg", IMREAD_GRAYSCALE));
    
    imshow("pesto B", bgr1[0]);
    imshow("pesto G", bgr1[1]);
    imshow("pesto R", bgr1[2]);

    imshow("tomato B", bgr2[0]);
    imshow("tomato G", bgr2[1]);
    imshow("tomato R", bgr2[2]);

    waitKey(0);
    return 0;
    */
   
    vector<int> testImgsLabels;
    vector<string> testImgsPaths, categories;
    FileStorage file("model/full_testset.yml", FileStorage::READ);
    file["testPaths"]>>testImgsPaths;
    file["testLabels"]>>testImgsLabels;
    file["categories"]>>categories;

    array<Mat,3> codewords;
    file = FileStorage("model/codewords.yml", FileStorage::READ);
    file["codewords_0"]>>codewords[0];
    file["codewords_1"]>>codewords[1];
    file["codewords_2"]>>codewords[2];
    file.release();

    if (argc > 1)   predictAndShow(testImgsPaths[atoi(argv[1])], testImgsLabels[atoi(argv[1])], categories, codewords);
    else            predictBatch(testImgsPaths, testImgsLabels, codewords, categories);

    return 0;
}
