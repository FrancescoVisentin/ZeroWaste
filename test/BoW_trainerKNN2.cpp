#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <thread>

using namespace std;
using namespace cv;
using namespace cv::ml;


Ptr<SIFT> sift = SIFT::create();
Ptr<BFMatcher> matcher = BFMatcher::create();

Mat getHist(const Mat& desc, const Mat& codewords) {
    vector<DMatch> matches;
    matcher->match(desc, codewords, matches);

    Mat hist = Mat::zeros(1, codewords.rows, CV_32F);
    for (int i = 0; i < matches.size(); i++) {
        hist.at<float>(0, matches[i].trainIdx)++;
    }

    return hist;
}

void getHistograms(const vector<string>& testImgsPaths, const array<Mat,3> codewords, vector<Mat>& testHist) {
    for (int i = 0; i < testImgsPaths.size(); i++) {
        Mat bgr[3];
        split(imread(testImgsPaths[i]), bgr);
    
        Mat hist;
        for (int i = 0; i < 3; i++) {
            Mat desc;
            vector<KeyPoint> key;
            sift->detectAndCompute(bgr[i], noArray(), key, desc);

            Mat channelHist = getHist(desc, codewords[i]);
            hist.push_back(channelHist);
        }
        hist = hist.reshape(1, 1);

        testHist.push_back(hist);
    }
}


void getJointNORM(const vector<vector<Mat>>& histPerChannel, Mat& trainHistograms) {
    for (int i = 0; i < histPerChannel[0].size(); i++) {
        Mat hist;
        for (int j = 0; j < 3; j++) {
            hist.push_back(histPerChannel[j][i]);
        }
        hist = hist.reshape(1, 1);
        normalize(hist, hist, 500, NORM_MINMAX);

        trainHistograms.push_back(hist);
    }
}

void getJointIFDF(const vector<vector<Mat>>& histPerChannel, Mat& trainHistograms) {
    vector<Mat> histograms;
    Mat n_i = Mat(1, histPerChannel[0][0].cols*3, CV_32F, 1);
    for (int i = 0; i < histPerChannel[0].size(); i++) {
        Mat hist;
        for (int j = 0; j < 3; j++) {
            hist.push_back(histPerChannel[j][i]);
        }
        hist = hist.reshape(1, 1);

        Mat tmp;
        threshold(hist, tmp, 1, 1, THRESH_BINARY);
        n_i += tmp;

        histograms.push_back(hist);
    }

    Mat N = Mat(histograms[0].size(), CV_32F, histograms.size()+1);
    Mat idf = N.mul(1/n_i);
    log(idf, idf);

    for (int i = 0; i < histograms.size(); i++) {
        double n_d = sum(histograms[i])[0];
        histograms[i] = histograms[i].mul(idf/n_d);

        normalize(histograms[i], histograms[i], 0, 1, NORM_MINMAX);
    
        trainHistograms.push_back(histograms[i]);
    }
}

void trainKNN(string model, const Mat& descPerImg, const Mat& labelPerImg) {
    Ptr<KNearest> knn = KNearest::create();

    knn->setIsClassifier(true);
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->setDefaultK(1);
    Ptr<TrainData> td = TrainData::create(descPerImg, ROW_SAMPLE, labelPerImg);
    
    knn->train(td);
    
    knn->save(model);
}

void trainSVM(string model, const Mat& descPerImg, const Mat& labelPerImg) {
    Ptr<SVM> svm = SVM::create();

    svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY);
    svm->setDegree(2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 1e-6));
    Ptr<TrainData> td = TrainData::create(descPerImg, ROW_SAMPLE, labelPerImg);

    Ptr<ParamGrid> gammaGrid = makePtr<ParamGrid>(SVM::getDefaultGrid(SVM::GAMMA));
    Ptr<ParamGrid> coef0Grid = makePtr<ParamGrid>(SVM::getDefaultGrid(SVM::COEF));
    Ptr<ParamGrid> CGrid     = makePtr<ParamGrid>(SVM::getDefaultGrid(SVM::C));

    svm->trainAuto(td, 5, *gammaGrid, *coef0Grid, cv::ml::ParamGrid(), *CGrid);

    svm->save(model);
}


int main(int argc, char** argv) {
    Mat trainImgsLabels;
    vector<vector<Mat>> histPerChannel(3);
    FileStorage file("model/backup.yml", FileStorage::READ);
    file["hist_0"]>>histPerChannel[0];
    file["hist_1"]>>histPerChannel[1];
    file["hist_2"]>>histPerChannel[2];
    file["labels"]>>trainImgsLabels;
    file.release();

    Mat trainNORM;
    getJointNORM(histPerChannel, trainNORM);
    trainKNN("model/model_knn_norm.yml", trainNORM, trainImgsLabels);
    trainSVM("model/model_svm_norm.yml", trainNORM, trainImgsLabels);
    
    Mat trainIFDF;
    getJointIFDF(histPerChannel, trainIFDF);
    trainKNN("model/model_knn_ifdf.yml", trainIFDF, trainImgsLabels);
    trainSVM("model/model_svm_ifdf.yml", trainIFDF, trainImgsLabels);
    
    cout<<"TRAINED!\n";

    vector<string> testImgsPaths;
    file = FileStorage("model/full_testset.yml", FileStorage::READ);
    file["testPaths"]>>testImgsPaths;
    file.release();

    if (true) {    
        array<Mat,3> codewords;
        file = FileStorage("model/codewords.yml", FileStorage::READ);
        file["codewords_0"]>>codewords[0];
        file["codewords_1"]>>codewords[1];
        file["codewords_2"]>>codewords[2];
        file.release();

        vector<Mat> testHist;
        getHistograms(testImgsPaths, codewords, testHist);

        file = FileStorage("model/testHist.yml", FileStorage::WRITE);
        file<<"testHist"<<testHist;
        file.release();

        cout<<"DONE!\n";    
    }
    
    return 0;
}