#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ml;

void getJointNORM(const vector<Mat>& testHist, vector<Mat>& testHistNorm) {
    for (int i = 0; i < testHist.size(); i++) {
        Mat hist = testHist[i].clone();
        normalize(hist, hist, 1, NORM_MINMAX);

        testHistNorm.push_back(hist);
    }
}

void getJointIFDF(const vector<Mat>& testHist, const Mat& idf, vector<Mat>& testHistIFDF) {
    for (int i = 0; i < testHist.size(); i++) {
        Mat hist = testHist[i].clone();
        double n_d = sum(hist)[0];
        hist = hist.mul(idf/n_d);
        normalize(hist, hist, 0, 1, NORM_MINMAX);

        testHistIFDF.push_back(hist);
    }
}


void predictSVM(string model, const vector<Mat>& testHist, const vector<int>& testImgsLabels, const vector<string> categories) {
    cout<<"\nEvaluating model predictions on "<<testHist.size()<<" images...\n";
    
    int count = 0;
    map<int, int> missPerCategory;
    map<int, int> countPerCategory;
    Ptr<SVM> svm = SVM::load(model);

    for (int i = 0; i < testHist.size(); i++) {
        int label = svm->predict(testHist[i]);

        countPerCategory[testImgsLabels[i]]++;
        if (label != testImgsLabels[i]) {
            missPerCategory[testImgsLabels[i]]++;
            count++;
        } 

        //cout<<"True label: "<<categories[testImgsLabels[i]]<<" Predicted label: "<<categories[label]<<"\n";
    }
    
    cout<<"Wrong predictions: "<<count<<"/"<<testHist.size()<<"\n";
    for (int i = 0; i < categories.size(); i++) {
        cout<<categories[i]<<": "<<missPerCategory[i]<<"/"<<countPerCategory[i]<<"\n";
    }
}

void predictKNN(string model, const vector<Mat>& testHist, const vector<int>& testImgsLabels, const vector<string> categories) {
    cout<<"Evaluating model predictions on "<<testHist.size()<<" images...\n";
    
    int count = 0;
    Mat neigh, dist;
    map<int, int> missPerCategory;
    map<int, int> countPerCategory;
    Ptr<KNearest> knn = KNearest::load(model);

    for (int i = 0; i < testHist.size(); i++) {
        Mat res;
        knn->findNearest(testHist[i], 10, res, neigh, dist);

        map<float,int> m;
        for (int i = 0; i < 10; i++) {
            if (dist.at<float>(0)*1.05 < dist.at<float>(i)) {
                break;
            }
            m[neigh.at<float>(i)]++;
        }

        float p = neigh.at<float>(0);
        int c = m[p];
        for (auto t : m) {
            if (t.second > c) {
                p = t.first; 
                c = t.second;

                //cout<<"pippo!\n";
                }
        }

        int label = p;
        countPerCategory[testImgsLabels[i]]++;
        if (label != testImgsLabels[i]) {
            missPerCategory[testImgsLabels[i]]++;
            count++;
        } 

        //cout<<"True label: "<<categories[testImgsLabels[i]]<<" Predicted label: "<<categories[label]<<"\n"<<neigh<<"\n"<<dist<<"\n\n";
    }
    
    cout<<"Wrong predictions: "<<count<<"/"<<testHist.size()<<"\n";
    for (int i = 0; i < categories.size(); i++) {
        cout<<categories[i]<<": "<<missPerCategory[i]<<"/"<<countPerCategory[i]<<"\n";
    }
}


int main(int argc, char** argv) {
    vector<int> testImgsLabels;
    vector<string> testImgsPaths, categories;
    FileStorage file("model/full_testset.yml", FileStorage::READ);
    file["testPaths"]>>testImgsPaths;
    file["testLabels"]>>testImgsLabels;
    file["categories"]>>categories;

    Mat idf;
    array<Mat,3> codewords;
    file = FileStorage("model/codewords.yml", FileStorage::READ);
    file["codewords_0"]>>codewords[0];
    file["codewords_1"]>>codewords[1];
    file["codewords_2"]>>codewords[2];
    file["idf"]>>idf;
    file.release();

    vector<Mat> testHist;
    file =  FileStorage("model/testHist.yml", FileStorage::READ);
    file["testHist"]>>testHist;

    vector<Mat> testHistNorm;
    getJointNORM(testHist, testHistNorm);

    vector<Mat> testHistIFDF;
    getJointIFDF(testHist, idf, testHistIFDF);

    predictSVM("model/model_svm_norm.yml", testHistNorm, testImgsLabels, categories);
    predictSVM("model/model_svm_ifdf.yml", testHistIFDF, testImgsLabels, categories);
    
    predictKNN("model/model_knn_norm.yml", testHistNorm, testImgsLabels, categories);
    predictKNN("model/model_knn_ifdf.yml", testHistIFDF, testImgsLabels, categories);
    
    return 0;
}