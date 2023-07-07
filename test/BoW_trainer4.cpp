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

int N_CODEWORDS = 200;

void loadData(string path, vector<string>& categories, vector<string>& imgPaths, Mat& labels) {
    vector<string> directories;
    for(auto& p : filesystem::directory_iterator(path)) {
        if (p.is_directory()) directories.push_back(p.path());
    }
    
    std::sort(directories.begin(), directories.end());

    for(int i = 0; i < directories.size(); i++) {
        vector<string> cat_paths;
        glob(directories[i], cat_paths);
        for(int j = 0; j < cat_paths.size(); j++) {
            imgPaths.push_back(cat_paths[j]);
            labels.push_back(Mat(1,1,CV_32SC1, i));
        }
        
        categories.push_back(directories[i].erase(0, path.size()));
    }
}

void getTrainingData(int channel, const vector<string>& imgPaths, vector<Mat>& trainHist, Mat& codewords) {
    Ptr<SIFT> sift = SIFT::create();
    Ptr<BFMatcher> matcher = BFMatcher::create();

    Mat descriptors;
    vector<Mat> descPerImg;
    for (int i = 0; i < imgPaths.size(); i++) {
        Mat img = imread(imgPaths[i]);
        extractChannel(img, img, channel);

        Mat desc;
        vector<KeyPoint> key;
        sift->detectAndCompute(img, noArray(), key, desc);

        descPerImg.push_back(desc);
        descriptors.push_back(desc);
    }


    Mat label;
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 2000, 1.0);
    kmeans(descriptors, N_CODEWORDS, label, criteria, 3, KMEANS_PP_CENTERS, codewords);

    trainHist = vector<Mat>(imgPaths.size());
    for (int i = 0; i < imgPaths.size(); i++) {
        vector<DMatch> matches;
        matcher->match(descPerImg[i], codewords, matches);

        Mat hist = Mat::zeros(1, codewords.rows, CV_32F);
        for (int i = 0; i < matches.size(); i++) {
            hist.at<float>(0, matches[i].trainIdx)++;
        }

        trainHist[i] = hist;
    }
}

void getJointHistograms(const vector<vector<Mat>>& histPerChannel, Mat& trainHistograms) {
    for (int i = 0; i < histPerChannel[0].size(); i++) {
        Mat hist;
        for (int j = 0; j < 3; j++) {
            hist.push_back(histPerChannel[j][i]);
        }
        hist = hist.reshape(1, 1);
        normalize(hist, hist, 0, 500, NORM_MINMAX);

        trainHistograms.push_back(hist);
    }
}

void trainBoW(const Mat& descPerImg, const Mat& labelPerImg) {
    Ptr<SVM> svm = SVM::create();

    svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 1e-6));
    Ptr<TrainData> td = TrainData::create(descPerImg, ROW_SAMPLE, labelPerImg);
	
    svm->train(td);
    //svm->trainAuto(td, 4);

    svm->save("model/model.yml");
}

int main(int argc, char** argv) {
    string basePath = "/home/francesco/Scaricati/data8/";

    Mat trainImgsLabels;
    vector<string> categories, trainImgsPaths;
    loadData(basePath, categories, trainImgsPaths, trainImgsLabels);

    auto start = chrono::steady_clock::now();

    /*
    vector<thread> threads;
    vector<Mat> codewords(3);
    vector<vector<Mat>> histPerChannel(3);
    for (int i = 0; i < 3; i++) {
        threads.push_back(thread(getTrainingData, i, ref(trainImgsPaths), ref(histPerChannel[i]), ref(codewords[i])));
    }
    
    for (auto& th : threads) {
        th.join();
    }
    */

    vector<Mat> codewords(3);
    vector<vector<Mat>> histPerChannel(3);
    for (int i = 0; i < 3; i++) {
        getTrainingData(i, trainImgsPaths, histPerChannel[i], codewords[i]);
    }
    

    auto end = chrono::steady_clock::now();
    cout<< chrono::duration_cast<chrono::minutes>(end-start).count() << " minutes\n";

    Mat trainHistograms;
    getJointHistograms(histPerChannel, trainHistograms);
    
    trainBoW(trainHistograms, trainImgsLabels);

    FileStorage file("model/codewords.yml", FileStorage::WRITE);
    for (int i = 0; i < 3; i++) {
        file <<"codewords_"+to_string(i)<<codewords[i];
    }
    file.release();
    
    return 0;
}