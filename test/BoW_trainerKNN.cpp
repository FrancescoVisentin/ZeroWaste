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

int N_CODEWORDS = 350;

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

    FileStorage file("model/backup.yml", FileStorage::WRITE);
    file<<"labels"<<labels;
    file.release();
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
    
    FileStorage file("model/backup.yml", FileStorage::APPEND);
    file<<"hist_"+to_string(channel)<<trainHist;
    file.release();
}

void getJointHistograms(const vector<vector<Mat>>& histPerChannel, Mat& trainHistograms) {
    vector<Mat> histograms;
    Mat n_i = Mat(1, N_CODEWORDS*3, CV_32F, 1);
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
    cout<<n_i<<"\n";
    cout<<idf<<"\n";
    log(idf, idf);

    for (int i = 0; i < histograms.size(); i++) {
        double n_d = sum(histograms[i])[0];
        histograms[i] = histograms[i].mul(idf/n_d);

        normalize(histograms[i], histograms[i], 0, 1, NORM_MINMAX);
    
        trainHistograms.push_back(histograms[i]);
    }

    FileStorage file("model/codewords.yml", FileStorage::WRITE);
    file <<"idf"<<idf;
    file.release();
}

void trainBoW(const Mat& descPerImg, const Mat& labelPerImg) {
    Ptr<KNearest> knn = KNearest::create();

    knn->setIsClassifier(true);
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->setDefaultK(1);
    Ptr<TrainData> td = TrainData::create(descPerImg, ROW_SAMPLE, labelPerImg);
    
    knn->train(td);
    
    knn->save("model/model_knn.yml");
}

int main(int argc, char** argv) {
    string basePath = "/home/francesco/Scaricati/dataFINALE/";

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

    FileStorage file("model/codewords.yml", FileStorage::APPEND);
    for (int i = 0; i < 3; i++) {
        file <<"codewords_"+to_string(i)<<codewords[i];
    }
    file.release();
    
    return 0;
}