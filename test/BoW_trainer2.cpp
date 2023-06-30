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
Ptr<SVM> svm = SVM::create();

int N_CODEWORDS = 200;

void getDirectoriesAndCategories(string path, vector<string>& directories, vector<string>& categories) {
    vector<string> paths;
    for(auto p : filesystem::recursive_directory_iterator(path)) {
        if (p.is_directory()) {
            paths.push_back(p.path().string());
        }
    }
    
    std::sort(paths.begin(), paths.end());

    for(int i = 0; i < paths.size(); i++) {
        directories.push_back(paths[i]);
        categories.push_back(paths[i].erase(0, path.size()));
    }

    FileStorage file("model/testset.yml", FileStorage::WRITE);
    file<<"categories"<<categories;
    file.release();
}

void splitData(const vector<string>& dir, int ratio, vector<string>& trainPaths, vector<int>& trainLabels) {
    vector<int> testLabels;    
    vector<string> testPaths;
    for (int i = 0; i < dir.size(); i++) {
        vector<string> cat_paths;
        glob(dir[i], cat_paths);
        
        for(int j = 0; j < cat_paths.size(); j++) {
            if (j % ratio == 0) {
                testPaths.push_back(cat_paths[j]);
                testLabels.push_back(i);
                //continue; //skip test set
            }
            trainPaths.push_back(cat_paths[j]);
            trainLabels.push_back(i);  
          }
    }

    FileStorage file("model/testset.yml", FileStorage::APPEND);
    file<<"testPaths"<<testPaths;
    file<<"testLabels"<<testLabels;
    file.release();
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

void getDescAndCodewords(const vector<string>& trainImgsPaths, vector<array<Mat,3>>& descPerImg, Mat codewords[3]) {
    cout<<"\tExtracting SIFT features...\n";
    Mat descriptors[3];
    for (int i = 0; i < trainImgsPaths.size(); i++) {
        Mat bgr[3];
        split(imread(trainImgsPaths[i]), bgr);

        array<Mat,3> channelDesc;
        for (int j = 0; j < 3; j++) {
            Mat desc;
            vector<KeyPoint> key;
            sift->detectAndCompute(bgr[j], noArray(), key, desc);   

            channelDesc[j] = desc;
            descriptors[j].push_back(desc);
        }
        descPerImg.push_back(channelDesc);
    }
    cout<<"\tFeatures extracted!\n";

    FileStorage file("model/codewords.yml", FileStorage::WRITE);
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 2000, 1.0);
    for (int j = 0; j < 3; j++) {
        cout<<"\tStarting k-means...\n";
        Mat label;
        double d = kmeans(descriptors[j], N_CODEWORDS, label, criteria, 3, KMEANS_PP_CENTERS, codewords[j]);

        file <<"codewords_"+to_string(j)<<codewords[j];
        
        cout<<"Clustering completed! avg distance: "<<d/trainImgsPaths.size()<<"\n";
    }
    file.release();
}

void getTrainingData(const vector<array<Mat,3>>& descPerImg, const vector<int> trueImgLabels, const Mat codewords[3], Mat& trainHistograms, Mat& trainLabels) {
    for (int i = 0; i < descPerImg.size(); i++) {
        Mat hist;
        for (int j = 0; j < 3; j++) {
            Mat channelHist = getHistogram(descPerImg[i][j], codewords[j]);
            hist.push_back(channelHist);
        }
        hist = hist.reshape(1, 1);
        normalize(hist, hist, 0, 500, NORM_MINMAX);

        trainHistograms.push_back(hist);
        trainLabels.push_back(Mat(1,1,CV_32SC1, trueImgLabels[i]));
    }
}

void trainBoW(const Mat& descPerImg, const Mat& labelPerImg) {
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

    vector<string> dir, categories;
    getDirectoriesAndCategories(basePath, dir, categories);

    vector<int> trainImgsLabels;
    vector<string> trainImgsPaths;
    splitData(dir, 5, trainImgsPaths, trainImgsLabels);

    Mat codewords[3];
    vector<array<Mat,3>> descPerImg;
    cout<<"Creating the codewords...\n";
    getDescAndCodewords(trainImgsPaths, descPerImg, codewords);
    cout<<"Codewords created!\n";

    Mat trainHistograms, trainLabels;
    cout<<"Computing the histograms for the training data...\n";
    getTrainingData(descPerImg, trainImgsLabels, codewords, trainHistograms, trainLabels);
    cout<<"Histograms computed!\n";

    cout<<"Training started...\n";
    trainBoW(trainHistograms, trainLabels);
    cout<<"Training completed!\n";
    
    cout<<"\nDONE!\n";
    return 0;
}