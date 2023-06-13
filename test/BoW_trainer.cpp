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

int N_CODEWORDS = 210;

void getDirectoriesAndCategories(string path, vector<string>& directories, vector<string>& categories) {
    for(auto p : filesystem::recursive_directory_iterator(path))
        if (p.is_directory()) {
            string dir = p.path().string();

            directories.push_back(dir);
            categories.push_back(dir.erase(0, path.size()));
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
                continue; //skip test set
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

void getDescAndCodewords(const vector<string>& trainImgsPaths, vector<Mat>& descPerImg, Mat& codewords) {
    cout<<"\tExtracting SIFT features...\n";
    Mat descriptors;
    for (int i = 0; i < trainImgsPaths.size(); i++) {
        Mat desc;
        vector<KeyPoint> key;
        sift->detectAndCompute(imread(trainImgsPaths[i]), noArray(), key, desc);   

        descriptors.push_back(desc);
        descPerImg.push_back(desc);
    }
    cout<<"\tFeatures extracted!\n";

    cout<<"\tStarting k-means...\n";
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 1000, 1.0);
    Mat label;
    double d = kmeans(descriptors, N_CODEWORDS, label, criteria, 3, KMEANS_PP_CENTERS, codewords);

    FileStorage file("model/codewords.yml", FileStorage::WRITE);
    file <<"codewords"<<codewords;
    file.release();
    
    cout<<"Clustering completed! avg distance: "<<d/trainImgsPaths.size()<<"\n";
}

void getTrainingData(const vector<Mat>& descPerImg, const vector<int> trueImgLabels, const Mat& codewords, Mat& trainHistograms, Mat& trainLabels) {
    for (int i = 0; i < descPerImg.size(); i++) {
        Mat hist = getHistogram(descPerImg[i], codewords);
        
        trainHistograms.push_back(hist);
        trainLabels.push_back(Mat(1,1,CV_32SC1, trueImgLabels[i]));
    }
}

void trainBoW(const Mat& descPerImg, const Mat& labelPerImg) {
    svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
	Ptr<TrainData> td = TrainData::create(descPerImg, ROW_SAMPLE, labelPerImg);

	svm->train(td);
    //svm->trainAuto(td, 4);

    svm->save("model/model.yml");
}

int main(int argc, char** argv) {
    string basePath = "data/";

    vector<string> dir, categories;
    getDirectoriesAndCategories(basePath, dir, categories);

    vector<int> trainImgsLabels;
    vector<string> trainImgsPaths;
    splitData(dir, 5, trainImgsPaths, trainImgsLabels);

    Mat codewords;
    vector<Mat> descPerImg;
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