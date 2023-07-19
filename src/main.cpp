#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <segmentation.hpp>
#include <classifier.hpp>
#include <metrics.hpp>
#include <common.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace zw;

Mat getOutputImg(vector<Mat>& trayOutputs, vector<Mat>& detectedFoodsMask, const vector<vector<pair<Rect,int>>>& detectedItemsPerTray) {
    Mat out = trayOutputs[0];
    Mat tmp = detectedFoodsMask[0];
    for (int i = 0; i < trayOutputs.size(); i++) {
        for (int j = 0; j < detectedItemsPerTray[i].size(); j++) {
            rectangle(detectedFoodsMask[i], detectedItemsPerTray[i][j].first, Scalar::all(255), 2);
        }

        if (i > 0) {
            if (trayOutputs[i].size() != trayOutputs[0].size()) {
                resize(trayOutputs[i], trayOutputs[i], trayOutputs[0].size());
                resize(detectedFoodsMask[i], detectedFoodsMask[i], detectedFoodsMask[0].size());
            }

            hconcat(out, trayOutputs[i], out);
            hconcat(tmp, detectedFoodsMask[i], tmp);
        }
    }

    Mat out2;
    cvtColor(tmp, out2, COLOR_GRAY2BGR);
    for (int i = 0; i < tmp.rows; i++) {
        for (int j = 0; j <tmp.cols; j++) {
            int v = tmp.at<u_char>(i,j);
            if (v > 0 && v < 255) out2.at<Vec3b>(i,j) = foodColors[v];
        }
    }

    vconcat(out, out2, out);
    return out;
}


void processTray(string trayPath, Mat& out) {
    vector<string> imgPaths;
    cv::utils::fs::glob(trayPath, "*.jpg", imgPaths, false, false);

    Classifier cf;
    vector<Mat> trayOutputs;
    vector<Mat> detectedFoodsMask;
    vector<vector<pair<Rect,int>>> detectedItemsPerTray;
    for (int i = 0; i < imgPaths.size(); i++) {
        cout << "\b\b\b"+to_string(i+1)+"/"+to_string(imgPaths.size()) << flush;
        
        Mat src = imread(imgPaths[i]);

        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(3,3), 1);

        Mat roiMask = Mat::zeros(gray.size(), CV_8UC3);
        vector<Mat> platesMask, saladMask;
        vector<Rect> platesROI, saladROI, breadROI;
        getPlatesROI(gray, roiMask, platesROI, platesMask);
        getSaladROI(gray, roiMask, saladROI, saladMask);
        getBreadROI(src-roiMask, breadROI);

        vector<pair<Rect,int>> trayItems;
        Mat foodMask = Mat::zeros(src.size(), CV_8U);
        segmentAndDetectPlates(src, platesROI, platesMask, cf, foodMask, trayItems);
        segmentSalad(src, saladROI, saladMask, foodMask, trayItems);
        segmentBread(src, breadROI, foodMask, trayItems);

        trayOutputs.push_back(src);
        detectedFoodsMask.push_back(foodMask);
        detectedItemsPerTray.push_back(trayItems);
    }

    // Saves segmentation mask computed for each tray
    cv::utils::fs::createDirectory(trayPath+"/masks");
    for (int i = 0; i < detectedFoodsMask.size(); i++) {
        imwrite(trayPath+"/masks/img_"+to_string(i)+".jpg", detectedFoodsMask[i]);
    }

    // Saves bounding boxes computed for each tray
    cv::utils::fs::createDirectory(trayPath+"/bounding_boxes");
    for (int i = 0; i < detectedItemsPerTray.size(); i++) {
        ofstream file(trayPath+"/bounding_boxes/img_"+to_string(i)+".txt");
        for (int j = 0; j < detectedItemsPerTray[i].size(); j++) {
            file << "ID: " << detectedItemsPerTray[i][j].second << " " << detectedItemsPerTray[i][j].first << "\n"; 
        }
        file.close();
    }

    // Computes the required metrics for the tray
    averagePrecision(detectedItemsPerTray, trayPath);
    IoU(detectedItemsPerTray, trayPath);
    leftoverRatio(detectedFoodsMask, trayPath);

    // Output image to show the results
    out = getOutputImg(trayOutputs, detectedFoodsMask, detectedItemsPerTray);
}


int main(int argc, char** argv) {
    if(argc != 2) {
        cerr << "ERROR: missing input base path"                 << "\n"
             << "Usage: ./main <base-path-to-input-trays>"       << "\n"
             << "Ex:    ./main ../inputs/Food_leftover_dataset/" << "\n";
        return -1;
    }
    string basePath = argv[1];
    
    // Loads paths to input trays directories
    vector<string> dirPaths;
    cv::utils::fs::glob(basePath, "tray*", dirPaths, false, true);

    if (dirPaths.size() <= 0) {
        cerr << "ERROR: wrong or empty input base path"          << "\n"
             << "Usage: ./main <base-path-to-input-trays>"       << "\n"
             << "Ex:    ./main ../inputs/Food_leftover_dataset/" << "\n";
        return -1;
    }

    // Process each tray
    for (int i = 0; i < dirPaths.size(); i++) {
        cout << "Processing tray at: " <<  dirPaths[i] << " Image: -/-" << flush;
        
        Mat trayResult;
        processTray(dirPaths[i], trayResult);
        cout << "\tDone!\n";
        
        // Show output
        resize(trayResult, trayResult, Size(1280, 640));
        imshow("Tray "+to_string(i+1), trayResult);
        waitKey(0);
    }
    
    return 0;
}