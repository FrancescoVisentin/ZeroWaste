#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <segmentation.hpp>
#include <classifier.hpp>
#include <common.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace zw;

void processTray(string trayPath, Mat& out) {
    vector<string> imgPaths;
    cv::utils::fs::glob(trayPath, "*.jpg", imgPaths, false, false);

    vector<Mat> trayOutputs;
    vector<Mat> detectedFoodsMask;
    vector<vector<pair<Rect,int>>> detectedItemsPerTray;
    for (int i = 0; i < imgPaths.size(); i++) {
        Mat src = imread(imgPaths[i]);

        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(3,3), 1);

        Mat roiMask = Mat::zeros(gray.size(), CV_8UC3);
        vector<Rect> platesROI, saladROI, breadROI;
        getPlatesROI(gray, roiMask, platesROI);
        getSaladROI(gray, roiMask, saladROI);
        getBreadROI(src-roiMask, breadROI);

        vector<pair<Rect,int>> trayItems;
        Mat foodMask = Mat::zeros(src.size(), CV_8U);
        segmentAndDetectPlates(src, platesROI, foodMask, trayItems);
        segmentAndDetectSalad(src, saladROI, foodMask, trayItems);
        segmentAndDetectBread(src, breadROI, foodMask, trayItems);

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

    // Output image to show the results
    out = trayOutputs[0];
    for (int i = 1; i < trayOutputs.size(); i++) {
        if (trayOutputs[i].size() != trayOutputs[0].size())
            resize(trayOutputs[i], trayOutputs[i], trayOutputs[0].size());

        hconcat(out, trayOutputs[i], out);
    }
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
        cout << "Processing tray at: " <<  dirPaths[i] << flush;
        
        Mat trayResult;
        processTray(dirPaths[i], trayResult);
        cout << "\tDone!\n";
        
        // Show output
        resize(trayResult, trayResult, Size(1280, 320));
        imshow("Tray "+to_string(i+1), trayResult);
        waitKey(0);
    }
    
    return 0;
}