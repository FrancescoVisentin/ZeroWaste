#include <metrics.hpp>

using namespace std;
using namespace cv;
using namespace zw;

void zw::averagePrecision(const vector<vector<pair<Rect,int>>>& detectedItemsPerTray, std::string trayPath) {
    int totalGroundTruths = 0;
    vector<vector<int>> classPredictions (14, vector<int>(0));
    for (int i = 0; i < detectedItemsPerTray.size()-1; i++) {
        string filename = (i == 0) ? "/bounding_boxes/food_image_bounding_box.txt" : 
                                     "/bounding_boxes/leftover"+to_string(i)+"_bounding_box.txt";

        ifstream file(trayPath+filename, ios_base::in);
        
        // Load the ground truth boxes for the given image and creates a relevance vector with the IDs of the detected items
        vector<Rect> imgGroundThruth (14, Rect());
        for (string line; getline(file, line);) {
            line = line.substr(line.find("ID: ")+4);
            int id = stoi(line.substr(0, line.find(";")));

            line = line.substr(line.find('[')+1);
            vector<int> rectComponents;
            for (int i = 0; i < 4; i++) {
                string delimeter = (i != 3) ? ", " : "]";

                string comp = line.substr(0, line.find(delimeter));
                line.erase(0, line.find(delimeter) + delimeter.length());
                rectComponents.push_back(stoi(comp));
            }

            Rect bb = Rect(rectComponents[0], rectComponents[1] ,rectComponents[2], rectComponents[3]);
            imgGroundThruth[id] = bb;
            totalGroundTruths++;
        }

        // Computes IoU for the class of each detected box. If its IoU is > 0.5 save it considers it a correclty deteceted item
        for (const pair<Rect,int>& detectedBox : detectedItemsPerTray[i]) {
            int id = detectedBox.second;
            Rect dBox = detectedBox.first;
            Rect gBox = imgGroundThruth[id];

            // Missclassified item (False positive)
            if (gBox.area() == 0) {
                classPredictions[id].push_back(0);
                continue;
            }

            // Computes intersection area
            int intersectionWidth = std::max(0, std::min(dBox.x+dBox.width, gBox.x+gBox.width)-std::max(dBox.x, gBox.x));
            int intersectionHeight = std::max(0, std::min(dBox.y+dBox.height, gBox.y+gBox.height)-std::max(dBox.y, gBox.y));
            
            if (intersectionWidth <= 0 || intersectionHeight <= 0) {
                classPredictions[id].push_back(0);
                continue;
            }            
            int intersectionArea = intersectionWidth * intersectionHeight;

            // Computes union area            
            int unionArea = gBox.area() + dBox.area() - intersectionArea;

            double iou = 1.*intersectionArea/unionArea;
            if (iou > 0.5)  classPredictions[id].push_back(1);  // True positive
            else            classPredictions[id].push_back(0);  // False positive
        }
    }

    // Ground truth was not loaded correctly
    if (totalGroundTruths <= 0 ) return;
    
    // Calculates AP for each detected class
    vector<double> classAP;
    for (int k = 1; k < 14; k++) {
        // Class not detected
        if (classPredictions[k].size() <= 0) continue;
        
        vector<int> predVector = classPredictions[k];
        vector<double> precision (predVector.size(), 0.0);
        vector<double> recall (predVector.size(), 0.0);
        
        int truePositives = 0;
        int falsePositives = 0;
        for (int i = 0; i < predVector.size(); i++) {
            if (predVector[i] == 1) truePositives++;
            else                    falsePositives++;

            precision[i] = 1.*(truePositives)/(truePositives+falsePositives);
            recall[i] = 1.*(truePositives/totalGroundTruths);
        }
        
        vector<double> interpolatedPrecision(precision.size(), 0.0);
        interpolatedPrecision[0] = precision[0];
        for (int i = 1; i < precision.size(); i++) {
            interpolatedPrecision[i] = std::max(interpolatedPrecision[i - 1], precision[i]);
        }
        
        vector<double> apPerRecallThreshold(11, 0.0);
        vector<double> recallThresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        
        for (int i = 0; i < recallThresholds.size(); i++) {
            double recallThreshold = recallThresholds[i];
            for (int j = 0; j < recall.size(); ++j) {
                if (recall[j] >= recallThreshold) {
                    apPerRecallThreshold[i] = interpolatedPrecision[j];
                    break;
                }
            }
        }
        
        double averagePrecision = 0.0;
        for (int i = 0; i < apPerRecallThreshold.size(); i++) {
            averagePrecision += apPerRecallThreshold[i];
        }
        averagePrecision /= apPerRecallThreshold.size();
        classAP.push_back(averagePrecision);
    }

    double sum = 0;
    for (double AP : classAP) sum += AP;
    double mAP = sum/classAP.size();

    // Save the results
    ofstream file(trayPath+"/tray_metrics.txt");
    file << "\nFOOD LOCALIZATION (mAP)\n";
    file << "Mean average precision over " << classAP.size() << " detected items in the tray: \tmAP = " << mAP << "\n";
}

void zw::mIoU(const vector<vector<pair<Rect,int>>>& detectedItemsPerTray, std::string trayPath) {
    // Computes mIoU for the base image and leftovers 1 and 2
    vector<double> mIoU;
    for (int i = 0; i < detectedItemsPerTray.size()-1; i++) {
        string filename = (i == 0) ? "/bounding_boxes/food_image_bounding_box.txt" : 
                                     "/bounding_boxes/leftover"+to_string(i)+"_bounding_box.txt";

        ifstream file(trayPath+filename, ios_base::in);
        
        // Load the ground truth boxes for the given image
        int n = 0;
        vector<Rect> imgGroundThruth (14, Rect());
        for (string line; getline(file, line);) {
            line = line.substr(line.find("ID: ")+4);
            int id = stoi(line.substr(0, line.find(";")));

            line = line.substr(line.find('[')+1);
            vector<int> rectComponents;
            for (int i = 0; i < 4; i++) {
                string delimeter = (i != 3) ? ", " : "]";

                string comp = line.substr(0, line.find(delimeter));
                line.erase(0, line.find(delimeter) + delimeter.length());
                rectComponents.push_back(stoi(comp));
            }

            Rect bb = Rect(rectComponents[0], rectComponents[1] ,rectComponents[2], rectComponents[3]);
            imgGroundThruth[id] = bb;
            n++;
        }

        // Computes IoU for the class of each detected box
        vector<double> IoU;
        for (const pair<Rect,int>& detectedBox : detectedItemsPerTray[i]) {
            Rect dBox = detectedBox.first;
            Rect gBox = imgGroundThruth[detectedBox.second];

            // Missclassified item
            if (gBox.area() == 0) {
                IoU.push_back(0);
                continue;
            }

            // Computes intersection area
            int intersectionWidth = std::max(0, std::min(dBox.x+dBox.width, gBox.x+gBox.width)-std::max(dBox.x, gBox.x));
            int intersectionHeight = std::max(0, std::min(dBox.y+dBox.height, gBox.y+gBox.height)-std::max(dBox.y, gBox.y));
            
            if (intersectionWidth <= 0 || intersectionHeight <= 0) {
                IoU.push_back(0);
                continue;
            }            
            int intersectionArea = intersectionWidth * intersectionHeight;

            // Computes union area            
            int unionArea = gBox.area() + dBox.area() - intersectionArea;

            double iou = 1.*intersectionArea/unionArea;
            IoU.push_back(iou);
        }
        
        double sum = 0;
        for (double iou : IoU) sum += iou;
        mIoU.push_back(sum/n);
    }

    // Save the results
    ofstream file(trayPath+"/tray_metrics.txt", ios_base::app);
    file << "\n\nFOOD SEGMENTATION (mIoU)\n";
    for (int i = 0; i < mIoU.size(); i++) {
        file << "Img " << i+1 << "\t mIoU = " << mIoU[i] << ":\n";
    }
}


void zw::leftoverRatio(const vector<Mat>& foodMasks, std::string trayPath) {
    if (foodMasks.size() == 0) return;

    // Pixel count for each food detected in the original image
    Mat baseImg = foodMasks[0];
    vector<int> baseCountPerID (14, 0);
    for (int i = 0; i < baseImg.rows; i++) {
        for (int j = 0; j < baseImg.cols; j++) {
            if (baseImg.at<u_char>(i,j) != 0) baseCountPerID[baseImg.at<u_char>(i,j)]++;
        }
    }

    // For each leftover image computes its leftover ratio with respect to the "original" image  
    vector<vector<double>> ratiosPerPair;
    for (int k = 1; k < foodMasks.size(); k++) {
        Mat lefoverImg = foodMasks[k];
        vector<int> leftoverCountPerID (14, 0);
        for (int i = 0; i < lefoverImg.rows; i++) {
           for (int j = 0; j < lefoverImg.cols; j++) {
                if (lefoverImg.at<u_char>(i,j) != 0) leftoverCountPerID[lefoverImg.at<u_char>(i,j)]++;
            }
        }

        // Computes the lefover ratios for each detected food
        vector<double> ratios (14, 0);
        for (int i = 1; i < 14; i++) {
            if (baseCountPerID[i] == 0) continue;

            double Ri = 1.*leftoverCountPerID[i]/baseCountPerID[i];
            ratios[i] = Ri;
        }

        ratiosPerPair.push_back(ratios);
    }

    // Save the results
    ofstream file(trayPath+"/tray_metrics.txt", ios_base::app);
    file << "\n\nFOOD LEFTOVER ESTIMATION\n";
    for (int i = 0; i < ratiosPerPair.size(); i++) {
        file << "Leftover img " << i+1 << ":\n";
        for (int j = 1; j < 14; j++) {
            if (ratiosPerPair[i][j] == 0) continue;

            file << "\tID: " << j << "\tRi = " << ratiosPerPair[i][j] << "\n";
        }
    }
}