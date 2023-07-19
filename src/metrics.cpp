#include <metrics.hpp>

using namespace std;
using namespace cv;
using namespace zw;

double zw::averagePrecision(const vector<vector<pair<Rect,int>>>& detectedItemsPerTray, std::string resPath) {

    vector<int> relevance;

    // Read the IDs from the text file
    std::ifstream relevantIdsFile(resPath + "/relevant_ids.txt");
    std::set<int> relevantIds;
    int id;
    while (relevantIdsFile >> id) {
        relevantIds.insert(id);
    }
    relevantIdsFile.close();
    
    // da  detectedItemsPerTray ricava gli IDs
    for (const auto& trayItems : detectedItemsPerTray) {
        for (const auto& item : trayItems) {
            
            int relevant = (relevantIds.find(item.second) != relevantIds.end()) ? 1 : 0;
            relevance.push_back(relevant);
        }
    }

    
    return averagePrecision;

    
}

double zw::IoU(const vector<vector<pair<Rect,int>>>& detectedItemsPerTray, std::string resPath) {

    return 0;
}


double zw::leftoverRatio(const vector<Mat>& foodMasks, std::string resPath) {
    
    return 0;
}
