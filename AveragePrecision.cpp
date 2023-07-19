#include <iostream>
#include <vector>


//       relevance è un vettore di 0 e 1 (0 = non rilevato, 1 = rilevato correttamente)
    //   relevance = {1, 0, 1, 0, 1, 0, 1, 0, 1}

double AveragePrecision(const std::vector<int>& relevance) {
    int numRelevant = 0;
    for (int i = 0; i < relevance.size(); ++i) {
        if (relevance[i] == 1) {
            numRelevant++;
        }
    }
    
    std::vector<double> precision(relevance.size(), 0.0);
    std::vector<double> recall(relevance.size(), 0.0);
    
    int truePositives = 0;
    for (int i = 0; i < relevance.size(); ++i) {
        if (relevance[i] == 1) {
            truePositives++;
            precision[i] = static_cast<double>(truePositives) / (i + 1);
            recall[i] = static_cast<double>(truePositives) / numRelevant;
        }
    }
    
    std::vector<double> interpolatedPrecision(precision.size(), 0.0);
    interpolatedPrecision[0] = precision[0];
    for (int i = 1; i < precision.size(); ++i) {
        interpolatedPrecision[i] = std::max(interpolatedPrecision[i - 1], precision[i]);
    }
    
    std::vector<double> apPerRecallThreshold(11, 0.0);
    std::vector<double> recallThresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    
    for (int i = 0; i < recallThresholds.size(); ++i) {
        double recallThreshold = recallThresholds[i];
        for (int j = 0; j < recall.size(); ++j) {
            if (recall[j] >= recallThreshold) {
                apPerRecallThreshold[i] = interpolatedPrecision[j];
                break;
            }
        }
    }
    
    double averagePrecision = 0.0;
    for (int i = 0; i < apPerRecallThreshold.size(); ++i) {
        averagePrecision += apPerRecallThreshold[i];
    }
    averagePrecision /= apPerRecallThreshold.size();
    
    return averagePrecision;
}
