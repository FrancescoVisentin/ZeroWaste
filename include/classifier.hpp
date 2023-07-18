#ifndef classifier_h
#define classifier_h

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <common.hpp>
#include <iostream>
#include <string>

namespace zw {
    constexpr int FIRST_COURSE_SAT = 90;
    constexpr float FIRST_COURSE_MAX_AVG = 0.5;

    const std::array<std::vector<int>,11> platesContent = {
        std::vector<int> {},                                      // Empty plate
        std::vector<int> {PASTA_TOMATO},                          // Pasta tomato
        std::vector<int> {PASTA_PESTO},                           // Pasta pesto
        std::vector<int> {PASTA_MEATSAUCE},                       // Pasta meat sauce
        std::vector<int> {PASTA_CLAMSMUSSELS},                    // Pasta clams mussels
        std::vector<int> {RICE_PEPPERPEAS},                       // Rice pepper and peas
        std::vector<int> {FISH_CUTLET, BASIL_POTATO},             // Fish cutlet and basil potatoes
        std::vector<int> {BEANS, PORK_CUTLET},                    // Pork cutlet and beans
        std::vector<int> {RABBIT, BEANS},                         // Rabbit and beans
        std::vector<int> {RABBIT},                                // Rabbit
        std::vector<int> {SEAFOOD_SALAD, BEANS, BASIL_POTATO}     // Seafood salad, potatoes and beans
    };

    //Object used to classify the plates found inside a tray
    class Classifier {
        private:
            // Vector of reference histograms with their corrisponding label
            std::vector<std::pair<cv::Mat, int>> trayRefHist;

            bool initialized = false;

        public:
            Classifier(){};
            
            // Given the plates ROIs assignins to each one a label  
            // When called on the fisrt referance tray image 
            void classifyAndUpdate(const cv::Mat& src, const std::vector<cv::Rect>& platesROI, const std::vector<cv::Mat>& platesMask, std::vector<std::vector<int>>& itemPerPlate);
    };
}


#endif