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

    // Item composition for each one of the dishes in the menu
    const std::array<std::vector<int>,10> dishContent = {
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

    // 3 reference dominant colors for each dish
    const std::array<std::array<cv::Vec3b,3>,10>  dishRefColors = {
        std::array<cv::Vec3b,3> {cv::Vec3b( 95, 160, 200), cv::Vec3b( 30,  80, 150), cv::Vec3b( 10,  15,  65)},    // Pasta tomato
        std::array<cv::Vec3b,3> {cv::Vec3b(100, 170, 190), cv::Vec3b( 30, 100, 130), cv::Vec3b( 10,  40,  50)},    // Pasta pesto
        std::array<cv::Vec3b,3> {cv::Vec3b(115, 170, 200), cv::Vec3b( 45, 100, 150), cv::Vec3b( 10,  30,  60)},    // Pasta meat sauce
        std::array<cv::Vec3b,3> {cv::Vec3b( 70, 145, 205), cv::Vec3b( 20,  80, 155), cv::Vec3b(  5,  25,  70)},    // Pasta clams mussels
        std::array<cv::Vec3b,3> {cv::Vec3b(125, 170, 195), cv::Vec3b( 55, 100, 140), cv::Vec3b( 10,  40,  60)},    // Rice pepper and peas
        std::array<cv::Vec3b,3> {cv::Vec3b(130, 180, 200), cv::Vec3b( 55, 110, 155), cv::Vec3b( 15,  40,  70)},    // Fish cutlet and basil potatoes
        std::array<cv::Vec3b,3> {cv::Vec3b(120, 145, 180), cv::Vec3b( 60,  80, 125), cv::Vec3b( 15,  25,  45)},    // Pork cutlet and beans
        std::array<cv::Vec3b,3> {cv::Vec3b(105, 135, 170), cv::Vec3b( 40,  65, 110), cv::Vec3b( 10,  15,  34)},    // Rabbit and beans
        std::array<cv::Vec3b,3> {cv::Vec3b(120, 160, 195), cv::Vec3b( 45,  75, 120), cv::Vec3b( 10,  20,  40)},    // Rabbit
        std::array<cv::Vec3b,3> {cv::Vec3b(130, 175, 200), cv::Vec3b( 60, 100, 140), cv::Vec3b( 15,  25,  55)}     // Seafood salad, potatoes and beans
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