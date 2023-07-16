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
    const std::array<std::vector<int>,11> platesID = {
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


    void detect(const cv::Mat& src, std::vector<int>& cat);
}


#endif