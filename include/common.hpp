#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>

namespace zw {
    enum FoodLabels {
        BACKGROUND          =  0,
        PASTA_PESTO         =  1,
        PASTA_TOMATO        =  2,
        PASTA_MEATSAUCE     =  3,
        PASTA_CLUMSMUSSELS  =  4,
        RICE_PEPPERPEAS     =  5,
        PORK_CUTLET         =  6,
        FISH_CUTLET         =  7,
        RABBIT              =  8,
        SEAFOOD_SALAD       =  9,
        BEANS               = 10,
        BASIL_POTATO        = 11,
        SALAD               = 12,
        BREAD               = 13
    };

    const std::array<cv::Vec3b, 14> foodColors = {
        (  0,   0,   0),    // Background
        (251, 110,  58),    // Pasta pesto
        (  1, 191, 175),    // Pasta tomato
        (216, 231,  48),    // Pasta meat sauce
        (251, 140, 151),    // Pasta clums and mussels
        (245, 163, 140),    // Rice peppers and peas
        ( 19, 180, 142),    // Pork cutlet
        ( 27, 201, 120),    // Fish cutlet
        ( 61,  92, 237),    // Rabbit
        (120,   9, 146),    // Seafood salas
        ( 78, 139, 138),    // Beans
        (240, 142, 116),    // Basil potato
        (214, 195, 224),    // Salad
        (176,  98, 229)     // Bread
    };

    // Draw the given mask as an overlay over the input image 
    static void drawMask(cv::Mat& src, const cv::Mat& mask, int id) {
        cv::Mat overlay = src.clone();
        for (int i = 0; i < overlay.rows; i++) {
            for (int j = 0; j < overlay.rows; j++) {
                if (mask.at<uchar>(i,j) > 0)
                    overlay.at<cv::Vec3b>(i,j) = foodColors[id];
            }
        }

        float alpha = 0.4;
        cv::addWeighted(overlay, alpha, src, 1-alpha, 0, src);
    }
}

#endif