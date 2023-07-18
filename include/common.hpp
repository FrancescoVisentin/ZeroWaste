#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>

namespace zw {
    enum FoodLabels {
        BACKGROUND          =  0,
        PASTA_PESTO         =  1,
        PASTA_TOMATO        =  2,
        PASTA_MEATSAUCE     =  3,
        PASTA_CLAMSMUSSELS  =  4,
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
        cv::Vec3b (  0,   0,   0),    // Background
        cv::Vec3b (251, 110,  58),    // Pasta pesto
        cv::Vec3b (  1, 191, 175),    // Pasta tomato
        cv::Vec3b (216, 231,  48),    // Pasta meat sauce
        cv::Vec3b (251, 140, 151),    // Pasta clams and mussels
        cv::Vec3b (245, 163, 140),    // Rice peppers and peas
        cv::Vec3b ( 19, 180, 142),    // Pork cutlet
        cv::Vec3b ( 27, 201, 120),    // Fish cutlet
        cv::Vec3b ( 61,  92, 237),    // Rabbit
        cv::Vec3b (120,   9, 146),    // Seafood salas
        cv::Vec3b ( 78, 139, 138),    // Beans
        cv::Vec3b (240, 142, 116),    // Basil potato
        cv::Vec3b (114, 145, 224),    // Salad
        cv::Vec3b (176,  98, 229)     // Bread
    };

    // Draw the given mask as an overlay over the input image 
    static void drawMask(cv::Mat& src, const cv::Mat& mask) {
        cv::Mat overlay = src.clone();
        for (int i = 0; i < overlay.rows; i++) {
            for (int j = 0; j < overlay.cols; j++) {
                if (mask.at<uchar>(i,j) > 0)
                    overlay.at<cv::Vec3b>(i,j) = foodColors[mask.at<uchar>(i,j)];
            }
        }

        float alpha = 0.4;
        cv::addWeighted(overlay, alpha, src, 1-alpha, 0, src);
    }
}

#endif