#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void getColHist(Mat& src, Mat& hist, Mat& mask) {
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);

        hist.push_back(tmp);
    }
    hist = hist.reshape(1,1);
}


void getSatHist(Mat& src, Mat& hist, Mat& mask) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    extractChannel(hsv, hsv, 1);

    //imshow("sat", hsv);
    //waitKey(0);
    
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    int histSize = 256;
    
    int channel = 0;
    calcHist(&hsv, 1, &channel, mask, hist, 1, &histSize, histRange, true, true);

    hist = hist.reshape(1,1);
    //normalize(hist, hist, 1, NORM_MINMAX);
    //cout<<hist.size()<<" "<<hist<<"\n";
}

void getHist(Mat& src, Mat& hist, Mat& mask) {
    Mat h1;
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);

        h1.push_back(tmp);
    }

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Mat h2;
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            float range[]{0, 180}; //the upper boundary is exclusive
            const float* histRange[] = { range };
            int histSize = 180;
            
            Mat tmp;
            calcHist(&hsv, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        else {
            float range[]{0, 256}; //the upper boundary is exclusive
            const float* histRange[] = { range };
            int histSize = 256;
            
            Mat tmp;
            calcHist(&hsv, 1, &i, mask, tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        
    }

    hist.push_back(h1);
    hist.push_back(h2);

    hist = hist.reshape(1,1);
}

void kmeans_segmentation(Mat& src, Mat& dst, int K) {
    Mat features = Mat(src.rows*src.cols, 3, CV_32F);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            for (int z = 0; z < 3; z++)
                features.at<float>(i*src.cols + j, z) = src.at<Vec3b>(i,j)[z];

    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER, 15, 1.0);

    Mat labels,centers;
    kmeans(features, K, labels, criteria, 5, KMEANS_PP_CENTERS, centers);
    
    centers.convertTo(centers, CV_8UC3);
    dst = Mat(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int l = labels.at<int>(i*src.cols + j, 0);
            dst.at<Vec3b>(i,j) = centers.at<Vec3b>(l);
        }
    }

    Mat out = Mat(100, 300, CV_8UC3, centers.at<Vec3b>(0));
    for (int i = 1; i < 5; i++) {
        Mat tmp = Mat(100, 300, CV_8UC3, centers.at<Vec3b>(i));
        vconcat(out, tmp, out);
    }
    imshow("oo", out);
}

int main(int argc, char **argv){
    vector<string> imgPaths;
    cv::utils::fs::glob("/home/francesco/Scaricati/hist_main/second_course/all/", "*.jpg", imgPaths, false, false);
    
    for (int k = 0; k < imgPaths.size(); k++) {
        Mat src = imread(imgPaths[k]);

        Mat sat;
        cvtColor(src, sat, COLOR_BGR2HSV);
        extractChannel(sat, sat, 1);

        Mat satMask;
        inRange(sat, 80, 250, satMask);
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20,20));
        morphologyEx(satMask, satMask, MORPH_CLOSE, element);

        Mat features;
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                if (satMask.at<uchar>(i,j) != 0 ) {
                    Mat tmp = Mat(1,3, CV_32F);
                    for (int z = 0; z < 3; z++)
                        tmp.at<float>(0, z) = src.at<Vec3b>(i,j)[z];

                    features.push_back(tmp);
                }
            }
        }
        TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER, 15, 1.0);

        int K = 3;

        Mat labels,centers;
        kmeans(features, K, labels, criteria, 5, KMEANS_PP_CENTERS, centers);

        vector<pair<int,int>> count(K, pair<int,int>(-1,0));      
        for (int i = 0; i < labels.rows; i++) {
            count[labels.at<u_char>(i)].first = labels.at<u_char>(i);
            count[labels.at<u_char>(i)].second++;
        }
        sort(count.begin(), count.end(), [=](auto a, auto b) { 
                return a.second > b.second;}
        );

        centers.convertTo(centers, CV_8UC3);
        Mat out = Mat(200, 600, CV_8UC3, centers.at<Vec3b>(count[0].first));
        for (int i = 1; i < K; i++) {
            Mat tmp = Mat(200, 600, CV_8UC3, centers.at<Vec3b>(count[i].first));
            vconcat(out, tmp, out);
        }

        imshow("sat mask", satMask);
        imshow("color", out);
        cvtColor(satMask, satMask, COLOR_GRAY2BGR);
        imshow("src", src&satMask);
        waitKey(0);
    }

    
    return 0;
}

// PRIMI
    // Cozze
    // 206 149  72      201 143  68     204 146  72             205 145  70
    // 155  84  23      150  80  22     153  82  23     --->    155  80  20
    //  68  24   6       66  23   7      67  23   6              70  25   5

    // Ragù
    // 201 169 115             200  170 115
    // 148  99  46     --->    150  100  45
    //  62  29  12             60    30  10

    // Pesto
    // 189 171  98             190 170 100
    // 130 100  32     --->    130 100  30
    //  48  42  11              50  40  10

    // Pomodoro
    // 201 160  94      204 163  97             200 160  95
    // 147  80  34      152  82  33     --->    150  80  30
    //  66  18   8       65  17   9              65  15  10

    // Riso
    // 194 168 123             195 170 125
    // 142 106  54     --->    140 100  55
    //  62  38  13              60  40  10

// SECONDI
    // Arrosto
    // 174 138 118      182 150 124             180 145 120
    // 121  76  59      130  87  62     --->    125  80  60
    //  43  20  14       54  26  15              45  25  15

    // Coniglio
    // 195 158 119             195  160 120
    // 121  74  46     --->    120   75  45
    //  42  19  11             40    20  10

    // Coniglio fagioli
    // 172 135 104             170 135 105
    // 110  65  40     --->    110  65  40
    //  34  15   9              34  15  10

    // Mare
    // 197 174 132             200 175 130
    // 142  98  58     --->    140 100  60
    //  55  27  15              55  25  15

    // Pesce
    // 207 178 129      202 183 128     197 179 127             200 180 130
    // 149 103  54      163 120  54     152 108  48     --->    155 110  55
    //  53  30  15       78  47  19      70  40  14              70  40  15


