#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stack>

using namespace cv;
using namespace std;

int K = 20;

vector<Vec3b> colors = {
    Vec3b(255, 0, 255),
    Vec3b(255, 0, 0),
    Vec3b(0, 0, 255),
    Vec3b(255, 255, 0),
    Vec3b(50, 50, 255),
    Vec3b(255, 100, 100),
    Vec3b(0, 255, 0),
    Vec3b(100, 0, 100),
    Vec3b(255, 0, 255),
    Vec3b(180, 50, 78),
    Vec3b(255, 55, 100),
    Vec3b(0, 40, 145),
    Vec3b(115, 90, 45),
    Vec3b(54, 90, 225),
    Vec3b(52, 78, 235),
    Vec3b(2, 0, 255),
    Vec3b(69, 143, 145),
    Vec3b(200, 200, 205),
    Vec3b(120, 120, 120),
    Vec3b(150, 150, 150)
};

vector<u_char> m1 = {
        0,0,0,1,1,1,0,0,0,
        0,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,0,
        1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,
        0,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,0,
        0,0,0,1,1,1,0,0,0
};

vector<u_char> m2 = {
        0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0
};

vector<u_char> m3 = {
        0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1, //
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
};


void get_Jimg(const Mat& class_map, Mat& J_img) {
    int window_size = 33;
    
    Mat window = Mat(m3).reshape(1, window_size);
    
    J_img = Mat(class_map.size(), CV_32FC1);
    for (int row = 0; row < class_map.rows; row++) {
        for (int col = 0; col < class_map.cols; col++) {
            int x_top_left = (col - window_size/2);
            int y_top_left = (row - window_size/2);

            int c = 0;
            float mx = 0, my = 0;
            vector<int> count(K,0);
            vector<float> mix(K,0), miy(K,0);
            for (int j = y_top_left; j < y_top_left+window_size; j++) {
                for (int i = x_top_left; i < x_top_left+window_size; i++) {
                    if (i < 0 || i >= class_map.cols || j < 0 || j >= class_map.rows || !window.at<u_char>(j-y_top_left, i-x_top_left)) {
                        continue;
                    }

                    int label = class_map.at<u_char>(j,i);
                    mix[label]+=i;
                    miy[label]+=j;
                    count[label]++;
                    c++;
            
                    mx+=i;
                    my+=j;
                }
            }

            mx /= (c*1.); my /= (c*1.);

            for (int i = 0; i < K; i++) {
                mix[i] /= count[i];
                miy[i] /= count[i];
            }

            float ST = 0;
            vector<float> SWi(K,0);
            for (int j = y_top_left; j < y_top_left+window_size; j++) {
                for (int i = x_top_left; i < x_top_left+window_size; i++) {
                    if (i < 0 || i >= class_map.cols || j < 0 || j >= class_map.rows || !window.at<u_char>(j-y_top_left, i-x_top_left)) {
                        continue;
                    }

                    ST+=(pow((i-mx),2) + pow((j-my),2));

                    int label = class_map.at<u_char>(j,i);
                    SWi[label] += (pow((i-mix[label]),2) + pow((j-miy[label]),2));
                }
            }

            float SW = 0;
            for (float s : SWi) SW += s;

            float J = (ST-SW)/SW;
            J_img.at<float>(row, col) = J;   
        }
    }
}

void spatial_segmentation(const Mat& class_map) {
    Mat J_img;
    get_Jimg(class_map, J_img);

    Mat mean, stddev;
    meanStdDev(J_img, mean, stddev);
    double Tj = mean.at<double>(0) + stddev.at<double>(0)*-0.6;
    
    cout<<"J-avg: "<<mean<<" STD: "<<stddev<<"\n";                      //TODO remove
    cout<<"TJ: "<<Tj<<"\n";

    Mat segmented_regions;
    threshold(J_img, segmented_regions, Tj, 255, THRESH_BINARY_INV);
    segmented_regions.convertTo(segmented_regions, CV_8UC1);

    Mat final_img = Mat::zeros(J_img.size(), CV_8UC1);

    int count = 0;
    float J_avg = 0;
    Mat components, stats, centroids;
    int nLabels = connectedComponentsWithStats(segmented_regions, components, stats, centroids, 4);
    
    imshow("all seeds", segmented_regions);

    for (int i = 0; i < components.rows; i++) {
        for (int j = 0; j < components.cols; j++) {
            if (stats.at<int>(components.at<int>(i,j), CC_STAT_AREA) < 512) {
                components.at<int>(i,j) = 0;
                
                J_avg += J_img.at<float>(i,j);
                count++;
            }
            else {
                J_img.at<float>(i,j) = 10000;
                if (components.at<int>(i,j) != 0) final_img.at<u_char>(i,j) = 255;
            }
        }
    }

    imshow("Initail sure segmentation", final_img);
    cout<<"J avg: "<<J_avg<<"\n";


    J_avg /= count;
    threshold(J_img, segmented_regions, J_avg, 255, THRESH_BINARY_INV);
    segmented_regions.convertTo(segmented_regions, CV_8UC1);

    imshow("growing regions", segmented_regions);

    imshow("sum", final_img + (segmented_regions*0.5));


    nLabels = connectedComponentsWithStats(segmented_regions, components, stats, centroids, 4);
}

void kmeans_segmentation(const Mat& src, Mat& labels, Mat& dst1, Mat& dst2) {
    Mat features = Mat(src.rows*src.cols, 3, CV_32F);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int z = 0; z < 3; z++) {
                features.at<float>(i*src.cols + j, z) = src.at<Vec3b>(i,j)[z];
            }
        }
    }

    vector<Vec3b> centers;
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER, 15, 1.0);
    kmeans(features, K, labels, criteria, 5, KMEANS_PP_CENTERS, centers);
    
    dst1 = Mat(src.size(), CV_8UC3);
    dst2 = Mat(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int l = labels.at<int>(i*src.cols + j, 0);
            dst1.at<Vec3b>(i,j) = centers[l];
            dst2.at<Vec3b>(i,j) = colors[l];
        }
    }

    labels = labels.reshape(1, src.rows);
    labels.convertTo(labels, CV_8UC1);
}


int main(int argc, char** argv) {
    Mat img = (argc > 1) ? imread(argv[1]) : imread("original/20151130_121522.jpg");

    Size s = (img.rows < img.cols) ? Size(408, 306) : Size(306, 408); 
    resize(img, img, s, 0, 0, INTER_LINEAR);

    //Color segmentation
    Mat labels, km1, km2;
    kmeans_segmentation(img, labels, km1, km2);

    //Spatial segmentation
    spatial_segmentation(labels);
    
    waitKey(0);
    return 0;
}

//2448-3264
//306-408