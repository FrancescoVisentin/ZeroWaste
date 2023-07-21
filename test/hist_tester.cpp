#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;


void getHist(Mat& src, Mat& hist) {
    Mat h1;
    for (int i = 0; i < 3; i++) {
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        int histSize = 256;
        
        Mat tmp;
        calcHist(&src, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, true, true);
    
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
            calcHist(&hsv, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        else {
            float range[]{0, 256}; //the upper boundary is exclusive
            const float* histRange[] = { range };
            int histSize = 256;
            
            Mat tmp;
            calcHist(&hsv, 1, &i, cv::Mat(), tmp, 1, &histSize, histRange, true, true);
        
            h2.push_back(tmp);
        }
        
    }

    hist.push_back(h1);
    hist.push_back(h2);

    hist = hist.reshape(1,1);
}




#include <opencv2/core/utils/filesystem.hpp>


int main(int argc, char **argv){
    vector<string> dirPaths;
    cv::utils::fs::glob("/home/francesco/Scaricati/hist_main/second_course/", "*_*", dirPaths, false, true);
    
    std::sort(dirPaths.begin(), dirPaths.end());

    int count = 0;
    int count2 = 0;
    int tot = 0;
    Ptr<KNearest> knn = KNearest::load("/home/francesco/Scaricati/hist_main/second_course/modelKnn2.yml");
    for (int j = 0; j < dirPaths.size(); j++) {
        cout<<dirPaths[j]<<"\n";

        vector<string> paths;
        glob(dirPaths[j]+"/*.jpg", paths, true);
    
        for(int i = 0; i < paths.size(); i++) {
            Mat src = imread(paths[i]);

            vector<int> matches (11, 0);
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    Rect roi = Rect(a*src.cols/2, b*src.rows/2, src.cols/2, src.rows/2);
                    Mat m = Mat(src, roi);

                    Mat hist;
                    getHist(m, hist);

                    normalize(hist, hist, 1, NORM_MINMAX);

                    Mat res, neigh, dist;
                    knn->findNearest(hist, 4, res, neigh, dist);

                    int label = knn->predict(hist);
                    matches[label]++;
                }
            }

            int argMax = distance(matches.begin(), max_element(matches.begin(), matches.end()));
            if (argMax != j) count2++;

            Mat hist;
            getHist(src, hist);

            normalize(hist, hist, 1, NORM_MINMAX);

            Mat res, neigh, dist;
            knn->findNearest(hist, 4, res, neigh, dist);

            int label = knn->predict(hist);
            if (label != j+6) {
                cout<<"\t"+paths[i]<<" Wrong! label : "<<j<<" predicted: "<<neigh<<" "<<dist<<"\n";
                count++;
            }
            else cout<<"\t"+paths[i]<<" Correct!\n";
            
            tot++;
        }
    }

    cout<<"\nWrong: "<<count<<"/"<<tot<<"\n";
    cout<<"\nWrong: "<<count2<<"/"<<tot<<"\n";

    return 0;
}