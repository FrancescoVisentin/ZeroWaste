#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
    Ptr<SIFT> s = SIFT::create();

    vector<string> p;
    glob("/home/francesco/Scaricati/pane/13_bread/*.jpg", p);

    Mat d;
    for (int i = 0; i < p.size(); i++) {
        Mat src = imread(p[i]);
        vector<KeyPoint> key;
        Mat desc;
        s->detectAndCompute(src, noArray(), key, desc);

        d.push_back(desc);
    }

    FileStorage file("../build/bread.yml", FileStorage::WRITE);
    file<<"bread"<<d;
    file.release();

    return 0;
}