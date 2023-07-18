#include <iostream>
#include <algorithm>

struct BoundingBox {
    int left, top, right, bottom;
};

float IoU(const BoundingBox& box1, const BoundingBox& box2) {
    // interctio area
    int intersectionWidth = std::max(0, std::min(box1.right, box2.right) - std::max(box1.left, box2.left));
    int intersectionHeight = std::max(0, std::min(box1.bottom, box2.bottom) - std::max(box1.top, box2.top));
    
    if (intersectionWidth <= 0 || intersectionHeight <= 0) {
        return 0.0f;
    }
    
    int intersectionArea = intersectionWidth * intersectionHeight;

    // union
    int box1Area = (box1.right - box1.left) * (box1.bottom - box1.top);
    int box2Area = (box2.right - box2.left) * (box2.bottom - box2.top);
    
    int unionArea = box1Area + box2Area - intersectionArea;

    // IoU
    float iou = static_cast<float>(intersectionArea) / unionArea;
    return iou;
}
