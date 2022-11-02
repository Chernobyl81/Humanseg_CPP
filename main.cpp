#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnx/cv/human_segmentation.h"

int main(int argc, char **argv)
{
    cv::Mat image = cv::imread(argv[2]);
    
    // auto start = std::chrono::steady_clock::now(); 
    onnx::hs::HumanSegmentaion hs(argv[1]);
    hs.detect(image);

    // auto end = std::chrono::steady_clock::now();
    // double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    // std::cout << "preprocess use " << dr_s << "ms" << std::endl;
    // a_tensor - 0.5f;
    // a_tensor / 0.5f;

    // cv::Mat img;
    // cv::eigen2cv(a_tensor, img);
    // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    // cv::imwrite("/home/david/Desktop/R-C-pre.jpg", img);
}
