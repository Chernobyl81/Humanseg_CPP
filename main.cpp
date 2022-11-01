#include <iostream>
#include <chrono>
#include "onnx/cv/human_segmentation.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

void normalize(cv::Mat &img, Eigen::Tensor<float, 3, Eigen::RowMajor> &tensor)
{
    cv::cv2eigen(img, tensor);
    tensor = tensor / 255.0f;
    tensor = tensor - 0.5f;
    tensor = tensor / 0.5f;

    Eigen::array<int, 3> shuffling({2, 0, 1});
    Eigen::Tensor<float, 3, Eigen::RowMajor> transposed = tensor.shuffle(shuffling);
    tensor = transposed;
}

void preprocess(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(398, 224));
    Eigen::Tensor<float, 3, Eigen::RowMajor> tensor;
    cv::cv2eigen(img, tensor);
    normalize(img, tensor);
    std::cout << tensor(0,0,191) << std::endl;
    // cv::eigen2cv(tensor, img);
}

int main(int argc, char **argv)
{
    // Eigen::RowVector3f rv{0.5, 0.5, 0.5};
    // std::cout << rv << std::endl;

    // // float f[1][1][3] = {{{0.5, 0.5, 0.5}}};
    // float f[3] = {0.5, 0.6, 0.7};
    // Eigen::Matrix<float, 1, 1, 3, Eigen::RowMajor> m1(&f[1]);
    // Eigen::Matrix<Eigen::Matrix<Eigen::Matrix<float, 1, 3>,1 ,1>, 1, 1, 1> k;
    // std::cout << k.rows() << "x" << k.cols() << std::endl;

    // auto m2 = Eigen::Matrix3f::Random();
    // std::cout << m1 << std::endl;
   
    cv::Mat image = cv::imread("/home/david/Desktop/R-C.jpg");
    
    auto start = std::chrono::steady_clock::now();

    preprocess(image);
    
    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "preprocess use " << dr_s << "ms" << std::endl;
    // a_tensor - 0.5f;
    // a_tensor / 0.5f;

    // cv::Mat img;
    // cv::eigen2cv(a_tensor, img);
    // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    // cv::imwrite("/home/david/Desktop/R-C-pre.jpg", img);
}
