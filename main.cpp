#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnx/cv/human_segmentation.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

void test()
{
    // Define a tensor
    Eigen::Tensor<int, 3> A(7, 9, 3);
    A.setRandom();
    std::cout << A.dimensions() << std::endl;
    std::cout << A << std::endl;
    std::cout << "-----------" << std::endl;

    Eigen::Tensor<int, 3> C(7, 9, 1);
    C.setRandom();
    std::cout << C.dimensions() << std::endl;
    std::cout << C << std::endl;
    std::cout << "-----------" << std::endl;

    Eigen::array<int, 3> bcast = {1, 1, 3};
    Eigen::Tensor<int, 3> ac =  C.broadcast(bcast);
    std::cout << ac.dimensions() << std::endl;
    std::cout << ac << std::endl;
    // std::cout << res << std::endl;

    // Define the pairs of indices to multiply (i.e. index 1 on A twice)
    // std::array<Eigen::IndexPair<long>, 1> idx = { Eigen::IndexPair<long> {1, 1} };

    // // Contract. B will have dimensions [55,2,55,2]
    // Eigen::Tensor<double,4> B = A.contract(A, idx);
    // std::cout << B.dimensions() << std::endl;
    // std::cout << B << std::endl;
}

void matting(char **argv)
{
    cv::Mat image = cv::imread(argv[2]);
    auto imageSize = image.size();

    cv::Mat bg = cv::imread(argv[3]);

    Eigen::Tensor<float, 3, Eigen::RowMajor> bg_tensor = onnx::hs::HumanSegmentaion::generateBg(bg, imageSize);
    onnx::hs::HumanSegmentaion hs(argv[1]);

    auto start = std::chrono::steady_clock::now();
    hs.detect(image, bg_tensor);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << dr_s << "ms" << std::endl;
}

int main(int argc, char **argv)
{
    matting(argv);
   // test();
}
