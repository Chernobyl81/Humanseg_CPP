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
    Eigen::Tensor<int, 3> ac = C.broadcast(bcast);
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

int matting(char **argv)
{
    
    // camera seg
    cv::Mat bg = cv::imread(argv[3]);
    VideoCapture cap(0);

    if (cap.isOpened() == false)
    {
        std::cout << "Cannot open the video camera" << std::endl;
        std::cin.get();
        return EXIT_FAILURE;
    }

    auto dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    auto dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    cv::Size s{static_cast<int>(dWidth), static_cast<int>(dHeight)};
    std::cout << "width: " << s.width << " height: " << s.height << std::endl;

    Eigen::Tensor<float, 3, Eigen::RowMajor> bg_tensor = 
        onnx::hs::HumanSegmentaion::GenerateBg(bg, s);
    onnx::hs::HumanSegmentaion hs(argv[1], std::stoi(argv[4]));

    std::string window_name = "My Camera Feed";
    namedWindow(window_name);

    while (true)
    {

        Mat frame;
        Mat output;
        bool bSuccess = cap.read(frame);

        if (bSuccess == false)
        {
            std::cout << "Video camera is disconnected" << std::endl;
            std::cin.get();
            break;
        }

        auto start = std::chrono::steady_clock::now();
        
        hs.detect(frame, bg_tensor, output);
        
        auto end = std::chrono::steady_clock::now();
        double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Total time: " << dr_s << "ms" << std::endl;

        imshow(window_name, output);

        if (waitKey(10) == 27)
        {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }

    cap.release();
    return 0;
}

int main(int argc, char **argv)
{
    matting(argv);
    // test();
}
