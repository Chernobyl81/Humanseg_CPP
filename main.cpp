#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnx/cv/human_segmentation.h"


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

    int fontWeight = 2;
    int fontSize = 1;
    Scalar fontColor(255, 255, 255);
    cv::Point textPosition{20, 30};

    cv::Size frame_size{static_cast<int>(dWidth), static_cast<int>(dHeight)};
    std::cout << "width: " << frame_size.width << " height: " << frame_size.height << std::endl;

    onnx::hs::Tensor3f bg_tensor = onnx::hs::HumanSegmentaion::GenerateBg(bg, frame_size);
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
        // std::cout << "FPS: " << 1000/dr_s << "\n<<<<<<<<" << std::endl;
        std::string text = "FPS: " + std::to_string(1000/dr_s);

        putText(output, text, textPosition, FONT_HERSHEY_COMPLEX, fontSize, fontColor, fontWeight);
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
}
