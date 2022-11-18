#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnx/cv/human_segmentation.h"
#include "grpc_client.h"


namespace tc = triton::client;

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

    Tensor3f bg_tensor = onnx::hs::HumanSegmentaion::GenerateBg(bg, frame_size);

    auto device = onnx::hs::HumanSegmentaion::StringToDevice(argv[5]);
    onnx::hs::HumanSegmentaion hs(argv[1], std::stoi(argv[4]), device);

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

        auto fps = 1000 / dr_s;
        std::string text = "FPS(";
        text.append(argv[5])
            .append("): ")
            .append(std::to_string(fps));
        std::cout << text << std::endl;

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


void grpc_test(char **argv)
{
    tc::Error err;
    tc::Headers http_headers;
    std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client;

    bool verbose = true;
    err = tc::InferenceServerGrpcClient::Create(&grpc_client, "localhost:8001", verbose);

    if (!err.IsOk())
    {
        std::cerr << "error: unable to create client for inference: " << err.Message()
                  << std::endl;
        EXIT_FAILURE;
    }

    inference::ModelMetadataResponse model_metadata;
    err = grpc_client->ModelMetadata(&model_metadata, "ppseg_onnx", "1", http_headers);
    if (!err.IsOk())
    {
        std::cerr << "error: failed to get model metadata: " << err << std::endl;
    }

    inference::ModelConfigResponse model_config;
    err = grpc_client->ModelConfig(&model_config, "ppseg_onnx", "1", http_headers);
    if (!err.IsOk())
    {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }
}

int main(int argc, char **argv)
{
    matting(argv);
    // grpc_test(argv);
    // onnx::core::HumanSegModelInfo model_info;
    // std::cout << model_info.MODEL_OUTPUT_SHAPE.max_size() << std::endl;
}
