#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnx/cv/human_segmentation.h"
#include "grpc_client.h"

namespace tc = triton::client;

struct ModelInfo
{

    // for trition
    int max_batch_size_ = 1;
    std::string output_name_;
    std::string input_name_;
    std::string input_format_;
    std::string input_datatype_;
    int input_c_;
    int input_h_;
    int input_w_;

    int type1_;
    int type3_;
};

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

bool ParseType(const std::string &dtype, int *type1, int *type3)
{
    if (dtype.compare("UINT8") == 0)
    {
        *type1 = CV_8UC1;
        *type3 = CV_8UC3;
    }
    else if (dtype.compare("INT8") == 0)
    {
        *type1 = CV_8SC1;
        *type3 = CV_8SC3;
    }
    else if (dtype.compare("UINT16") == 0)
    {
        *type1 = CV_16UC1;
        *type3 = CV_16UC3;
    }
    else if (dtype.compare("INT16") == 0)
    {
        *type1 = CV_16SC1;
        *type3 = CV_16SC3;
    }
    else if (dtype.compare("INT32") == 0)
    {
        *type1 = CV_32SC1;
        *type3 = CV_32SC3;
    }
    else if (dtype.compare("FP32") == 0)
    {
        *type1 = CV_32FC1;
        *type3 = CV_32FC3;
    }
    else if (dtype.compare("FP64") == 0)
    {
        *type1 = CV_64FC1;
        *type3 = CV_64FC3;
    }
    else
    {
        return false;
    }

    return true;
}

void ParseModelGrpc(
    const inference::ModelMetadataResponse &model_metadata,
    const inference::ModelConfigResponse &model_config,
    const size_t batch_size,
    ModelInfo *model_info)
{
    if (model_metadata.inputs().size() != 1)
    {
        std::cerr << "expecting 1 input, got " << model_metadata.inputs().size()
                  << std::endl;
        exit(1);
    }

    if (model_metadata.outputs().size() != 1)
    {
        std::cerr << "expecting 1 output, got " << model_metadata.outputs().size()
                  << std::endl;
        exit(1);
    }

    if (model_config.config().input().size() != 1)
    {
        std::cerr << "expecting 1 input in model configuration, got "
                  << model_config.config().input().size() << std::endl;
        exit(1);
    }

    auto input_metadata = model_metadata.inputs(0);
    auto input_config = model_config.config().input(0);

    auto output_metadata = model_metadata.outputs(0);

    if (output_metadata.datatype().compare("FP32") != 0)
    {
        std::cerr << "expecting output datatype to be FP32, model '"
                  << model_metadata.name() << "' output type is '"
                  << output_metadata.datatype() << "'" << std::endl;
        exit(1);
    }
    model_info->max_batch_size_ = model_config.config().max_batch_size();

    // Model specifying maximum batch size of 0 indicates that batching
    // is not supported and so the input tensors do not expect a "N"
    // dimension (and 'batch_size' should be 1 so that only a single
    // image instance is inferred at a time).
    if (model_info->max_batch_size_ == 0)
    {
        if (batch_size != 1)
        {
            std::cerr << "batching not supported for model \""
                      << model_metadata.name() << "\"" << std::endl;
            exit(1);
        }
    }
    else
    {
        //  model_info->max_batch_size_ > 0
        if (batch_size > (size_t)model_info->max_batch_size_)
        {
            std::cerr << "expecting batch size <= " << model_info->max_batch_size_
                      << " for model '" << model_metadata.name() << "'" << std::endl;
            exit(1);
        }
    }

    // Output is expected to be a vector. But allow any number of
    // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    // }, { 10, 1, 1 } are all ok).
    bool output_batch_dim = (model_info->max_batch_size_ > 0);
    size_t non_one_cnt = 0;
    for (const auto dim : output_metadata.shape())
    {
        if (output_batch_dim)
        {
            output_batch_dim = false;
        }
        else if (dim == -1)
        {
            std::cerr << "variable-size dimension in model output not supported"
                      << std::endl;
            exit(1);
        }
        else if (dim > 1)
        {
            non_one_cnt += 1;
            if (non_one_cnt > 1)
            {
                std::cerr << "expecting model output to be a vector" << std::endl;
                exit(1);
            }
        }
    }

    // Model input must have 3 dims, either CHW or HWC (not counting the
    // batch dimension), either CHW or HWC
    const bool input_batch_dim = (model_info->max_batch_size_ > 0);
    const int expected_input_dims = 3 + (input_batch_dim ? 1 : 0);
    if (input_metadata.shape().size() != expected_input_dims)
    {
        std::cerr << "expecting input to have " << expected_input_dims
                  << " dimensions, model '" << model_metadata.name()
                  << "' input has " << input_metadata.shape().size() << std::endl;
        exit(1);
    }

    if ((input_config.format() != inference::ModelInput::FORMAT_NCHW) &&
        (input_config.format() != inference::ModelInput::FORMAT_NHWC))
    {
        std::cerr
            << "unexpected input format "
            << inference::ModelInput_Format_Name(input_config.format())
            << ", expecting "
            << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NHWC)
            << " or "
            << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NCHW)
            << std::endl;
        exit(1);
    }

    model_info->output_name_ = output_metadata.name();
    model_info->input_name_ = input_metadata.name();
    model_info->input_datatype_ = input_metadata.datatype();

    if (input_config.format() == inference::ModelInput::FORMAT_NHWC)
    {
        model_info->input_format_ = "FORMAT_NHWC";
        model_info->input_h_ = input_metadata.shape(input_batch_dim ? 1 : 0);
        model_info->input_w_ = input_metadata.shape(input_batch_dim ? 2 : 1);
        model_info->input_c_ = input_metadata.shape(input_batch_dim ? 3 : 2);
    }
    else
    {
        model_info->input_format_ = "FORMAT_NCHW";
        model_info->input_c_ = input_metadata.shape(input_batch_dim ? 1 : 0);
        model_info->input_h_ = input_metadata.shape(input_batch_dim ? 2 : 1);
        model_info->input_w_ = input_metadata.shape(input_batch_dim ? 3 : 2);
    }

    if (!ParseType(
            model_info->input_datatype_, &(model_info->type1_),
            &(model_info->type3_)))
    {
        std::cerr << "unexpected input datatype '" << model_info->input_datatype_
                  << "' for model \"" << model_metadata.name() << std::endl;
        exit(1);
    }
}

void grpc_test(char **argv)
{
    ModelInfo model_info;
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

    ParseModelGrpc(model_metadata, model_config, 1, &model_info);
}

int main(int argc, char **argv)
{
    // matting(argv);
    grpc_test(argv);
}
