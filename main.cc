#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "onnx/cv/human_segmentation.h"
#include "grpc_client.h"

#define GET_TRANSFORMATION_CODE(x) cv::COLOR_##x

enum ScaleType
{
    NONE = 0,
    VGG = 1,
    INCEPTION = 2
};

namespace tc = triton::client;
using namespace onnx::core;
using namespace onnx::hs;

typedef std::unique_ptr<tc::InferenceServerGrpcClient> GRPC_CLIENT;

struct ModelInfo
{

    // for trition
    int max_batch_size_ = 1;
    std::string output_name_;
    std::string input_name_;
    std::string input_format_;
    std::string input_datatype_;

    // channels
    int input_c_;

    // height
    int input_h_;

    // weigh
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

    Tensor3f bg_tensor = HumanSegmentaion::GenerateBg(bg, frame_size);

    auto device = HumanSegmentaion::StringToDevice(argv[5]);
    HumanSegmentaion hs(argv[1], std::stoi(argv[4]), device);

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

void ParseModelGrpc(const inference::ModelMetadataResponse &model_metadata,
                    const inference::ModelConfigResponse &model_config,
                    size_t batch_size,
                    ModelInfo *model_info)
{
    if (model_metadata.inputs().size() != 1)
    {
        std::cerr << "expecting 1 input, got " << model_metadata.inputs().size() << std::endl;
        exit(1);
    }

    if (model_metadata.outputs().size() != 1)
    {
        std::cerr << "expecting 1 output, got " << model_metadata.outputs().size() << std::endl;
        exit(1);
    }

    if (model_config.config().input().size() != 1)
    {
        std::cerr << "expecting 1 input in model configuration, got "
                  << model_config.config().input().size() << std::endl;
        exit(1);
    }

    auto input_metadata = model_metadata.inputs(0);
    auto output_metadata = model_metadata.outputs(0);
    auto input_config = model_config.config().input(0);

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
        if (batch_size > (size_t)model_info->max_batch_size_)
        {
            std::cerr << "expecting batch size <= " << model_info->max_batch_size_
                      << " for model '" << model_metadata.name() << "'" << std::endl;
            exit(1);
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

void normalize(const cv::Mat &image, Tensor3f &tensor)
{
    cv::cv2eigen(image, tensor);
    tensor = tensor / 255.0f;
    tensor = tensor - 0.5f;
    tensor = tensor / 0.5f;
}

void Preprocess(const cv::Mat &img,
                const std::string &format,
                int img_type1,
                int img_type3,
                size_t img_channels,
                const cv::Size &img_size,
                const ScaleType scale,
                std::vector<uint8_t> *input_data)
{
    // Image channels are in BGR order. Currently model configuration
    // data doesn't provide any information as to the expected channel
    // orderings (like RGB, BGR). We are going to assume that RGB is the
    // most likely ordering and so change the channels to that ordering.

    cv::Mat sample;
    if ((img.channels() == 3) && (img_channels == 1))
    {
        cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2GRAY));
    }
    else if ((img.channels() == 4) && (img_channels == 1))
    {
        cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2GRAY));
    }
    else if ((img.channels() == 3) && (img_channels == 3))
    {
        cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2RGB));
    }
    else if ((img.channels() == 4) && (img_channels == 3))
    {
        cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2RGB));
    }
    else if ((img.channels() == 1) && (img_channels == 3))
    {
        cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(GRAY2RGB));
    }
    else
    {
        std::cerr << "unexpected number of channels " << img.channels()
                  << " in input image, model expects " << img_channels << "."
                  << std::endl;
        exit(1);
    }

    cv::Mat sample_resized;
    if (sample.size() != img_size)
    {
        cv::resize(sample, sample_resized, img_size);
    }
    else
    {
        sample_resized = sample;
    }

    cv::Mat sample_type;
    sample_resized.convertTo(
        sample_type, (img_channels == 3) ? img_type3 : img_type1);

    cv::Mat sample_final;
    if (scale == ScaleType::INCEPTION)
    {
        if (img_channels == 1)
        {
            sample_final = sample_type.mul(cv::Scalar(1 / 255.0));
            sample_final = sample_final - cv::Scalar(0.5);
            sample_final = sample_final.mul(Scalar(2.0));
        }
        else
        {
            sample_final = sample_type.mul(cv::Scalar(1 / 255.0));
            sample_final = sample_final - cv::Scalar(0.5);
            sample_final = sample_final.mul(Scalar(2.0));
        }
    }
    else if (scale == ScaleType::VGG)
    {
        if (img_channels == 1)
        {
            sample_final = sample_type - cv::Scalar(128);
        }
        else
        {
            sample_final = sample_type - cv::Scalar(123, 117, 104);
        }
    }
    else
    {
        sample_final = sample_type;
    }

    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample_final.total() * sample_final.elemSize();
    size_t pos = 0;
    input_data->resize(img_byte_size);

    // For NHWC format Mat is already in the correct order but need to
    // handle both cases of data being contigious or not.
    if (format.compare("FORMAT_NHWC") == 0)
    {
        if (sample_final.isContinuous())
        {
            memcpy(&((*input_data)[0]), sample_final.datastart, img_byte_size);
            pos = img_byte_size;
        }
        else
        {
            size_t row_byte_size = sample_final.cols * sample_final.elemSize();
            for (int r = 0; r < sample_final.rows; ++r)
            {
                memcpy(
                    &((*input_data)[pos]), sample_final.ptr<uint8_t>(r), row_byte_size);
                pos += row_byte_size;
            }
        }
    }
    else
    {
        // (format.compare("FORMAT_NCHW") == 0)
        //
        // For CHW formats must split out each channel from the matrix and
        // order them as BBBB...GGGG...RRRR. To do this split the channels
        // of the image directly into 'input_data'. The BGR channels are
        // backed by the 'input_data' vector so that ends up with CHW
        // order of the data.
        std::vector<cv::Mat> input_bgr_channels;
        for (size_t i = 0; i < img_channels; ++i)
        {
            input_bgr_channels.emplace_back(
                img_size.height, img_size.width, img_type1, &((*input_data)[pos]));
            pos += input_bgr_channels.back().total() *
                   input_bgr_channels.back().elemSize();
        }

        cv::split(sample_final, input_bgr_channels);
    }

    if (pos != img_byte_size)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
                  << img_byte_size << std::endl;
        exit(1);
    }
}

void postprocess(const tc::InferResult *result,
                 const std::string &output_name,
                 const cv::Mat &originMat,
                 const Tensor3f &bgTensor,
                 cv::Mat &matted)
{
    tc::Error err;
    auto start = std::chrono::steady_clock::now();
    auto size = originMat.size();

    Tensor3f origin_tensor(originMat.cols, originMat.rows, 3);
    cv::cv2eigen(originMat, origin_tensor);

    std::array<float, HumanSegModelInfo::SHAPE> scroe_map{};

    std::string model_name;
    std::cout << result->ModelName(&model_name) << std::endl;
    size_t buf_size = 398 * 224 * 2;
    const uint8_t *buf;
    err = result->RawData(output_name, &buf, &buf_size);
    if (!err.IsOk())
    {
        std::cout << err.Message() << std::endl;
    }
}

const Tensor3f GenerateBg(cv::Mat &bg, const cv::Size &size)
{
    cv::resize(bg, bg, size);
    static Tensor3f bg_tensor(size.height, size.width, 3);
    cv::cv2eigen(bg, bg_tensor);
    return bg_tensor;
}

GRPC_CLIENT create_grpc_client(
    const tc::Headers &http_headers,
    const std::string &url,
    const size_t batch_size = 1,
    const bool verbose = false)
{
    GRPC_CLIENT client;
    auto err = tc::InferenceServerGrpcClient::Create(&client, url, verbose);
    if (!err.IsOk())
    {
        std::cerr << "Error: unable to create client for inference: "
                  << err.Message()
                  << std::endl;
        EXIT_FAILURE;
    }

    return client;
}

void get_model_info_from_server(GRPC_CLIENT &grpc_client,
                                ModelInfo &model_info,
                                const tc::Headers &http_headers,
                                inference::ModelMetadataResponse &model_metadata,
                                inference::ModelConfigResponse &model_config,
                                size_t batch_size = 1)
{
    tc::Error err;
    err = grpc_client->ModelMetadata(&model_metadata, "ppseg_onnx", "1", http_headers);
    if (!err.IsOk())
    {
        std::cerr << "error: failed to get model metadata: " << err << std::endl;
    }

    err = grpc_client->ModelConfig(&model_config, "ppseg_onnx", "1", http_headers);
    if (!err.IsOk())
    {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }

    ParseModelGrpc(model_metadata, model_config, batch_size, &model_info);
}

void model_shape(const ModelInfo &model_info, const size_t batch_size, std::vector<int64_t> &shape)
{
    // Include the batch dimension if required
    if (model_info.max_batch_size_ != 0)
    {
        shape.push_back(batch_size);
    }
    if (model_info.input_format_.compare("FORMAT_NHWC") == 0)
    {
        shape.push_back(model_info.input_h_);
        shape.push_back(model_info.input_w_);
        shape.push_back(model_info.input_c_);
    }
    else
    {
        shape.push_back(model_info.input_c_);
        shape.push_back(model_info.input_h_);
        shape.push_back(model_info.input_w_);
    }
}

void do_inference(GRPC_CLIENT &grpc_client,
                  const ModelInfo &model_info,
                  const std::vector<tc::InferInput *> &inputs_ptr,
                  const std::vector<const tc::InferRequestedOutput *> &outputs_ptr,
                  const tc::InferOptions &options,
                  const tc::Headers &headers,
                  tc::InferResult *result,
                  const Mat &origin_mat,
                  const Tensor3f &bgTensor,
                  Mat &output_mat,
                  bool verbose = false)
{
    auto start = std::chrono::steady_clock::now();
    auto err = grpc_client->Infer(&result, options, inputs_ptr, outputs_ptr, headers);
    if (!err.IsOk())
    {
        std::cerr << "failed sending synchronous infer request: " << err
                  << std::endl;
        exit(1);
    }

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Inference time: " << dr_s << "ms" << std::endl;

    if (!result->RequestStatus().IsOk())
    {
        std::cerr << "inference  failed with error: " << result->RequestStatus()
                  << std::endl;
        exit(1);
    }
    
    start = std::chrono::steady_clock::now();
    size_t buf_size;
    const uint8_t *buffer;
    std::array<float, HumanSegModelInfo::SHAPE> scroe_map{};
    err = result->RawData(model_info.output_name_, &buffer, &buf_size);
    if (!err.IsOk())
    {
        std::cout << "get rawData error " << err.Message() << std::endl;
    }

    auto casted = (float *)(buffer);
    std::copy(casted + HumanSegModelInfo::SHAPE, casted + HumanSegModelInfo::OUTPUT_TENSOR_SIZE + 1, scroe_map.data());
    Eigen::TensorMap<Tensor3f> score_tensor{scroe_map.data(), 1, 224, 398};

    Tensor3f origin_tensor(origin_mat.cols, origin_mat.rows, 3);
    cv::cv2eigen(origin_mat, origin_tensor);

    // Transpose
    const Shape3i shuffling({1, 2, 0});
    Tensor3f scrore_transposed = score_tensor.shuffle(shuffling);

    Mat tmp;
    auto size = origin_mat.size();
    cv::eigen2cv(scrore_transposed, tmp);
    cv::resize(tmp, tmp, size);

    Tensor3f alpha(size.height, size.width, 1);
    cv::cv2eigen(tmp, alpha);

    Shape3i bcast = {1, 1, 3};
    Tensor3f ab = alpha.broadcast(bcast);

    Tensor3f comb = ab * origin_tensor + (1 - ab) * bgTensor;
    Eigen::Tensor<cv::uint8_t, 3, Eigen::RowMajor> results = comb.cast<cv::uint8_t>();

#ifdef __GRPC_STAT__
    std::cout << result->DebugString() << std::endl;

    tc::InferStat infer_stat;
    grpc_client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "completed_request_count "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "cumulative_total_request_time_ns "
              << infer_stat.cumulative_total_request_time_ns << std::endl;
    std::cout << "cumulative_send_time_ns "
              << infer_stat.cumulative_send_time_ns << std::endl;
    std::cout << "cumulative_receive_time_ns "
              << infer_stat.cumulative_receive_time_ns << std::endl;

    inference::ModelStatisticsResponse model_stat;
    grpc_client->ModelInferenceStatistics(&model_stat, model_name);
    std::cout << "======Model Statistics======" << std::endl;
    std::cout << model_stat.DebugString() << std::endl;
#endif
    cv::eigen2cv(results, output_mat);
    end = std::chrono::steady_clock::now();
    dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "postprocess time: " << dr_s << "ms" << std::endl;
}

void grpc_test(GRPC_CLIENT &grpc_client,
               const ModelInfo &model_info,
               const tc::Headers &header,
               const size_t batch_size,
               const Mat &origin_mat,
               const Tensor3f &bg,
               Mat &output_mat,
               bool verbose = false)
{
    tc::Error err;

    std::vector<int64_t> shape;
    model_shape(model_info, batch_size, shape);

    tc::InferInput *input;
    err = tc::InferInput::Create(&input, model_info.input_name_, shape, model_info.input_datatype_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get input: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);

    tc::InferRequestedOutput *output;
    err = tc::InferRequestedOutput::Create(&output, model_info.output_name_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get output: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

    std::vector<tc::InferInput *> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput *> outputs = {output_ptr.get()};

    tc::InferOptions options("ppseg_onnx");
    options.model_version_ = "1";

    std::vector<uint8_t> input_data;
    Preprocess(origin_mat,
               model_info.input_format_,
               model_info.type1_,
               model_info.type3_,
               model_info.input_c_,
               cv::Size(model_info.input_w_, model_info.input_h_),
               ScaleType::INCEPTION,
               &input_data);

    err = input_ptr->AppendRaw(input_data);

    tc::InferResult *result;
    do_inference(grpc_client,
                 model_info,
                 inputs,
                 outputs,
                 options,
                 header,
                 result,
                 origin_mat,
                 bg,
                 output_mat,
                 verbose);
}

int camera_seg(char **argv)
{
    size_t batch_size = std::stoi(argv[4]);

    bool verbose = false;
    if (std::string(argv[2]).compare("verbose") == 0)
        verbose = true;
    else
        verbose = false;

    tc::Headers headers;
    auto client = create_grpc_client(headers, argv[1], batch_size, verbose);

    inference::ModelMetadataResponse model_metadata;
    inference::ModelConfigResponse model_config;
    ModelInfo model_info;
    get_model_info_from_server(client, model_info, headers, model_metadata, model_config, batch_size);

    std::vector<int64_t> shape;
    model_shape(model_info, batch_size, shape);

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

    Tensor3f bg_tensor = HumanSegmentaion::GenerateBg(bg, frame_size);

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

        grpc_test(client, model_info, headers, batch_size, frame, bg_tensor, output, verbose);

        auto end = std::chrono::steady_clock::now();
        double dr_s = std::chrono::duration<double, std::milli>(end - start).count();

        auto fps = 1000 / dr_s;
        std::string text = "FPS(";
        text.append("GRPC): ")
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

int main(int argc, char **argv)
{
    camera_seg(argv);
    // matting(argv);
}
