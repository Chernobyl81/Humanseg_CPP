#include "human_segmentation.h"

using namespace onnx::hs;
using namespace onnx::core;

HumanSegmentaion::HumanSegmentaion(const char *modelPath,
                                   int numThreads,
                                   Device device)
    : m_ortEnv(Ort::Env(ORT_LOGGING_LEVEL_ERROR, modelPath)),
      m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      m_numThreads(numThreads),
      m_ortSession(std::make_unique<Ort::Session>(m_ortEnv, modelPath, initSessionOptions(device)))

{
}

Ort::SessionOptions HumanSegmentaion::initSessionOptions(Device device)
{
    Ort::SessionOptions sessionOptions;
    // #ifdef __ANDROID__
    //     OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0x000);
    // #endif
    if (device == Device::CUDA)
    {
        std::cout << "Run with CUDA" << std::endl;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0x000));
    }
    else if (device == Device::TensorRT)
    {
        std::cout << "Run with TR" << std::endl;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0x000));
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0x000));
    }

    sessionOptions.SetIntraOpNumThreads(this->m_numThreads);
    sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetLogSeverityLevel(4);

    return sessionOptions;
}

void HumanSegmentaion::detect(Mat &frame, const Tensor3f &bgTensor, Mat &matted)
{
    auto start = std::chrono::steady_clock::now();
    auto origin_shape = frame.size();
    auto origin_mat = frame.clone();

    auto preprocessed = this->preprocess(frame);
    float *inputData = preprocessed.data();

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        inputData,
        HumanSegModelInfo::INPUT_TENSOR_SIZE,
        m_model_info.MODEL_INPUT_SHAPE.data(),
        m_model_info.MODEL_INPUT_SHAPE.size());

    std::array<float, HumanSegModelInfo::OUTPUT_TENSOR_SIZE> output{};
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        output.data(),
        HumanSegModelInfo::OUTPUT_TENSOR_SIZE,
        m_model_info.MODEL_OUTPUT_SHAPE.data(),
        m_model_info.MODEL_OUTPUT_SHAPE.size());

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << ">>>>>> \nCreate Tensor uses: " << dr_s << "ms" << std::endl;
    start = std::chrono::steady_clock::now();

    m_ortSession->Run(
        Ort::RunOptions{nullptr},
        &m_model_info.MODEL_INPUT_NAMES,
        &input_tensor,
        1,
        &m_model_info.MODEL_OUTPUT_NAMES,
        &outputTensor,
        1);

    end = std::chrono::steady_clock::now();
    dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Inference time: " << dr_s << "ms" << std::endl;

    this->postprocess(output, origin_mat, bgTensor, matted);
}

void HumanSegmentaion::postprocess(const std::array<float, HumanSegModelInfo::OUTPUT_TENSOR_SIZE> &output,
                                   const cv::Mat &originMat,
                                   const Tensor3f &bgTensor,
                                   cv::Mat &matted)
{
    auto start = std::chrono::steady_clock::now();
    auto size = originMat.size();

    Tensor3f origin_tensor(originMat.cols, originMat.rows, 3);
    cv::cv2eigen(originMat, origin_tensor);

    std::array<float, HumanSegModelInfo::SHAPE> scroe_map{};
    auto k = output.data();
    std::copy(k + HumanSegModelInfo::SHAPE, k + HumanSegModelInfo::OUTPUT_TENSOR_SIZE, scroe_map.data());

    Eigen::TensorMap<Tensor3f> score_tensor{scroe_map.data(), 1, 224, 398};

    // Transpose
    const Shape3i shuffling({1, 2, 0});
    Tensor3f scrore_transposed = score_tensor.shuffle(shuffling);

    Mat tmp;
    cv::eigen2cv(scrore_transposed, tmp);
    cv::resize(tmp, tmp, size);

    Tensor3f alpha(size.height, size.width, 1);
    cv::cv2eigen(tmp, alpha);

    Shape3i bcast = {1, 1, 3};
    Tensor3f ab = alpha.broadcast(bcast);

    Tensor3f comb = ab * origin_tensor + (1 - ab) * bgTensor;
    Eigen::Tensor<cv::uint8_t, 3, Eigen::RowMajor> results = comb.cast<cv::uint8_t>();

    cv::eigen2cv(results, matted);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Postprocess time: " << dr_s << "ms" << std::endl;
}

void HumanSegmentaion::normalize(cv::Mat &img, Tensor3f &tensor) const
{
    cv::cv2eigen(img, tensor);
    tensor = tensor / 255.0f;
    tensor = tensor - 0.5f;
    tensor = tensor / 0.5f;
}

const Tensor4f HumanSegmentaion::preprocess(cv::Mat &frame)
{
    auto start = std::chrono::steady_clock::now();

    cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
    cv::resize(frame, frame, cv::Size(398, 224));

    Tensor3f imageTensor;
    this->normalize(frame, imageTensor);

    // Tanspose
    Shape3i shuffling({2, 0, 1});
    Tensor3f transposed = imageTensor.shuffle(shuffling);

    // Add a dimension
    Tensor4f reshaped = transposed.reshape(m_model_info.MODEL_INPUT_SHAPE);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Preprocess time: " << dr_s << "ms" << std::endl;

    return reshaped;
}

const Tensor3f HumanSegmentaion::GenerateBg(cv::Mat &bg, const cv::Size &size)
{
    cv::resize(bg, bg, size);
    static Tensor3f bg_tensor(size.height, size.width, 3);
    cv::cv2eigen(bg, bg_tensor);
    return bg_tensor;
}