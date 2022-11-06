#include "human_segmentation.h"

using namespace onnx::hs;

HumanSegmentaion::HumanSegmentaion(const char *modelPath,
                                   int numThreads)
    : m_ortEnv(Ort::Env(ORT_LOGGING_LEVEL_ERROR, modelPath)),
      m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      m_numThreads(numThreads),
      m_ortSession(std::make_unique<Ort::Session>(m_ortEnv, modelPath, initSessionOptions()))

{
}

Ort::SessionOptions HumanSegmentaion::initSessionOptions()
{
    Ort::SessionOptions session_options;
#ifdef __ANDROID__
    OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0x001);
#elif __CUDA__
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0x001);
#endif
    session_options.SetIntraOpNumThreads(this->m_numThreads);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(4);

    return session_options;
}

void HumanSegmentaion::detect(Mat &frame, Tensor3f &bgTensor, Mat &matted)
{
    auto start = std::chrono::steady_clock::now();
    auto origin_shape = frame.size();
    auto origin_mat = frame.clone();

    Tensor4f preprocessed = this->preprocess(frame);
    float *inputData = preprocessed.data();

    std::array<float, OUTPUT_TENSOR_SIZE> output{};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        inputData,
        INPUT_TENSOR_SIZE,
        MODEL_INPUT_SHAPE.data(),
        MODEL_INPUT_SHAPE.size());

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        output.data(),
        OUTPUT_TENSOR_SIZE,
        MODEL_OUTPUT_SHAPE.data(),
        MODEL_OUTPUT_SHAPE.size());

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Create Tensor time: " << dr_s << "ms" << std::endl;
    start = std::chrono::steady_clock::now();

    m_ortSession->Run(
        Ort::RunOptions{nullptr},
        &MODEL_INPUT_NAMES,
        &input_tensor,
        1,
        &MODEL_OUTPUT_NAMES,
        &output_tensor,
        1);

    end = std::chrono::steady_clock::now();
    dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Inference time: " << dr_s << "ms" << std::endl;

    this->postprocess(output, origin_mat, bgTensor, matted);
}

void HumanSegmentaion::postprocess(std::array<float, OUTPUT_TENSOR_SIZE> &output,
                                   cv::Mat &originMat,
                                   Tensor3f &bg_tensor,
                                   cv::Mat &matted)
{
    auto start = std::chrono::steady_clock::now();
    auto size = originMat.size();

    Tensor3f origin_tensor(originMat.cols, originMat.rows, 3);
    cv::cv2eigen(originMat, origin_tensor);

    std::array<float, SHAPE> scroe_map{};
    auto k = output.data();
    std::copy(k + SHAPE, k + OUTPUT_TENSOR_SIZE, scroe_map.data());

    Eigen::TensorMap<Tensor3f> score_tensor{scroe_map.data(), 1, 224, 398};

    // Transpose
    Shape3i shuffling({1, 2, 0});
    Tensor3f scrore_transposed = score_tensor.shuffle(shuffling);

    Mat tmp;
    cv::eigen2cv(scrore_transposed, tmp);
    cv::resize(tmp, tmp, size);

    Tensor3f alpha(size.height, size.width, 1);
    cv::cv2eigen(tmp, alpha);

    Shape3i bcast = {1, 1, 3};
    Tensor3f ab = alpha.broadcast(bcast);

    Tensor3f comb = ab * origin_tensor + (1 - ab) * bg_tensor;
    Eigen::Tensor<cv::uint8_t, 3, Eigen::RowMajor> results = comb.cast<cv::uint8_t>();

    // Mat r_mat;
    cv::eigen2cv(results, matted);
    std::cout << "Matted channels: " << matted.channels() << " depth " << matted.depth() << std::endl;

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Postprocess time: " << dr_s << "ms" << std::endl;
}

void HumanSegmentaion::normalize(cv::Mat &img, Tensor3f &tensor)
{
    cv::cv2eigen(img, tensor);
    tensor = tensor / 255.0f;
    tensor = tensor - 0.5f;
    tensor = tensor / 0.5f;
}

Tensor4f HumanSegmentaion::preprocess(cv::Mat &frame)
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
    Tensor4f reshaped = transposed.reshape(MODEL_INPUT_SHAPE);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Preprocess time: " << dr_s << "ms" << std::endl;

    return reshaped;
}

Tensor3f HumanSegmentaion::GenerateBg(cv::Mat &bg, cv::Size &size)
{
    cv::resize(bg, bg, size);
    static Tensor3f bg_tensor(size.height, size.width, 3);
    cv::cv2eigen(bg, bg_tensor);
    return bg_tensor;
}

// std::cout << "Result shape("
//           << result.dimension(0)
//           << "x" << result.dimension(1)
//           << "x" << result.dimension(2)
//           << "x" << result.dimension(3)
//           << ")" << std::endl;
// std::cout << result(0, 0, 0, 191) << std::endl;
// std::cout << result.size() << std::endl;

// auto input_name = _ort_session->GetInputNameAllocated(0, _allocator);
// std::cout << _ort_session->GetInputCount() << "," << input_name.get() << std::endl;
// auto output_name = _ort_session->GetOutputNameAllocated(0, _allocator);
// std::cout << _ort_session->GetOutputCount() << "," << output_name.get() << std::endl;

// auto k = _ort_session->GetOutputTypeInfo(0);
// auto d = k.GetTensorTypeAndShapeInfo();
// std::cout << d.GetShape().size() << std::endl;