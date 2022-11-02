#include "human_segmentation.h"

using namespace onnx::hs;

HumanSegmentaion::HumanSegmentaion(const char *model_path,
                                   size_t num_threads)
    : _ort_env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, model_path)),
      _num_threads(num_threads),
      _memory_info_handler(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      _ort_session(std::make_unique<Ort::Session>(_ort_env, model_path, initSessionOptions()))

{

}

void HumanSegmentaion::detect(Mat &image)
{
    Eigen::Tensor<float, 4, Eigen::RowMajor> result = this->preprocess(image);

    std::cout << "Result shape("
              << result.dimension(0)
              << "x" << result.dimension(1)
              << "x" << result.dimension(2)
              << "x" << result.dimension(3)
              << ")" << std::endl;
    std::cout << result(0, 0, 0, 191) << std::endl;
    std::cout << result.size() << std::endl;
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = _ort_session->GetInputNameAllocated(0, allocator);
    std::cout << _ort_session->GetInputCount() << "," << input_name.get() << std::endl;
    auto output_name = _ort_session->GetOutputNameAllocated(0, allocator);
    std::cout << _ort_session->GetOutputCount() << "," << output_name.get() << std::endl;
      // Ort::Value inputs = Ort::Value::CreateTensor<float>(_memory_info_handler,k, result.size());
    // _ort_session->Run(
    //     Ort::RunOptions{nullptr},
    //     "x",
    //     1,
    //     1,
    //     _ort_session->GetOutputCount(),
    //     _ort_session->GetOutputNameAllocated(0)))
}

Ort::SessionOptions HumanSegmentaion::initSessionOptions()
{
    Ort::SessionOptions session_options;
#ifdef __ANDROID__
    OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0x001);
#endif
    session_options.SetIntraOpNumThreads(this->_num_threads);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(4);
    return session_options;
}

void HumanSegmentaion::normalize(const cv::Mat &img, Eigen::Tensor<float, 3, Eigen::RowMajor> &tensor)
{
    cv::cv2eigen(img, tensor);
    tensor = tensor / 255.0f;
    tensor = tensor - 0.5f;
    tensor = tensor / 0.5f;

    Eigen::array<int, 3> shuffling({2, 0, 1});
    Eigen::Tensor<float, 3, Eigen::RowMajor> transposed = tensor.shuffle(shuffling);
    tensor = transposed;
}

Eigen::Tensor<float, 4, Eigen::RowMajor> HumanSegmentaion::preprocess(cv::Mat &img)
{
    auto start = std::chrono::steady_clock::now();

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(398, 224));

    Eigen::Tensor<float, 3, Eigen::RowMajor> tensor;
    this->normalize(img, tensor);

    // Add a dimension
    Eigen::array<int, 4> dims{{1, 3, 398, 224}};
    Eigen::Tensor<float, 4, Eigen::RowMajor> result = tensor.reshape(dims);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Preprocess time: " << dr_s << "ms" << std::endl;
    // cv::eigen2cv(tensor, img);
    return result;
}