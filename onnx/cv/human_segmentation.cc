#include "human_segmentation.h"

using namespace onnx::hs;

HumanSegmentaion::HumanSegmentaion(const char *model_path,
                                   int num_threads)
    : _ort_env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, model_path)),
      _memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      _ort_session(std::make_unique<Ort::Session>(_ort_env, model_path, initSessionOptions())),
      _num_threads(num_threads)
{
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


void HumanSegmentaion::detect(Mat &frame, Eigen::Tensor<float, 3, Eigen::RowMajor>& bg_tensor, Mat& matted)
{
    auto origin_shape = frame.size();
    std::cout << "input width: " << origin_shape.width << " input height: " << origin_shape.height << std::endl;
    auto origin_mat = frame.clone();

    Eigen::Tensor<float, 4, Eigen::RowMajor> preprocessed = this->preprocess(frame);
    float *inputData = preprocessed.data();

    std::array<float, OUTPUT_TENSOR_SIZE> output{};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        _memory_info,
        inputData,
        INPUT_TENSOR_SIZE,
        _inputShape.data(),
        _inputShape.size());

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        _memory_info,
        output.data(),
        OUTPUT_TENSOR_SIZE,
        _outputShape.data(),
        _outputShape.size());

    auto start = std::chrono::steady_clock::now();

    _ort_session->Run(
        Ort::RunOptions{nullptr},
        &input_names,
        &input_tensor,
        1,
        &output_names,
        &output_tensor,
        1);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Inference time: " << dr_s << "ms" << std::endl;

    this->postprocess(output, origin_mat, bg_tensor, matted);
    // cv::imwrite("/home/david/Desktop/rrrrrr.jpeg" ,this->postprocess(output, origin_mat, bg_tensor));
}

void HumanSegmentaion::postprocess(std::array<float, OUTPUT_TENSOR_SIZE> &output, 
    cv::Mat &originMat,
    Eigen::Tensor<float, 3, Eigen::RowMajor>& bg_tensor,
    cv::Mat &matted)
{
    auto start = std::chrono::steady_clock::now();
    auto size = originMat.size();
    
    Eigen::Tensor<float, 3, Eigen::RowMajor> origin_tensor(originMat.cols, originMat.rows, 3);
    cv::cv2eigen(originMat, origin_tensor);
   
    std::array<float, SHAPE> scroe_map{};
    auto k = output.data();
    std::copy(k + SHAPE, k + OUTPUT_TENSOR_SIZE, scroe_map.data());

    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> score_tensor{scroe_map.data(), 1, 224, 398};
   
    // Transpose
    Eigen::array<int, 3> shuffling({1, 2, 0});
    Eigen::Tensor<float, 3, Eigen::RowMajor> scrore_transposed = score_tensor.shuffle(shuffling);
    
    Mat tmp;
    cv::eigen2cv(scrore_transposed, tmp);
    cv::resize(tmp, tmp, size);
   
    Eigen::Tensor<float, 3, Eigen::RowMajor> alpha(size.height, size.width, 1);
    cv::cv2eigen(tmp, alpha);
    
    Eigen::array<int, 3> bcast = {1, 1, 3};
    Eigen::Tensor<float, 3, Eigen::RowMajor> ab = alpha.broadcast(bcast);

    Eigen::Tensor<float, 3, Eigen::RowMajor> comb = ab * origin_tensor + (1 - ab) * bg_tensor;
    Eigen::Tensor<cv::uint8_t, 3, Eigen::RowMajor> results = comb.cast<cv::uint8_t>();
    
    //Mat r_mat;
    cv::eigen2cv(results, matted);
    std::cout << "Matted channels: " << matted.channels() << " depth " << matted.depth() << std::endl;

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Postprocess time: " << dr_s << "ms" << std::endl;
    
   // return r_mat;
}

void HumanSegmentaion::normalize(cv::Mat &img, Eigen::Tensor<float, 3, Eigen::RowMajor> &tensor)
{
    cv::cv2eigen(img, tensor);
    tensor = tensor / 255.0f;
    tensor = tensor - 0.5f;
    tensor = tensor / 0.5f;
}


Eigen::Tensor<float, 4, Eigen::RowMajor> HumanSegmentaion::preprocess(cv::Mat &img)
{
    auto start = std::chrono::steady_clock::now();

    cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
    cv::resize(img, img, cv::Size(398, 224));

    Eigen::Tensor<float, 3, Eigen::RowMajor> imageTensor;
    this->normalize(img, imageTensor);

    // Tanspose
    Eigen::array<int, 3> shuffling({2, 0, 1});
    Eigen::Tensor<float, 3, Eigen::RowMajor> transposed = imageTensor.shuffle(shuffling);
   
    // Add a dimension
    Eigen::Tensor<float, 4, Eigen::RowMajor> reshaped = transposed.reshape(_inputShape);

    auto end = std::chrono::steady_clock::now();
    double dr_s = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Preprocess time: " << dr_s << "ms" << std::endl;

    return reshaped;
}

Eigen::Tensor<float, 3, Eigen::RowMajor> HumanSegmentaion::GenerateBg(cv::Mat& bg, cv::Size& size)
{
    cv::resize(bg, bg, size);
    static Eigen::Tensor<float, 3, Eigen::RowMajor> bg_tensor(size.height, size.width, 3);
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