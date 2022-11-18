#ifndef HUMAN_SEGMENTATION_H
#define HUMAN_SEGMENTATION_H

#include <memory>
#include <chrono>
#include "onnxruntime_cxx_api.h"
#include "tensorrt_provider_factory.h"
// #include "Eigen/Core"
// #include "Eigen/Dense"
// #include "../../build/config.h"
// #include "opencv2/opencv.hpp"
// #include "opencv2/core/eigen.hpp"
#include "../onnx/core/human_seg_model_info.h"


using namespace cv;
using namespace onnx::core;

namespace onnx
{
    namespace hs
    {
        // typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
        // typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
        // typedef Eigen::array<int, 3> Shape3i;
        // typedef std::array<int64_t, 4> ModelShape;

        class HumanSegmentaion
        {
        public:
            enum class Device
            {
                CPU = 0,
                CUDA,
                TensorRT
            };

        private:
            const HumanSegModelInfo m_model_info;
            // static const int WIDTH = 398;
            // static const int HEIGHT = 224;
            // static constexpr int SHAPE = WIDTH * HEIGHT;
            // static constexpr size_t INPUT_TENSOR_SIZE = SHAPE * 3;
            // static constexpr size_t OUTPUT_TENSOR_SIZE = SHAPE * 2;

            // const ModelShape MODEL_INPUT_SHAPE{1, 3, HEIGHT, WIDTH};
            // const ModelShape MODEL_OUTPUT_SHAPE{1, 2, HEIGHT, WIDTH};

            // const char *MODEL_INPUT_NAMES = {"x"};
            // const char *MODEL_OUTPUT_NAMES = {"save_infer_model/scale_0.tmp_1"};

            Ort::Env m_ortEnv;
            Ort::AllocatorWithDefaultOptions m_allocator;
            Ort::MemoryInfo m_memoryInfo;

            size_t m_numThreads;
            std::unique_ptr<Ort::Session> m_ortSession;

            Ort::SessionOptions initSessionOptions(Device device);
            const Tensor4f preprocess(cv::Mat &frame);

            void normalize(Mat &frame, Tensor3f &output) const;

            void postprocess(const std::array<float, HumanSegModelInfo::OUTPUT_TENSOR_SIZE> &modelOutputs,
                             const cv::Mat &originFrame,
                             const Tensor3f &bgTensor,
                             cv::Mat &matted);

        public:
            explicit HumanSegmentaion(const char *modelPath, int numThreads = 2, Device device = Device::CPU);
            ~HumanSegmentaion() = default;

            void detect(Mat &frame, const Tensor3f &bgTensor, Mat &matted);
            static const Tensor3f GenerateBg(Mat &bg, const Size &frameSize);
            static const Device StringToDevice(const std::string& s) {
                if (s == "CUDA")
                    return Device::CUDA;
                else if (s == "TensorRT")
                    return Device::TensorRT;
                else
                    return Device::CPU;
            }
        };
    }
}

#endif