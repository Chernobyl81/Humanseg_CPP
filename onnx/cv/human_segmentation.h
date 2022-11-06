#ifndef HUMAN_SEGMENTATION_H
#define HUMAN_SEGMENTATION_H

#include <memory>
#include <chrono>
#include "onnxruntime_cxx_api.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

using namespace cv;

namespace onnx
{
    namespace hs
    {
        class HumanSegmentaion
        {
        private:
            Ort::Env _ort_env;
            Ort::AllocatorWithDefaultOptions _allocator;
            Ort::MemoryInfo _memory_info;
            std::unique_ptr<Ort::Session> _ort_session;
            size_t _num_threads;
            
            std::array<int64_t, 4> _inputShape{1, 3, 224, 398};
            std::array<int64_t, 4> _outputShape{1, 2, 224, 398};

            const char *input_names = {"x"};
            const char *output_names = {"save_infer_model/scale_0.tmp_1"};

            static const int WIDTH = 398;
            static const int HEIGHT = 224;
            static const int SHAPE = 224 * 398;
            static const size_t INPUT_TENSOR_SIZE = WIDTH * HEIGHT * 3;
            static const size_t OUTPUT_TENSOR_SIZE = WIDTH * HEIGHT * 2;

            Ort::SessionOptions initSessionOptions();
            Eigen::Tensor<float, 4, Eigen::RowMajor> preprocess(cv::Mat &img);

            void normalize(Mat &img, Eigen::Tensor<float, 3, Eigen::RowMajor> &tensor);

            void postprocess(std::array<float, OUTPUT_TENSOR_SIZE> &output,
                             cv::Mat &origin_mat,
                             Eigen::Tensor<float, 3, Eigen::RowMajor> &bg_tensor,
                             cv::Mat &matted
                             );

        public:
            HumanSegmentaion(const char *model_path, int num_threads = 2);
            ~HumanSegmentaion() = default;

            void detect(Mat &image, Eigen::Tensor<float, 3, Eigen::RowMajor> &, Mat&);
            static Eigen::Tensor<float, 3, Eigen::RowMajor> GenerateBg(Mat &bg, Size &size);
        };
    }
}

#endif