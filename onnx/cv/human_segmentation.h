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
            Ort::MemoryInfo _memory_info_handler;
            std::unique_ptr<Ort::Session> _ort_session;
            size_t _num_threads;
           
            Ort::SessionOptions initSessionOptions();
            void normalize(const Mat &img, Eigen::Tensor<float, 3, Eigen::RowMajor> &tensor);
            Eigen::Tensor<float, 4, Eigen::RowMajor> preprocess(cv::Mat &img);

        public:
            HumanSegmentaion(const char *model_path, size_t num_threads = 2);
            ~HumanSegmentaion() = default;

            void detect(Mat& image);
        };
    }
}

#endif