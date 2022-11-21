#ifndef MODEL_INFO_H
#define MODEL_INFO_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::array<int, 3> Shape3i;
typedef std::array<int64_t, 4> ModelShape;

namespace onnx
{
    namespace core
    {
        struct HumanSegModelInfo
        {
            
            static const int WIDTH = 398;
            static const int HEIGHT = 224;
            static constexpr int SHAPE = WIDTH * HEIGHT;

            static constexpr size_t INPUT_TENSOR_SIZE = SHAPE * 3;
            static constexpr size_t OUTPUT_TENSOR_SIZE = SHAPE * 2;

            const ModelShape MODEL_INPUT_SHAPE{1, 3, HEIGHT, WIDTH};
            const ModelShape MODEL_OUTPUT_SHAPE{1, 2, HEIGHT, WIDTH};

            const char *MODEL_INPUT_NAMES = {"x"};
            const char *MODEL_OUTPUT_NAMES = {"save_infer_model/scale_0.tmp_1"};

        };

    } // namespace core

} // namespace onnx

#endif