#include "human_segmentation.h"

onnx::cv::HumanSegmentaion::HumanSegmentaion(const char* model_path, Ort::Env env):
    _ort_env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, model_path)),
    _memory_info_handler(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
}


onnx::cv::HumanSegmentaion::~HumanSegmentaion()
{
}