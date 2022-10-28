#ifndef HUMAN_SEGMENTATION_H
#define HUMAN_SEGMENTATION_H

#include <memory>
#include "onnxruntime_cxx_api.h"

namespace onnx::cv
{
    class HumanSegmentaion
    {
    private:
        Ort::Env _ort_env;
        Ort::AllocatorWithDefaultOptions _allocator;
        Ort::MemoryInfo _memory_info_handler;

        std::unique_ptr<Ort::Session> _ort_session;


    public:
        HumanSegmentaion(const char* model_path, Ort::Env env);
        ~HumanSegmentaion();
    };
        
}

#endif