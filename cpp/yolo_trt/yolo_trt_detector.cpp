#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.2F;
constexpr float kIouThreshold = 0.45F;

const std::vector<std::string> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* object) const {
        delete object;
    }
};

struct Detection {
    std::string label;
    float confidence;
    int x1;
    int y1;
    int x2;
    int y2;
};

struct LetterboxMeta {
    float ratio;
    float pad_x;
    float pad_y;
    int original_width;
    int original_height;
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check_cuda(cudaError_t code, const std::string& op) {
    if (code != cudaSuccess) {
        throw std::runtime_error(op + " failed: " + cudaGetErrorString(code));
    }
}

std::string json_escape(const std::string& input) {
    std::ostringstream escaped;
    for (char ch : input) {
        switch (ch) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << ch; break;
        }
    }
    return escaped.str();
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open engine: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    file.read(bytes.data(), size);
    return bytes;
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("unresolved TensorRT tensor shape");
        }
        total *= static_cast<size_t>(dims.d[i]);
    }
    return total;
}

float iou(const Detection& a, const Detection& b) {
    const int x1 = std::max(a.x1, b.x1);
    const int y1 = std::max(a.y1, b.y1);
    const int x2 = std::min(a.x2, b.x2);
    const int y2 = std::min(a.y2, b.y2);
    const float inter = static_cast<float>(std::max(0, x2 - x1) * std::max(0, y2 - y1));
    const float area_a = static_cast<float>(std::max(0, a.x2 - a.x1) * std::max(0, a.y2 - a.y1));
    const float area_b = static_cast<float>(std::max(0, b.x2 - b.x1) * std::max(0, b.y2 - b.y1));
    return inter / std::max(area_a + area_b - inter, 1e-6F);
}

std::vector<Detection> nms(std::vector<Detection> detections) {
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<Detection> keep;
    std::vector<bool> removed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        keep.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!removed[j] && iou(detections[i], detections[j]) > kIouThreshold) {
                removed[j] = true;
            }
        }
    }
    if (keep.size() > 100) {
        keep.resize(100);
    }
    return keep;
}

std::vector<float> prepare_input(const cv::Mat& image, LetterboxMeta& meta) {
    meta.original_width = image.cols;
    meta.original_height = image.rows;
    const float ratio = std::min(static_cast<float>(kInputSize) / image.cols, static_cast<float>(kInputSize) / image.rows);
    const int new_width = static_cast<int>(std::round(image.cols * ratio));
    const int new_height = static_cast<int>(std::round(image.rows * ratio));
    meta.ratio = ratio;
    meta.pad_x = (kInputSize - new_width) / 2.0F;
    meta.pad_y = (kInputSize - new_height) / 2.0F;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(kInputSize, kInputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    const int x0 = static_cast<int>(std::round(meta.pad_x - 0.1F));
    const int y0 = static_cast<int>(std::round(meta.pad_y - 0.1F));
    resized.copyTo(canvas(cv::Rect(x0, y0, new_width, new_height)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    std::vector<float> input(3 * kInputSize * kInputSize);
    const int plane = kInputSize * kInputSize;
    for (int y = 0; y < kInputSize; ++y) {
        for (int x = 0; x < kInputSize; ++x) {
            const cv::Vec3b pixel = canvas.at<cv::Vec3b>(y, x);
            const int offset = y * kInputSize + x;
            input[offset] = pixel[0] / 255.0F;
            input[plane + offset] = pixel[1] / 255.0F;
            input[2 * plane + offset] = pixel[2] / 255.0F;
        }
    }
    return input;
}

std::vector<Detection> postprocess(const std::vector<float>& output, const nvinfer1::Dims& dims, const LetterboxMeta& meta) {
    int channels = 0;
    int anchors = 0;
    bool channels_first = false;
    if (dims.nbDims == 3 && dims.d[0] == 1) {
        channels = dims.d[1];
        anchors = dims.d[2];
        channels_first = true;
    } else if (dims.nbDims == 2) {
        channels = dims.d[0];
        anchors = dims.d[1];
        channels_first = true;
    } else {
        throw std::runtime_error("unsupported YOLO output shape");
    }
    if (channels != 84 && channels != 85) {
        throw std::runtime_error("unsupported YOLO channel count");
    }

    auto value_at = [&](int anchor, int channel) -> float {
        if (channels_first) {
            return output[static_cast<size_t>(channel) * anchors + anchor];
        }
        return output[static_cast<size_t>(anchor) * channels + channel];
    };

    std::vector<Detection> detections;
    for (int anchor = 0; anchor < anchors; ++anchor) {
        float best_score = 0.0F;
        int best_class = -1;
        const float objectness = channels == 85 ? value_at(anchor, 4) : 1.0F;
        const int class_offset = channels == 85 ? 5 : 4;
        for (int cls = 0; cls < 80; ++cls) {
            const float score = value_at(anchor, class_offset + cls) * objectness;
            if (score > best_score) {
                best_score = score;
                best_class = cls;
            }
        }
        if (best_score < kConfidenceThreshold || best_class < 0) {
            continue;
        }

        const float cx = value_at(anchor, 0);
        const float cy = value_at(anchor, 1);
        const float width = value_at(anchor, 2);
        const float height = value_at(anchor, 3);
        float x1 = (cx - width / 2.0F - meta.pad_x) / meta.ratio;
        float y1 = (cy - height / 2.0F - meta.pad_y) / meta.ratio;
        float x2 = (cx + width / 2.0F - meta.pad_x) / meta.ratio;
        float y2 = (cy + height / 2.0F - meta.pad_y) / meta.ratio;
        x1 = std::clamp(x1, 0.0F, static_cast<float>(meta.original_width));
        x2 = std::clamp(x2, 0.0F, static_cast<float>(meta.original_width));
        y1 = std::clamp(y1, 0.0F, static_cast<float>(meta.original_height));
        y2 = std::clamp(y2, 0.0F, static_cast<float>(meta.original_height));

        detections.push_back(Detection{
            kCocoClasses[static_cast<size_t>(best_class)],
            best_score,
            static_cast<int>(std::round(x1)),
            static_cast<int>(std::round(y1)),
            static_cast<int>(std::round(x2)),
            static_cast<int>(std::round(y2)),
        });
    }
    return nms(std::move(detections));
}

class YoloTrtWorker {
  public:
    explicit YoloTrtWorker(const std::string& engine_path) {
        const auto start = Clock::now();
        bytes_ = read_file(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("failed to create TensorRT runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(bytes_.data(), bytes_.size()));
        if (!engine_) {
            throw std::runtime_error("failed to deserialize TensorRT engine");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("failed to create TensorRT execution context");
        }
        discover_tensors();
        context_->setInputShape(input_name_.c_str(), nvinfer1::Dims4{1, 3, kInputSize, kInputSize});
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        allocate_buffers();
        init_latency_ms_ = elapsed_ms(start);
    }

    ~YoloTrtWorker() {
        for (void* ptr : device_buffers_) {
            cudaFree(ptr);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    std::string detect_json(const std::string& image_path) {
        const auto total_start = Clock::now();
        try {
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                throw std::runtime_error("failed to read image: " + image_path);
            }
            LetterboxMeta meta{};
            std::vector<float> input = prepare_input(image, meta);

            const auto infer_start = Clock::now();
            check_cuda(cudaMemcpyAsync(device_buffers_[input_index_], input.data(), input_bytes_, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync H2D");
            if (!context_->enqueueV3(stream_)) {
                throw std::runtime_error("enqueueV3 returned false");
            }
            for (size_t i = 0; i < output_indices_.size(); ++i) {
                const size_t tensor_index = output_indices_[i];
                check_cuda(
                    cudaMemcpyAsync(host_outputs_[i].data(), device_buffers_[tensor_index], output_bytes_[i], cudaMemcpyDeviceToHost, stream_),
                    "cudaMemcpyAsync D2H");
            }
            check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            const double inference_ms = elapsed_ms(infer_start);
            const std::vector<Detection> detections = postprocess(host_outputs_.front(), output_dims_.front(), meta);
            const double total_ms = elapsed_ms(total_start);
            return result_json(true, detections, inference_ms, total_ms, "");
        } catch (const std::exception& exc) {
            return result_json(false, {}, 0.0, elapsed_ms(total_start), exc.what());
        }
    }

    double init_latency_ms() const {
        return init_latency_ms_;
    }

  private:
    void discover_tensors() {
        const int count = engine_->getNbIOTensors();
        for (int i = 0; i < count; ++i) {
            const char* name = engine_->getIOTensorName(i);
            tensor_names_.emplace_back(name);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
                input_index_ = static_cast<size_t>(i);
            } else {
                output_names_.emplace_back(name);
                output_indices_.push_back(static_cast<size_t>(i));
            }
        }
        if (input_name_.empty() || output_names_.empty()) {
            throw std::runtime_error("TensorRT engine must have one input and at least one output");
        }
    }

    nvinfer1::Dims tensor_shape(const std::string& name) {
        nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
        bool unresolved = false;
        for (int i = 0; i < dims.nbDims; ++i) {
            unresolved = unresolved || dims.d[i] <= 0;
        }
        if (unresolved) {
            dims = engine_->getTensorShape(name.c_str());
        }
        return dims;
    }

    void allocate_buffers() {
        device_buffers_.resize(tensor_names_.size(), nullptr);
        input_dims_ = tensor_shape(input_name_);
        input_bytes_ = volume(input_dims_) * sizeof(float);
        check_cuda(cudaMalloc(&device_buffers_[input_index_], input_bytes_), "cudaMalloc input");
        context_->setTensorAddress(input_name_.c_str(), device_buffers_[input_index_]);

        for (size_t output_slot = 0; output_slot < output_names_.size(); ++output_slot) {
            const std::string& name = output_names_[output_slot];
            const size_t tensor_index = output_indices_[output_slot];
            const nvinfer1::Dims dims = tensor_shape(name);
            if (engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("worker currently expects float output tensors");
            }
            const size_t bytes = volume(dims) * sizeof(float);
            output_dims_.push_back(dims);
            output_bytes_.push_back(bytes);
            host_outputs_.emplace_back(volume(dims));
            check_cuda(cudaMalloc(&device_buffers_[tensor_index], bytes), "cudaMalloc output");
            context_->setTensorAddress(name.c_str(), device_buffers_[tensor_index]);
        }
    }

    std::string result_json(bool ok, const std::vector<Detection>& detections, double inference_ms, double total_ms, const std::string& error) const {
        std::vector<std::string> labels;
        for (const auto& detection : detections) {
            if (std::find(labels.begin(), labels.end(), detection.label) == labels.end()) {
                labels.push_back(detection.label);
            }
        }
        if (labels.empty()) {
            labels.push_back("no_detection");
        }
        std::ostringstream out;
        out << "{\"ok\":" << (ok ? "true" : "false")
            << ",\"model\":\"yolov8n_tensorrt_cpp_fp16\""
            << ",\"runtime\":\"yolo_tensorrt_cpp\""
            << ",\"detected_labels\":[";
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i) out << ',';
            out << '"' << json_escape(labels[i]) << '"';
        }
        out << "],\"detections\":[";
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            if (i) out << ',';
            out << "{\"label\":\"" << json_escape(d.label) << "\",\"confidence\":" << d.confidence
                << ",\"box\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << "]}";
        }
        out << "],\"inference_latency_ms\":" << inference_ms
            << ",\"total_latency_ms\":" << total_ms
            << ",\"engine_init_latency_ms\":" << init_latency_ms_
            << ",\"error\":\"" << json_escape(error) << "\"}";
        return out.str();
    }

    Logger logger_;
    std::vector<char> bytes_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context_;
    cudaStream_t stream_{};
    std::vector<std::string> tensor_names_;
    std::vector<std::string> output_names_;
    std::vector<size_t> output_indices_;
    std::vector<void*> device_buffers_;
    std::vector<std::vector<float>> host_outputs_;
    std::vector<nvinfer1::Dims> output_dims_;
    std::vector<size_t> output_bytes_;
    nvinfer1::Dims input_dims_{};
    std::string input_name_;
    size_t input_index_{0};
    size_t input_bytes_{0};
    double init_latency_ms_{0.0};
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: yolo_trt_worker <engine_path>\n";
        return 2;
    }
    try {
        YoloTrtWorker worker(argv[1]);
        std::cerr << "ready engine_init_latency_ms=" << worker.init_latency_ms() << '\n';
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") {
                break;
            }
            std::cout << worker.detect_json(line) << std::endl;
        }
    } catch (const std::exception& exc) {
        std::cerr << "fatal: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
