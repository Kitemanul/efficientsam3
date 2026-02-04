/**
 * EfficientSAM3 ONNX Runtime C++ Inference
 *
 * End-to-end text-grounding pipeline:
 *   image + text → image_encoder → text_encoder → decoder → NMS → visualization
 *
 * Usage:
 *   ./sam3_infer --image test.jpg --prompt "person" \
 *               --onnx-dir exports_repvit_m0_9/ \
 *               --bpe-path assets/bpe_simple_vocab_16e6.txt.gz \
 *               --output result.png
 */

#include "simple_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ============================================================================
// Command-line argument parser (minimal)
// ============================================================================

struct Args {
    std::string image;
    std::string prompt;
    std::string onnx_dir   = "exports_repvit_m0_9";
    std::string output     = "result.png";
    std::string bpe_path;
    int    resolution      = 1008;
    float  score_threshold = 0.1f;
    float  nms_threshold   = 0.5f;
    int    top_k           = 10;
};

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
            return argv[++i];
        };
        if (key == "--image")            args.image = next();
        else if (key == "--prompt")      args.prompt = next();
        else if (key == "--onnx-dir")    args.onnx_dir = next();
        else if (key == "--output")      args.output = next();
        else if (key == "--bpe-path")    args.bpe_path = next();
        else if (key == "--resolution")  args.resolution = std::stoi(next());
        else if (key == "--score-threshold") args.score_threshold = std::stof(next());
        else if (key == "--nms-threshold")   args.nms_threshold = std::stof(next());
        else if (key == "--top-k")       args.top_k = std::stoi(next());
        else {
            std::cerr << "Unknown argument: " << key << "\n";
        }
    }
    if (args.image.empty() || args.prompt.empty()) {
        throw std::runtime_error("--image and --prompt are required");
    }
    return args;
}

// ============================================================================
// Image preprocessing
// ============================================================================

/**
 * Preprocess image to model input: resize → float32 → normalize [-1,1] → CHW.
 * Returns a flat vector of size [1 * 3 * resolution * resolution].
 */
static std::vector<float> preprocess_image(const cv::Mat& bgr, int resolution) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resolution, resolution));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat fp32;
    rgb.convertTo(fp32, CV_32FC3, 1.0 / 255.0);  // [0, 1]

    // Normalize: (x - 0.5) / 0.5 = 2x - 1 → [-1, 1]
    fp32 = (fp32 - 0.5f) * 2.0f;

    // HWC → CHW
    const int H = resolution, W = resolution;
    std::vector<float> chw(3 * H * W);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                chw[c * H * W + y * W + x] = fp32.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return chw;
}

// ============================================================================
// ONNX Runtime helpers
// ============================================================================

/** Create an ORT tensor from a float vector with given shape. */
static Ort::Value create_tensor(Ort::MemoryInfo& mem_info,
                                std::vector<float>& data,
                                const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(
        mem_info, data.data(), data.size(), shape.data(), shape.size());
}

/** Create an ORT tensor from an int64 vector with given shape. */
static Ort::Value create_tensor_int64(Ort::MemoryInfo& mem_info,
                                      std::vector<int64_t>& data,
                                      const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<int64_t>(
        mem_info, data.data(), data.size(), shape.data(), shape.size());
}

/** Run a session and return output tensors. */
static std::vector<Ort::Value> run_session(
    Ort::Session& session,
    const std::vector<const char*>& input_names,
    std::vector<Ort::Value>& input_tensors,
    const std::vector<const char*>& output_names) {
    return session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), input_tensors.data(), input_tensors.size(),
        output_names.data(), output_names.size());
}

// ============================================================================
// NMS
// ============================================================================

static std::vector<int> nms(const float* boxes, const float* scores, int n,
                            float iou_threshold) {
    // Sort indices by score descending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    auto iou = [&](int i, int j) -> float {
        float x1 = std::max(boxes[i * 4 + 0], boxes[j * 4 + 0]);
        float y1 = std::max(boxes[i * 4 + 1], boxes[j * 4 + 1]);
        float x2 = std::min(boxes[i * 4 + 2], boxes[j * 4 + 2]);
        float y2 = std::min(boxes[i * 4 + 3], boxes[j * 4 + 3]);
        float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_i = (boxes[i * 4 + 2] - boxes[i * 4 + 0]) *
                        (boxes[i * 4 + 3] - boxes[i * 4 + 1]);
        float area_j = (boxes[j * 4 + 2] - boxes[j * 4 + 0]) *
                        (boxes[j * 4 + 3] - boxes[j * 4 + 1]);
        return inter / (area_i + area_j - inter + 1e-8f);
    };

    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);
    for (int idx : order) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (int other : order) {
            if (!suppressed[other] && other != idx) {
                if (iou(idx, other) > iou_threshold) {
                    suppressed[other] = true;
                }
            }
        }
    }
    return keep;
}

// ============================================================================
// Visualization
// ============================================================================

static const cv::Scalar PALETTE[] = {
    {56, 56, 255},   {151, 157, 255}, {31, 112, 255},  {29, 178, 255},
    {49, 210, 207},  {10, 249, 72},   {23, 204, 146},  {134, 219, 61},
    {52, 147, 26},   {187, 212, 0},   {168, 153, 44},  {255, 194, 0},
    {147, 69, 52},   {255, 115, 100}, {236, 24, 0},    {255, 56, 132},
    {133, 0, 82},    {255, 56, 203},  {200, 149, 255}, {199, 55, 255},
};
static constexpr int PALETTE_SIZE = sizeof(PALETTE) / sizeof(PALETTE[0]);

static void draw_results(cv::Mat& img,
                          const std::vector<float>& scores,
                          const std::vector<std::array<float, 4>>& boxes,
                          const std::vector<cv::Mat>& masks,
                          float alpha = 0.5f) {
    for (size_t i = 0; i < scores.size(); ++i) {
        cv::Scalar color = PALETTE[i % PALETTE_SIZE];
        const auto& box = boxes[i];
        const cv::Mat& mask = masks[i];

        // Mask overlay
        cv::Mat colored(img.size(), CV_8UC3, color);
        cv::Mat blended;
        cv::addWeighted(img, 1.0 - alpha, colored, alpha, 0, blended);
        blended.copyTo(img, mask);

        // Mask contour
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(img, contours, -1, color, 2);

        // Box
        cv::Point tl(static_cast<int>(box[0]), static_cast<int>(box[1]));
        cv::Point br(static_cast<int>(box[2]), static_cast<int>(box[3]));
        cv::rectangle(img, tl, br, color, 2);

        // Label
        char label[32];
        std::snprintf(label, sizeof(label), "%.2f", scores[i]);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::rectangle(img, {tl.x, tl.y - ts.height - 6}, {tl.x + ts.width + 4, tl.y}, color, -1);
        cv::putText(img, label, {tl.x + 2, tl.y - 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        Args args = parse_args(argc, argv);

        // ---- 1. Load tokenizer ----
        std::string bpe_path = args.bpe_path;
        if (bpe_path.empty()) {
            bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz";
        }
        std::cout << "Loading tokenizer from: " << bpe_path << "\n";
        SimpleTokenizer tokenizer(bpe_path);

        // ---- 2. Load ONNX models ----
        std::cout << "Loading ONNX models from: " << args.onnx_dir << "\n";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam3");
        Ort::SessionOptions opts;
        // opts.SetIntraOpNumThreads(4);  // Adjust for TV hardware

        std::string img_model = args.onnx_dir + "/image_encoder.onnx";
        std::string txt_model = args.onnx_dir + "/text_encoder.onnx";
        std::string dec_model = args.onnx_dir + "/decoder.onnx";

        Ort::Session img_sess(env, img_model.c_str(), opts);
        Ort::Session txt_sess(env, txt_model.c_str(), opts);
        Ort::Session dec_sess(env, dec_model.c_str(), opts);
        std::cout << "  Models loaded.\n";

        // ---- 3. Load and preprocess image ----
        cv::Mat img_bgr = cv::imread(args.image);
        if (img_bgr.empty()) {
            throw std::runtime_error("Cannot read image: " + args.image);
        }
        int orig_h = img_bgr.rows, orig_w = img_bgr.cols;
        std::cout << "Image: " << args.image << " (" << orig_w << "x" << orig_h << ")\n";

        auto img_data = preprocess_image(img_bgr, args.resolution);
        std::vector<int64_t> img_shape = {1, 3, args.resolution, args.resolution};

        // ---- 4. Tokenize text ----
        auto token_batch = tokenizer.tokenize({args.prompt});
        auto& token_ids = token_batch[0];  // [77]
        int non_pad = 0;
        for (auto id : token_ids) { if (id != 0) ++non_pad; }
        std::cout << "Prompt: '" << args.prompt << "' -> " << non_pad << " tokens\n";
        std::vector<int64_t> tid_shape = {1, static_cast<int64_t>(token_ids.size())};

        // ---- 5. Run image encoder ----
        std::cout << "Running image encoder ...\n";
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> img_inputs;
        img_inputs.push_back(create_tensor(mem_info, img_data, img_shape));

        const char* img_in_names[]  = {"images"};
        const char* img_out_names[] = {"feat_4x", "feat_2x", "feat_1x", "pos_1x"};
        auto img_outputs = run_session(img_sess, {img_in_names, img_in_names + 1},
                                       img_inputs, {img_out_names, img_out_names + 4});

        // ---- 6. Run text encoder ----
        std::cout << "Running text encoder ...\n";
        std::vector<Ort::Value> txt_inputs;
        txt_inputs.push_back(create_tensor_int64(mem_info, token_ids, tid_shape));

        const char* txt_in_names[]  = {"token_ids"};
        const char* txt_out_names[] = {"text_features", "text_mask"};
        auto txt_outputs = run_session(txt_sess, {txt_in_names, txt_in_names + 1},
                                       txt_inputs, {txt_out_names, txt_out_names + 2});

        // ---- 7. Run decoder ----
        std::cout << "Running decoder ...\n";
        std::vector<Ort::Value> dec_inputs;
        // Move image encoder outputs (feat_4x, feat_2x, feat_1x, pos_1x)
        for (int i = 0; i < 4; ++i) dec_inputs.push_back(std::move(img_outputs[i]));
        // Move text encoder outputs (text_features, text_mask)
        for (int i = 0; i < 2; ++i) dec_inputs.push_back(std::move(txt_outputs[i]));

        const char* dec_in_names[] = {
            "feat_4x", "feat_2x", "feat_1x", "pos_1x", "text_features", "text_mask"
        };
        const char* dec_out_names[] = {"scores", "boxes_xyxy", "mask_logits"};
        auto dec_outputs = run_session(dec_sess, {dec_in_names, dec_in_names + 6},
                                       dec_inputs, {dec_out_names, dec_out_names + 3});

        // ---- 8. Extract outputs ----
        auto scores_info = dec_outputs[0].GetTensorTypeAndShapeInfo();
        int num_queries = static_cast<int>(scores_info.GetShape()[1]);  // 200

        const float* raw_scores = dec_outputs[0].GetTensorData<float>();     // [1, 200]
        const float* raw_boxes  = dec_outputs[1].GetTensorData<float>();     // [1, 200, 4]
        const float* raw_masks  = dec_outputs[2].GetTensorData<float>();     // [1, 200, H, H]

        auto mask_shape = dec_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        int mask_h = static_cast<int>(mask_shape[2]);
        int mask_w = static_cast<int>(mask_shape[3]);

        std::cout << "Decoder output: " << num_queries << " queries, mask "
                  << mask_h << "x" << mask_w << "\n";

        // ---- 9. Post-processing ----
        // 9a. Score filter
        std::vector<int> valid;
        for (int i = 0; i < num_queries; ++i) {
            if (raw_scores[i] > args.score_threshold) {
                valid.push_back(i);
            }
        }
        std::cout << "After score filter (>" << args.score_threshold << "): "
                  << valid.size() << "\n";

        if (valid.empty()) {
            std::cout << "No detections. Saving original image.\n";
            cv::imwrite(args.output, img_bgr);
            return 0;
        }

        // Gather valid boxes/scores
        std::vector<float> v_scores(valid.size());
        std::vector<float> v_boxes(valid.size() * 4);
        for (size_t i = 0; i < valid.size(); ++i) {
            v_scores[i] = raw_scores[valid[i]];
            std::memcpy(&v_boxes[i * 4], &raw_boxes[valid[i] * 4], 4 * sizeof(float));
        }

        // 9b. NMS
        auto keep = nms(v_boxes.data(), v_scores.data(),
                        static_cast<int>(valid.size()), args.nms_threshold);
        std::cout << "After NMS: " << keep.size() << "\n";

        // 9c. Top-K (keep is already score-sorted from NMS)
        if (static_cast<int>(keep.size()) > args.top_k) {
            keep.resize(args.top_k);
        }
        std::cout << "Top-" << args.top_k << ": " << keep.size() << "\n";

        // 9d. Collect final results
        std::vector<float> final_scores;
        std::vector<std::array<float, 4>> final_boxes;
        std::vector<cv::Mat> final_masks;

        for (int ki : keep) {
            int qi = valid[ki];  // original query index

            final_scores.push_back(raw_scores[qi]);

            // Box: normalized [0,1] → pixel coords
            std::array<float, 4> box = {
                raw_boxes[qi * 4 + 0] * orig_w,
                raw_boxes[qi * 4 + 1] * orig_h,
                raw_boxes[qi * 4 + 2] * orig_w,
                raw_boxes[qi * 4 + 3] * orig_h,
            };
            final_boxes.push_back(box);

            // Mask: sigmoid → resize → threshold
            const float* logits = &raw_masks[qi * mask_h * mask_w];
            cv::Mat mask_prob(mask_h, mask_w, CV_32FC1);
            for (int y = 0; y < mask_h; ++y) {
                for (int x = 0; x < mask_w; ++x) {
                    float v = logits[y * mask_w + x];
                    mask_prob.at<float>(y, x) = 1.0f / (1.0f + std::exp(-v));
                }
            }
            cv::Mat resized_mask;
            cv::resize(mask_prob, resized_mask, cv::Size(orig_w, orig_h));
            cv::Mat binary_mask;
            cv::threshold(resized_mask, binary_mask, 0.5f, 255.0f, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8UC1);
            final_masks.push_back(binary_mask);
        }

        // ---- 10. Print results ----
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "Prompt: '" << args.prompt << "'\n";
        std::cout << std::string(50, '=') << "\n";
        for (size_t i = 0; i < final_scores.size(); ++i) {
            auto& b = final_boxes[i];
            std::printf("  [%zu] score=%.3f  box=(%.0f, %.0f, %.0f, %.0f)\n",
                        i, final_scores[i], b[0], b[1], b[2], b[3]);
        }

        // ---- 11. Visualization ----
        cv::Mat vis = img_bgr.clone();
        draw_results(vis, final_scores, final_boxes, final_masks);
        cv::imwrite(args.output, vis);
        std::cout << "\nSaved visualization to: " << args.output << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
