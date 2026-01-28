# EfficientSAM3 C++ 推理方案：LibTorch + TorchScript

## 背景

EfficientSAM3 的 **Transformer Fusion** 组件无法导出为 ONNX 格式（包含自定义 Cross-Attention、条件分支等复杂逻辑），因此无法使用纯 ONNX 在 C++ 中进行文本概念分割。

**解决方案**：使用 TorchScript 导出模型，用 LibTorch (C++ PyTorch) 加载推理。

---

## 方案概述

```
┌─────────────────────────────────────────────────────────┐
│                      工作流程                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Python 端                         C++ 端               │
│   ─────────                         ──────               │
│                                                          │
│   PyTorch 模型                                           │
│       │                                                  │
│       ▼                                                  │
│   torch.jit.script()                                    │
│       │                                                  │
│       ▼                                                  │
│   保存 .pt 文件  ──────────────►  LibTorch 加载          │
│                                       │                  │
│                                       ▼                  │
│                                   C++ 推理               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 一、TorchScript 导出（Python）

### 1.1 完整模型导出

```python
#!/usr/bin/env python3
"""导出 EfficientSAM3 为 TorchScript 格式"""

import torch
from sam3.model_builder import build_efficientsam3_image_model

# 1. 加载模型
print("加载模型...")
model = build_efficientsam3_image_model(
    checkpoint_path="weights/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    backbone_type="repvit",
    model_name="m0_9",
    text_encoder_type="MobileCLIP-S1"
)
model.eval()

# 2. 尝试 TorchScript 导出
print("导出 TorchScript...")

# 方法1: Tracing（追踪）
# 适用于没有条件分支的模型
try:
    dummy_image = torch.randn(1, 3, 1024, 1024)
    dummy_text_ids = torch.randint(0, 49408, (1, 32))

    with torch.no_grad():
        traced_model = torch.jit.trace(model, (dummy_image, dummy_text_ids))

    traced_model.save("efficientsam3_traced.pt")
    print("✅ Tracing 导出成功: efficientsam3_traced.pt")
except Exception as e:
    print(f"❌ Tracing 失败: {e}")

# 方法2: Scripting（脚本化）
# 支持条件分支和动态逻辑
try:
    scripted_model = torch.jit.script(model)
    scripted_model.save("efficientsam3_scripted.pt")
    print("✅ Scripting 导出成功: efficientsam3_scripted.pt")
except Exception as e:
    print(f"❌ Scripting 失败: {e}")

# 3. 验证导出的模型
print("\n验证导出的模型...")
try:
    loaded_model = torch.jit.load("efficientsam3_scripted.pt")
    loaded_model.eval()

    with torch.no_grad():
        output = loaded_model(dummy_image, dummy_text_ids)

    print(f"✅ 验证成功，输出类型: {type(output)}")
except Exception as e:
    print(f"❌ 验证失败: {e}")
```

### 1.2 分模块导出（备选）

如果完整模型导出失败，可以分模块导出：

```python
#!/usr/bin/env python3
"""分模块导出 EfficientSAM3"""

import torch
from sam3.model_builder import build_efficientsam3_image_model

model = build_efficientsam3_image_model(...)
model.eval()

# 1. 导出 Image Encoder
class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        return self.backbone(image)

img_encoder = ImageEncoderWrapper(model.detector.backbone.vision_backbone)
scripted_img_encoder = torch.jit.script(img_encoder)
scripted_img_encoder.save("image_encoder.pt")

# 2. 导出 Text Encoder
class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_backbone):
        super().__init__()
        self.text_backbone = text_backbone

    def forward(self, input_ids):
        return self.text_backbone(input_ids)

text_encoder = TextEncoderWrapper(model.detector.backbone.text_backbone)
scripted_text_encoder = torch.jit.script(text_encoder)
scripted_text_encoder.save("text_encoder.pt")

# 3. 导出 Detector Backend (Transformer + SegHead)
class DetectorBackend(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.transformer = detector.transformer
        self.segmentation_head = detector.segmentation_head

    def forward(self, image_features, text_features):
        # 融合 + 分割
        fused = self.transformer(image_features, text_features)
        masks = self.segmentation_head(fused)
        return masks

backend = DetectorBackend(model.detector)
scripted_backend = torch.jit.script(backend)
scripted_backend.save("detector_backend.pt")

print("✅ 分模块导出完成")
```

---

## 二、LibTorch 安装

### 2.1 下载预编译版本

```bash
# Linux (CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# Linux (CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# macOS (CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.0.zip
unzip libtorch-macos-arm64-2.1.0.zip
```

### 2.2 目录结构

```
libtorch/
├── bin/
├── include/
│   ├── torch/
│   │   ├── script.h      # TorchScript API
│   │   ├── torch.h       # 主头文件
│   │   └── ...
│   └── ...
├── lib/
│   ├── libtorch.so       # Linux
│   ├── libtorch.dylib    # macOS
│   └── ...
└── share/
    └── cmake/
        └── Torch/
            └── TorchConfig.cmake
```

---

## 三、C++ 推理代码

### 3.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(efficientsam3_inference)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 LibTorch 路径
set(CMAKE_PREFIX_PATH "/path/to/libtorch")

# 查找 LibTorch
find_package(Torch REQUIRED)

# 查找 OpenCV（可选，用于图像处理）
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(sam3_inference main.cpp)

# 链接库
target_link_libraries(sam3_inference
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# 设置编译选项
set_property(TARGET sam3_inference PROPERTY CXX_STANDARD 17)

# 如果使用 CUDA
if(TORCH_CUDA_LIBRARIES)
    target_link_libraries(sam3_inference ${TORCH_CUDA_LIBRARIES})
endif()
```

### 3.2 main.cpp

```cpp
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class EfficientSAM3 {
public:
    EfficientSAM3(const std::string& model_path) {
        try {
            // 加载 TorchScript 模型
            model_ = torch::jit::load(model_path);
            model_.eval();

            // 如果有 GPU，移动到 GPU
            if (torch::cuda::is_available()) {
                model_.to(torch::kCUDA);
                device_ = torch::kCUDA;
                std::cout << "使用 GPU 推理" << std::endl;
            } else {
                device_ = torch::kCPU;
                std::cout << "使用 CPU 推理" << std::endl;
            }
        } catch (const c10::Error& e) {
            std::cerr << "加载模型失败: " << e.what() << std::endl;
            throw;
        }
    }

    // 图像预处理
    torch::Tensor preprocessImage(const cv::Mat& image) {
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        // Resize 到 1024x1024
        cv::Mat resized;
        cv::resize(rgb_image, resized, cv::Size(1024, 1024));

        // 转换为 float 并归一化到 [-1, 1]
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
        resized = (resized - 0.5) / 0.5;

        // 转换为 Tensor [1, 3, 1024, 1024]
        torch::Tensor tensor = torch::from_blob(
            resized.data,
            {1, 1024, 1024, 3},
            torch::kFloat32
        );

        // NHWC -> NCHW
        tensor = tensor.permute({0, 3, 1, 2}).contiguous();

        return tensor.to(device_);
    }

    // 文本 tokenize（简化版，实际需要完整的 tokenizer）
    torch::Tensor tokenizeText(const std::string& text) {
        // 这里需要实现 tokenizer 或使用预计算的 token
        // 简化示例：假设已经有 token IDs
        std::vector<int64_t> token_ids(32, 0);  // 填充到长度 32

        // 示例：手动填充 "a dog" 的 token
        // 49406 = start token
        // 320 = "a"
        // 1929 = "dog"
        // 49407 = end token
        token_ids[0] = 49406;
        token_ids[1] = 320;
        token_ids[2] = 1929;
        token_ids[3] = 49407;

        torch::Tensor tensor = torch::tensor(token_ids, torch::kInt64);
        tensor = tensor.unsqueeze(0);  // [1, 32]

        return tensor.to(device_);
    }

    // 推理
    std::vector<cv::Mat> segment(const cv::Mat& image, const std::string& text_prompt) {
        torch::NoGradGuard no_grad;

        // 1. 预处理图像
        torch::Tensor image_tensor = preprocessImage(image);

        // 2. Tokenize 文本
        torch::Tensor text_tensor = tokenizeText(text_prompt);

        // 3. 推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(image_tensor);
        inputs.push_back(text_tensor);

        auto output = model_.forward(inputs);

        // 4. 处理输出
        torch::Tensor masks_tensor;
        if (output.isTensor()) {
            masks_tensor = output.toTensor();
        } else if (output.isTuple()) {
            auto tuple = output.toTuple();
            masks_tensor = tuple->elements()[0].toTensor();  // 假设第一个是 masks
        }

        // 5. 转换为 OpenCV Mat
        masks_tensor = masks_tensor.to(torch::kCPU).squeeze(0);  // [N, H, W]

        std::vector<cv::Mat> masks;
        for (int i = 0; i < masks_tensor.size(0); i++) {
            torch::Tensor mask = masks_tensor[i];

            // 二值化
            mask = (mask > 0).to(torch::kFloat32);

            // 转换为 cv::Mat
            cv::Mat cv_mask(mask.size(0), mask.size(1), CV_32F, mask.data_ptr<float>());

            // 上采样到原始尺寸
            cv::Mat resized_mask;
            cv::resize(cv_mask, resized_mask, image.size());

            // 转换为 8-bit
            cv::Mat mask_8u;
            resized_mask.convertTo(mask_8u, CV_8U, 255);

            masks.push_back(mask_8u.clone());
        }

        return masks;
    }

private:
    torch::jit::script::Module model_;
    torch::Device device_ = torch::kCPU;
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <model.pt> <image.jpg>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string text_prompt = (argc > 3) ? argv[3] : "a dog";

    try {
        // 1. 加载模型
        std::cout << "加载模型: " << model_path << std::endl;
        EfficientSAM3 sam3(model_path);

        // 2. 读取图像
        std::cout << "读取图像: " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "无法读取图像" << std::endl;
            return 1;
        }

        // 3. 推理
        std::cout << "分割: \"" << text_prompt << "\"" << std::endl;
        auto masks = sam3.segment(image, text_prompt);

        std::cout << "检测到 " << masks.size() << " 个对象" << std::endl;

        // 4. 保存结果
        for (size_t i = 0; i < masks.size(); i++) {
            std::string output_path = "mask_" + std::to_string(i) + ".png";
            cv::imwrite(output_path, masks[i]);
            std::cout << "保存: " << output_path << std::endl;
        }

        // 5. 可视化
        cv::Mat visualization = image.clone();
        for (const auto& mask : masks) {
            // 创建彩色覆盖
            cv::Mat colored_mask;
            cv::cvtColor(mask, colored_mask, cv::COLOR_GRAY2BGR);
            colored_mask.setTo(cv::Scalar(0, 255, 0), mask > 127);  // 绿色

            // 叠加
            cv::addWeighted(visualization, 0.7, colored_mask, 0.3, 0, visualization);
        }

        cv::imwrite("visualization.jpg", visualization);
        std::cout << "保存可视化: visualization.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### 3.3 编译和运行

```bash
# 创建构建目录
mkdir build && cd build

# 配置（设置 LibTorch 路径）
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# 编译
make -j4

# 运行
./sam3_inference efficientsam3_scripted.pt input.jpg "a dog"
```

---

## 四、Tokenizer 实现

C++ 中需要实现文本 Tokenizer。可以选择：

### 4.1 方案 A：预计算 Token

```python
# Python 预计算常用文本的 token
from sam3.model.tokenizer_ve import SimpleTokenizer

tokenizer = SimpleTokenizer()

common_prompts = {
    "a dog": tokenizer("a dog", context_length=32).tolist(),
    "a cat": tokenizer("a cat", context_length=32).tolist(),
    "a person": tokenizer("a person", context_length=32).tolist(),
    # ...
}

# 保存为 JSON
import json
with open("token_map.json", "w") as f:
    json.dump(common_prompts, f)
```

### 4.2 方案 B：C++ Tokenizer 库

使用 [SentencePiece](https://github.com/google/sentencepiece) 或 [tokenizers-cpp](https://github.com/huggingface/tokenizers)：

```cpp
#include <sentencepiece_processor.h>

class CLIPTokenizer {
public:
    CLIPTokenizer(const std::string& model_path) {
        processor_.Load(model_path);
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        processor_.Encode(text, &ids);
        return ids;
    }

private:
    sentencepiece::SentencePieceProcessor processor_;
};
```

---

## 五、注意事项

### 5.1 TorchScript 导出可能的问题

1. **不支持的操作**：某些 Python 特性可能不支持
2. **动态形状**：需要固定输入尺寸或使用 `torch.jit.trace`
3. **自定义算子**：可能需要注册

### 5.2 性能优化

```cpp
// 1. 使用 GPU
if (torch::cuda::is_available()) {
    model.to(torch::kCUDA);
    input = input.to(torch::kCUDA);
}

// 2. 半精度推理（GPU）
model.to(torch::kHalf);
input = input.to(torch::kHalf);

// 3. 禁用梯度
torch::NoGradGuard no_grad;

// 4. 批处理
// 一次处理多张图像
```

### 5.3 内存管理

```cpp
// LibTorch 使用智能指针，自动管理内存
// 但大 Tensor 建议手动释放

{
    torch::Tensor large_tensor = torch::randn({1000, 1000, 1000});
    // 使用 large_tensor
} // 离开作用域自动释放

// 或手动清空
large_tensor.reset();
```

---

## 六、与 ONNX 方案对比

| 特性 | LibTorch + TorchScript | ONNX |
|-----|----------------------|------|
| **Transformer Fusion** | ✅ 可能支持 | ❌ 无法导出 |
| **条件分支** | ✅ 支持 | ❌ 有限 |
| **自定义模块** | ✅ 支持 | ❌ 困难 |
| **库大小** | ~1GB | ~50MB |
| **跨框架** | ❌ 仅 PyTorch | ✅ 多框架 |
| **部署难度** | 中等 | 简单 |

---

## 七、总结

### 推荐流程

```
1. Python 导出 TorchScript
   torch.jit.script(model).save("model.pt")

2. 下载 LibTorch
   wget https://download.pytorch.org/libtorch/...

3. C++ 编写推理代码
   torch::jit::load("model.pt")

4. CMake 编译
   cmake && make

5. 运行推理
   ./sam3_inference model.pt image.jpg "a dog"
```

### 关键优势

- **支持复杂模型**：Transformer Fusion 等自定义层可以正常工作
- **无需 Python**：纯 C++ 运行时
- **性能接近原生**：与 PyTorch 相同的底层实现

### 潜在风险

- **导出可能失败**：需要测试 TorchScript 是否能成功导出
- **库较大**：LibTorch ~1GB
- **调试困难**：C++ 错误信息不如 Python 友好

---

## 八、下一步

1. **测试 TorchScript 导出**：验证模型是否能成功导出
2. **实现 Tokenizer**：C++ 文本处理
3. **性能调优**：GPU 加速、批处理等
4. **集成到项目**：与现有 C++ 代码整合
