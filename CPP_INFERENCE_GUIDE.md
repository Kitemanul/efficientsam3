# EfficientSAM3 C++ ONNX 推理指南

## 目录

- [1. 概述](#1-概述)
- [2. 整体架构](#2-整体架构)
- [3. 文件结构](#3-文件结构)
- [4. 依赖库](#4-依赖库)
- [5. 构建方法](#5-构建方法)
- [6. 使用方法](#6-使用方法)
- [7. ONNX 模型 I/O 规格](#7-onnx-模型-io-规格)
- [8. 模块详解：SimpleTokenizer](#8-模块详解simpletokenizer)
- [9. 模块详解：图像预处理](#9-模块详解图像预处理)
- [10. 模块详解：ONNX Runtime 推理](#10-模块详解onnx-runtime-推理)
- [11. 模块详解：后处理](#11-模块详解后处理)
- [12. 模块详解：可视化](#12-模块详解可视化)
- [13. C++ 与 Python 逐步对比](#13-c-与-python-逐步对比)
- [14. TV 平台交叉编译](#14-tv-平台交叉编译)
- [15. 常见问题](#15-常见问题)

---

## 1. 概述

本项目将 Python 版 `run_onnx_text_grounding.py` 完整改写为 C++，用于在 **WebOS / Tizen TV** 平台上运行 EfficientSAM3 的文本引导分割推理。

**完整推理管线**：

```
输入图片 + 文本 prompt
    ↓
[图像预处理] resize + normalize → float32 [1,3,1008,1008]
[文本 tokenize] BPE → int64 [1,77]
    ↓
[image_encoder.onnx] → feat_4x, feat_2x, feat_1x, pos_1x
[text_encoder.onnx]  → text_features, text_mask
    ↓
[decoder.onnx] → scores [1,200], boxes [1,200,4], masks [1,200,288,288]
    ↓
[后处理] score 过滤 → NMS → top-k → sigmoid → resize
    ↓
[可视化] mask overlay + contour + box + score label → 保存图片
```

**核心特点**：
- 纯 C++17 实现，无 Python 依赖
- SimpleTokenizer (CLIP BPE) 完整 C++ 移植
- ONNX Runtime C++ API 推理
- OpenCV 图像处理与可视化
- 与 Python 版本数值结果一致

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      main.cpp                           │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │ 命令行    │  │ 图像预处理 │  │ SimpleTokenizer    │    │
│  │ 参数解析  │  │ OpenCV   │  │ (BPE tokenizer)    │    │
│  └────┬─────┘  └────┬─────┘  └────────┬───────────┘    │
│       │              │                 │                 │
│       ▼              ▼                 ▼                 │
│  ┌────────────────────────────────────────────────┐     │
│  │           ONNX Runtime 推理引擎                  │     │
│  │                                                │     │
│  │  image_encoder → text_encoder → decoder        │     │
│  └────────────────────┬───────────────────────────┘     │
│                       │                                  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────┐     │
│  │           后处理 + 可视化                        │     │
│  │                                                │     │
│  │  score filter → NMS → top-k → sigmoid/resize   │     │
│  │  → draw masks/boxes/scores → imwrite           │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 文件结构

```
cpp_inference/
├── CMakeLists.txt              # CMake 构建配置
├── simple_tokenizer.h          # SimpleTokenizer 类声明 (104 行)
├── simple_tokenizer.cpp        # SimpleTokenizer 实现 (457 行)
└── main.cpp                    # 主程序入口 (422 行)
```

| 文件 | 行数 | 说明 |
|------|------|------|
| `CMakeLists.txt` | 33 | CMake 构建配置，链接 OpenCV + zlib + ONNX Runtime |
| `simple_tokenizer.h` | 104 | BPE tokenizer 类声明，包含 PairHash 辅助结构 |
| `simple_tokenizer.cpp` | 457 | BPE tokenizer 完整实现：vocab 加载、BPE 合并、编码 |
| `main.cpp` | 422 | 端到端推理管线：预处理、3 模型推理、后处理、可视化 |

对应 Python 文件：

| C++ 文件 | 对应 Python |
|----------|-------------|
| `simple_tokenizer.h/cpp` | `sam3/sam3/model/tokenizer_ve.py` 中的 `SimpleTokenizer` 类 |
| `main.cpp` | `run_onnx_text_grounding.py` |

---

## 4. 依赖库

| 库 | 最低版本 | 用途 |
|----|----------|------|
| **C++17** | GCC 7+ / Clang 5+ | structured bindings, `<string_view>` 等 |
| **OpenCV** | 4.0+ | 图像读取、resize、归一化、可视化 (contour/rect/text) |
| **zlib** | 1.2+ | 读取 gzip 压缩的 BPE 词表文件 |
| **ONNX Runtime** | 1.14+ | C++ API 模型推理 |
| **CMake** | 3.14+ | 构建系统 |

### 依赖安装 (macOS)

```bash
brew install opencv zlib cmake

# ONNX Runtime: 从 GitHub Release 下载预编译包
# https://github.com/microsoft/onnxruntime/releases
# 下载 onnxruntime-osx-arm64-<version>.tgz 并解压
```

### 依赖安装 (Ubuntu)

```bash
sudo apt install libopencv-dev zlib1g-dev cmake

# ONNX Runtime: 下载 onnxruntime-linux-x64-<version>.tgz
```

---

## 5. 构建方法

### 5.1 桌面端构建

```bash
cd cpp_inference
mkdir build && cd build

# 配置 (指定 ONNX Runtime 路径)
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime

# 编译
make -j$(nproc)
```

### 5.2 CMakeLists.txt 解析

```cmake
cmake_minimum_required(VERSION 3.14)
project(sam3_onnx_inference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)              # 必须 C++17 (structured bindings)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)            # 自动查找系统 OpenCV
find_package(ZLIB REQUIRED)              # 自动查找系统 zlib

# ONNX Runtime 无官方 CMake FindModule，需手动指定路径
if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR")
endif()

add_executable(sam3_infer
    main.cpp                              # 主程序
    simple_tokenizer.cpp                  # BPE tokenizer
)

target_include_directories(sam3_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}               # opencv2/opencv.hpp
    ${ONNXRUNTIME_DIR}/include           # onnxruntime_cxx_api.h
)

target_link_libraries(sam3_infer PRIVATE
    ${OpenCV_LIBS}                       # libopencv_core, libopencv_imgproc, ...
    ZLIB::ZLIB                           # libz
    # 动态链接 ONNX Runtime (根据平台自动选择 .so / .dylib)
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}
)
```

**构建输出**: 可执行文件 `sam3_infer`

---

## 6. 使用方法

### 6.1 基本用法

```bash
./sam3_infer \
    --image test.jpg \
    --prompt "person" \
    --onnx-dir exports_repvit_m0_9/ \
    --bpe-path sam3/assets/bpe_simple_vocab_16e6.txt.gz \
    --output result.png
```

### 6.2 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image` | string | **必填** | 输入图片路径 |
| `--prompt` | string | **必填** | 文本 prompt (如 `"person"`, `"cat"`) |
| `--onnx-dir` | string | `exports_repvit_m0_9` | 包含 3 个 ONNX 模型的目录 |
| `--output` | string | `result.png` | 输出可视化图片路径 |
| `--bpe-path` | string | `sam3/assets/bpe_simple_vocab_16e6.txt.gz` | BPE 词表文件路径 |
| `--resolution` | int | `1008` | 输入分辨率 (须与导出时一致) |
| `--score-threshold` | float | `0.1` | 置信度过滤阈值 |
| `--nms-threshold` | float | `0.5` | NMS IoU 阈值 |
| `--top-k` | int | `10` | 最大检测数量 |

### 6.3 输出示例

```
Loading tokenizer from: sam3/assets/bpe_simple_vocab_16e6.txt.gz
Loading ONNX models from: exports_repvit_m0_9
  Models loaded.
Image: test.jpg (640x480)
Prompt: 'person' -> 3 tokens
Running image encoder ...
Running text encoder ...
Running decoder ...
Decoder output: 200 queries, mask 288x288
After score filter (>0.1): 5
After NMS: 3
Top-10: 3
==================================================
Prompt: 'person'
==================================================
  [0] score=0.180  box=(120, 50, 380, 450)
  [1] score=0.155  box=(400, 80, 580, 440)
  [2] score=0.142  box=(50, 100, 200, 420)

Saved visualization to: result.png
```

---

## 7. ONNX 模型 I/O 规格

推理管线使用 3 个 ONNX 模型，数据在模型间串联传递。

### 7.1 image_encoder.onnx

图像特征提取器 (RepViT-m0.9 backbone + PixelDecoder)。

| 方向 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `images` | `[1, 3, 1008, 1008]` | float32 | 归一化到 [-1, 1] 的 RGB 图像 |
| 输出 | `feat_4x` | `[1, 256, 252, 252]` | float32 | 4× 下采样特征图 |
| 输出 | `feat_2x` | `[1, 256, 504, 504]` | float32 | 2× 下采样特征图 |
| 输出 | `feat_1x` | `[1, 256, 72, 72]` | float32 | 编码器输出特征 (stride 14) |
| 输出 | `pos_1x` | `[1, 256, 72, 72]` | float32 | 位置编码 |

> 文件大小: ~1.3 MB

### 7.2 text_encoder.onnx

文本编码器 (MobileCLIP-S1 text encoder)。

| 方向 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `token_ids` | `[1, 77]` | int64 | BPE token IDs (0-padded) |
| 输出 | `text_features` | `[1, 77, 256]` | float32 | 每个 token 的特征向量 |
| 输出 | `text_mask` | `[1, 77]` | bool | 有效 token 掩码 (非 padding 位置为 true) |

> 文件大小: ~0.5 MB

### 7.3 decoder.onnx

Transformer 检测 + 分割解码器。

| 方向 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `feat_4x` | `[1, 256, 252, 252]` | float32 | 来自 image_encoder |
| 输入 | `feat_2x` | `[1, 256, 504, 504]` | float32 | 来自 image_encoder |
| 输入 | `feat_1x` | `[1, 256, 72, 72]` | float32 | 来自 image_encoder |
| 输入 | `pos_1x` | `[1, 256, 72, 72]` | float32 | 来自 image_encoder |
| 输入 | `text_features` | `[1, 77, 256]` | float32 | 来自 text_encoder |
| 输入 | `text_mask` | `[1, 77]` | bool | 来自 text_encoder |
| 输出 | `scores` | `[1, 200]` | float32 | 200 个查询的置信度分数 |
| 输出 | `boxes_xyxy` | `[1, 200, 4]` | float32 | 归一化边界框 [x1,y1,x2,y2] ∈ [0,1] |
| 输出 | `mask_logits` | `[1, 200, 288, 288]` | float32 | 分割掩码 logits (未 sigmoid) |

> 文件大小: ~111.3 MB

### 7.4 数据流图

```
image [1,3,1008,1008]           token_ids [1,77]
         │                              │
         ▼                              ▼
  ┌──────────────┐               ┌─────────────┐
  │ image_encoder│               │text_encoder  │
  └──────┬───────┘               └──────┬──────┘
         │                              │
    feat_4x [1,256,252,252]     text_features [1,77,256]
    feat_2x [1,256,504,504]     text_mask [1,77]
    feat_1x [1,256,72,72]              │
    pos_1x  [1,256,72,72]             │
         │                              │
         └──────────┬───────────────────┘
                    ▼
             ┌───────────┐
             │  decoder   │
             └─────┬─────┘
                   │
          scores [1,200]
          boxes  [1,200,4]
          masks  [1,200,288,288]
```

---

## 8. 模块详解：SimpleTokenizer

### 8.1 概述

SimpleTokenizer 是 CLIP 的 BPE (Byte Pair Encoding) 分词器的 C++ 完整移植，对应 Python 中的 `sam3/sam3/model/tokenizer_ve.py::SimpleTokenizer`。

它将任意英文文本转换为 MobileCLIP-S1 文本编码器所需的 token ID 序列。

### 8.2 词表结构

BPE 词表文件 `bpe_simple_vocab_16e6.txt.gz` 是 gzip 压缩的文本文件，包含 48894 条 merge 规则。

完整词表共 **49408** 个 token：

| ID 范围 | 数量 | 内容 |
|---------|------|------|
| 0 ~ 255 | 256 | 基础 byte-unicode 字符 |
| 256 ~ 511 | 256 | 词尾变体 (base + `</w>`) |
| 512 ~ 49405 | 48894 | BPE merge 产生的子词 |
| 49406 | 1 | `<start_of_text>` (SOT) |
| 49407 | 1 | `<end_of_text>` (EOT) |

### 8.3 类接口

```cpp
class SimpleTokenizer {
public:
    // 构造: 加载 gzip BPE 词表
    explicit SimpleTokenizer(const std::string& bpe_path, int context_length = 77);

    // 编码单条文本 → token IDs (不含 SOT/EOT/padding)
    std::vector<int64_t> encode(const std::string& text) const;

    // 批量编码 → [batch, context_length] (含 SOT/EOT + zero-padding)
    std::vector<std::vector<int64_t>> tokenize(
        const std::vector<std::string>& texts) const;

    // 查询接口
    int vocab_size()     const;  // 49408
    int sot_token_id()   const;  // 49406
    int eot_token_id()   const;  // 49407
    int context_length() const;  // 77
};
```

### 8.4 内部数据结构

```cpp
// byte (0~255) → UTF-8 编码的 unicode 字符串
std::unordered_map<uint8_t, std::string> byte_encoder_;    // 256 项
std::unordered_map<std::string, uint8_t> byte_decoder_;    // 反向映射

// BPE merge 优先级: (tokenA, tokenB) → rank (越小越优先)
std::unordered_map<std::pair<std::string,std::string>, int, PairHash> bpe_ranks_;

// 词表编码/解码
std::unordered_map<std::string, int> encoder_;   // "hello</w>" → 3456
std::unordered_map<int, std::string> decoder_;   // 3456 → "hello</w>"

// BPE 结果缓存 (避免重复计算)
mutable std::unordered_map<std::string, std::string> cache_;
```

**PairHash** 自定义哈希结构：

`std::pair<std::string, std::string>` 默认不支持 `std::hash`，需要自定义：

```cpp
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 32) ^ (h2 >> 32);  // 组合两个哈希值
    }
};
```

### 8.5 构造函数详解

构造函数执行 8 个步骤初始化 tokenizer：

```
SimpleTokenizer(bpe_path, context_length=77)
    │
    ├── 1. build_bytes_to_unicode()     → byte_encoder_ (256项)
    │                                     byte_decoder_ (反向)
    ├── 2. read_gzip_file(bpe_path)     → 原始文本内容
    ├── 3. split by '\n', 取 [1:48895]  → 48894 条 merges
    ├── 4. 构建 vocab (4 阶段):
    │      Phase 1: 256 base chars       [0..255]
    │      Phase 2: 256 word-end chars   [256..511]
    │      Phase 3: 48894 merge tokens   [512..49405]
    │      Phase 4: SOT + EOT            [49406..49407]
    ├── 5. 构建 encoder_/decoder_
    ├── 6. 构建 bpe_ranks_
    ├── 7. 设置 sot_token_id_=49406, eot_token_id_=49407
    └── 8. 缓存预设 <start_of_text>, <end_of_text>
```

### 8.6 bytes_to_unicode 映射

这是 CLIP tokenizer 的核心设计：将每个 byte (0~255) 映射到一个 Unicode 字符，以避免不可打印字符的问题。

**映射规则**：
- **188 个直接映射**：byte 值本身就是合理的 Unicode codepoint
  - 33~126 (`!` 到 `~`)：94 个 ASCII 可打印字符
  - 161~172 (`¡` 到 `¬`)：12 个 Latin-1 Supplement 字符
  - 174~255 (`®` 到 `ÿ`)：82 个 Latin-1 Supplement 字符
- **68 个替代映射**：不可打印/特殊 byte 映射到 codepoint 256~323
  - 0~32 (控制字符 + 空格)：33 个
  - 127~160 (DEL + 高位控制字符)：34 个
  - 173 (soft hyphen)：1 个

**C++ 实现** (`simple_tokenizer.cpp:68-97`)：

```cpp
std::unordered_map<uint8_t, std::string> SimpleTokenizer::build_bytes_to_unicode() {
    // 收集直接映射的 byte 值
    std::vector<uint8_t> bs;
    for (int i = 33; i <= 126; ++i)  bs.push_back(i);   // 94 个
    for (int i = 161; i <= 172; ++i) bs.push_back(i);   // 12 个
    for (int i = 174; i <= 255; ++i) bs.push_back(i);   // 82 个
    // 共 188 个

    std::vector<uint32_t> cs(bs.begin(), bs.end());  // 自身映射

    // 剩余 68 个 byte → codepoint 256+
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (/* b 不在 bs 中 */) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    // 将 codepoint 转为 UTF-8 字符串
    std::unordered_map<uint8_t, std::string> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result[bs[i]] = codepoint_to_utf8(cs[i]);
    }
    return result;
}
```

辅助函数 `codepoint_to_utf8()` 将 Unicode codepoint 编码为 UTF-8 字节序列：

| Codepoint 范围 | UTF-8 字节数 | 编码格式 |
|----------------|-------------|----------|
| 0x00 ~ 0x7F | 1 | `0xxxxxxx` |
| 0x80 ~ 0x7FF | 2 | `110xxxxx 10xxxxxx` |
| 0x800 ~ 0xFFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |
| 0x10000 ~ 0x10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |

### 8.7 BPE 合并算法

BPE 算法将一个词逐步合并相邻 token，直到没有更多可合并的 pair。

**输入**: byte_encoder 编码后的 UTF-8 字符串 (如 `"hello"` → 各字符的 unicode 编码)

**算法步骤** (`simple_tokenizer.cpp:188-289`)：

```
bpe("hello 的 byte_encoded 形式"):
    1. 按 UTF-8 字符拆分: ["h", "e", "l", "l", "o"]
    2. 末尾加 </w>: ["h", "e", "l", "l", "o</w>"]
    3. 提取相邻 pairs: {(h,e), (e,l), (l,l), (l,o</w>)}
    4. while True:
         a. 在 bpe_ranks_ 中查找 rank 最小的 pair
         b. 如果没有可合并的 pair → break
         c. 合并所有该 pair 的出现
         d. 如果只剩 1 个 token → break
         e. 重新提取 pairs
    5. 返回空格分隔的结果: "hel lo</w>"
```

**关键细节**：
- UTF-8 字符拆分：不能按 byte 拆，要按 UTF-8 字符边界拆 (首字节高位判断长度)
- 缓存机制：`cache_` 存储已计算的 BPE 结果，同一词只计算一次
- `cache_` 声明为 `mutable`，允许在 `const` 方法 `bpe()` 中修改

### 8.8 文本清理

`clean_text()` 对输入文本做预处理 (`simple_tokenizer.cpp:296-320`)：

| 操作 | Python 等价 | C++ 实现 |
|------|------------|----------|
| 转小写 | `text.lower()` | `std::tolower()` |
| 折叠空白 | `re.sub(r'\s+', ' ', text)` | 遍历 + `prev_space` 标志 |
| 去首尾空白 | `text.strip()` | 跳过前导空白 + pop 尾部空白 |

> **简化**: Python 版还调用 `ftfy.fix_text()` 和 `html.unescape()`。C++ 版省略了这些，因为 TV 场景的 prompt 是纯 ASCII 英文。

### 8.9 正则分词

Python 使用正则表达式拆分文本：

```python
re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")
```

C++ 标准正则不支持 `\p{L}` (Unicode Letter 类别)，因此用手写状态机替代 (`simple_tokenizer.cpp:326-389`)：

```
regex_tokenize(text):
    for each character c in text:
        1. 空白 → 跳过
        2. 单引号 + 后续字符 → 检查 contractions:
           优先匹配 3 字符: 're, 've, 'll
           其次匹配 2 字符: 's, 't, 'm, 'd
        3. 字母 → 连续字母序列作为一个 token
        4. 数字 → 单个数字作为一个 token
        5. 其他 → 连续非空白/非字母/非数字字符作为一个 token
```

**匹配优先级与 Python 正则一致**：
- Contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`) 最先匹配
- 连续字母 (`[\p{L}]+`) → `isalpha()` 判断
- 单个数字 (`[\p{N}]`) → `isdigit()` 判断
- 其他符号序列 (`[^\s\p{L}\p{N}]+`)

> **局限性**: `isalpha()` 只覆盖 ASCII 字母。对于非 ASCII 文本 (中文、日文等) 需要引入 ICU 库。TV prompt 场景基本只用英文，此简化足够。

### 8.10 encode 编码流程

`encode()` 是 tokenizer 的核心编码方法 (`simple_tokenizer.cpp:395-423`)：

```
encode("a photo of a dog"):
    1. clean_text() → "a photo of a dog"
    2. regex_tokenize() → ["a", "photo", "of", "a", "dog"]
    3. 对每个 token:
       a. byte_encoder 映射: 每个 byte → unicode 字符串
          "photo" → "photo" (ASCII 直接映射)
       b. bpe() 合并: "photo" → "photo</w>"
       c. split by ' ': ["photo</w>"]
       d. encoder_ 查表: "photo</w>" → 1560
    4. 拼接所有 sub-token IDs: [320, 1560, 539, 320, 1929]
```

### 8.11 tokenize 批量编码

`tokenize()` 在 `encode()` 基础上添加特殊 token 和 padding (`simple_tokenizer.cpp:429-457`)：

```
tokenize(["a dog"]):
    1. encode("a dog") → [320, 1929]
    2. 添加 SOT/EOT: [49406, 320, 1929, 49407]
    3. 截断到 context_length (77) + 保留 EOT
    4. 后面补 0 到 77: [49406, 320, 1929, 49407, 0, 0, ..., 0]
                        \___________ 77 个 ___________/
```

---

## 9. 模块详解：图像预处理

### 9.1 预处理流程

```
原始图片 (任意尺寸, BGR)
    │
    ├── resize → (1008, 1008)           cv::resize()
    ├── BGR → RGB                       cv::cvtColor()
    ├── uint8 → float32, /255          convertTo(CV_32FC3, 1/255)
    ├── normalize: (x-0.5)/0.5 → [-1,1]
    └── HWC → CHW                      手动循环转置

    → float32 [1, 3, 1008, 1008]
```

### 9.2 C++ 实现 (`main.cpp:78-102`)

```cpp
static std::vector<float> preprocess_image(const cv::Mat& bgr, int resolution) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resolution, resolution));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);  // OpenCV 默认 BGR

    cv::Mat fp32;
    rgb.convertTo(fp32, CV_32FC3, 1.0 / 255.0);     // [0, 1]
    fp32 = (fp32 - 0.5f) * 2.0f;                     // [-1, 1]

    // HWC → CHW (ONNX 模型需要 NCHW 格式)
    const int H = resolution, W = resolution;
    std::vector<float> chw(3 * H * W);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                chw[c*H*W + y*W + x] = fp32.at<cv::Vec3f>(y, x)[c];

    return chw;
}
```

### 9.3 与 Python 对比

| 步骤 | Python | C++ |
|------|--------|-----|
| 读取图片 | `PIL.Image.open().convert("RGB")` | `cv::imread()` (BGR) |
| resize | `image.resize((1008,1008), BILINEAR)` | `cv::resize()` |
| 色彩空间 | PIL 原生 RGB | `cv::cvtColor(BGR2RGB)` |
| 归一化 | `(arr - 0.5) / 0.5` | `(fp32 - 0.5f) * 2.0f` |
| 布局转换 | `arr.transpose(2,0,1)` | 手动三重循环 |
| 加 batch 维 | `arr[np.newaxis]` | shape 设为 `{1,3,H,W}` |

---

## 10. 模块详解：ONNX Runtime 推理

### 10.1 初始化

```cpp
// 创建运行环境
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam3");
Ort::SessionOptions opts;
// opts.SetIntraOpNumThreads(4);  // TV 平台可调整

// 加载 3 个模型
Ort::Session img_sess(env, "image_encoder.onnx", opts);
Ort::Session txt_sess(env, "text_encoder.onnx", opts);
Ort::Session dec_sess(env, "decoder.onnx", opts);
```

### 10.2 Tensor 创建辅助函数

```cpp
// float 类型 tensor
static Ort::Value create_tensor(Ort::MemoryInfo& mem_info,
                                std::vector<float>& data,
                                const std::vector<int64_t>& shape);

// int64 类型 tensor (用于 token_ids)
static Ort::Value create_tensor_int64(Ort::MemoryInfo& mem_info,
                                      std::vector<int64_t>& data,
                                      const std::vector<int64_t>& shape);
```

这些辅助函数封装了 `Ort::Value::CreateTensor<T>()` 调用，使用 CPU 内存分配器 (`OrtArenaAllocator`)。

> **重要**: `data` 参数传引用而非 const 引用，因为 ONNX Runtime 的 `CreateTensor` 不拷贝数据，而是直接引用原始内存。因此 **`data` 的生命周期必须覆盖整个 session.Run() 调用**。

### 10.3 推理执行辅助函数

```cpp
static std::vector<Ort::Value> run_session(
    Ort::Session& session,
    const std::vector<const char*>& input_names,    // 输入 tensor 名称
    std::vector<Ort::Value>& input_tensors,          // 输入 tensor 值
    const std::vector<const char*>& output_names);   // 输出 tensor 名称
```

### 10.4 三阶段推理

**阶段 1: image_encoder** (`main.cpp:277-287`)

```cpp
const char* img_in_names[]  = {"images"};
const char* img_out_names[] = {"feat_4x", "feat_2x", "feat_1x", "pos_1x"};

std::vector<Ort::Value> img_inputs;
img_inputs.push_back(create_tensor(mem_info, img_data, {1,3,1008,1008}));

auto img_outputs = run_session(img_sess,
    {img_in_names, img_in_names + 1},       // 1 个输入
    img_inputs,
    {img_out_names, img_out_names + 4});     // 4 个输出
```

**阶段 2: text_encoder** (`main.cpp:290-297`)

```cpp
const char* txt_in_names[]  = {"token_ids"};
const char* txt_out_names[] = {"text_features", "text_mask"};

std::vector<Ort::Value> txt_inputs;
txt_inputs.push_back(create_tensor_int64(mem_info, token_ids, {1, 77}));

auto txt_outputs = run_session(txt_sess, ...);  // 1 输入, 2 输出
```

**阶段 3: decoder** (`main.cpp:300-312`)

```cpp
// 将前两个阶段的输出直接作为 decoder 的输入 (move 语义, 零拷贝)
std::vector<Ort::Value> dec_inputs;
for (int i = 0; i < 4; ++i) dec_inputs.push_back(std::move(img_outputs[i]));
for (int i = 0; i < 2; ++i) dec_inputs.push_back(std::move(txt_outputs[i]));

const char* dec_in_names[] = {
    "feat_4x", "feat_2x", "feat_1x", "pos_1x",
    "text_features", "text_mask"
};
const char* dec_out_names[] = {"scores", "boxes_xyxy", "mask_logits"};

auto dec_outputs = run_session(dec_sess, ...);  // 6 输入, 3 输出
```

> **性能提示**: 使用 `std::move` 将 image_encoder/text_encoder 的输出 tensor 直接传给 decoder，避免数据拷贝。

### 10.5 输出提取

```cpp
// 获取原始数据指针 (不拷贝)
const float* raw_scores = dec_outputs[0].GetTensorData<float>();  // [1, 200]
const float* raw_boxes  = dec_outputs[1].GetTensorData<float>();  // [1, 200, 4]
const float* raw_masks  = dec_outputs[2].GetTensorData<float>();  // [1, 200, 288, 288]

// 获取 mask 尺寸
auto mask_shape = dec_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
int mask_h = mask_shape[2];  // 288
int mask_w = mask_shape[3];  // 288
```

---

## 11. 模块详解：后处理

后处理将 decoder 的原始输出转换为最终检测结果。

### 11.1 后处理流程

```
decoder 输出: scores[200], boxes[200,4], masks[200,288,288]
    │
    ├── (a) Score 过滤: scores > threshold → valid 子集
    ├── (b) NMS: IoU 抑制重叠框 → keep 子集
    ├── (c) Top-K: 保留分数最高的 K 个
    ├── (d) Box 坐标: [0,1] 归一化 → 像素坐标
    └── (e) Mask 处理: sigmoid → resize → threshold(0.5)

    → final_scores, final_boxes, final_masks
```

### 11.2 Score 过滤 (`main.cpp:331-344`)

```cpp
std::vector<int> valid;
for (int i = 0; i < num_queries; ++i) {
    if (raw_scores[i] > args.score_threshold) {
        valid.push_back(i);  // 记录通过阈值的 query 索引
    }
}
```

默认阈值 0.1。stage1 checkpoint 的分数较低 (~0.14-0.18)，需要低阈值。

### 11.3 NMS 实现 (`main.cpp:140-175`)

```cpp
static std::vector<int> nms(const float* boxes, const float* scores,
                            int n, float iou_threshold) {
    // 1. 按 score 降序排列索引
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    // 2. 贪心抑制
    std::vector<bool> suppressed(n, false);
    std::vector<int> keep;
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
    return keep;  // 已按 score 降序排列
}
```

**IoU 计算**：

```
IoU(A, B) = intersection(A, B) / union(A, B)
           = (overlap_area) / (area_A + area_B - overlap_area + ε)
```

其中 `ε = 1e-8` 防止除零。

### 11.4 Top-K (`main.cpp:359-362`)

```cpp
// NMS 返回的 keep 已按 score 降序排列
if (keep.size() > args.top_k) {
    keep.resize(args.top_k);  // 直接截断
}
```

### 11.5 Box 坐标转换 (`main.cpp:376-381`)

```cpp
// 归一化 [0,1] → 像素坐标
std::array<float, 4> box = {
    raw_boxes[qi * 4 + 0] * orig_w,   // x1
    raw_boxes[qi * 4 + 1] * orig_h,   // y1
    raw_boxes[qi * 4 + 2] * orig_w,   // x2
    raw_boxes[qi * 4 + 3] * orig_h,   // y2
};
```

### 11.6 Mask 处理 (`main.cpp:384-398`)

```
mask_logits [288×288]
    │
    ├── sigmoid: 1/(1+exp(-x)) → [0, 1] 概率
    ├── resize: 双线性插值到原图尺寸 (orig_w × orig_h)
    └── threshold: > 0.5 → 二值 mask (0 或 255, uint8)
```

```cpp
// Sigmoid
for (int y = 0; y < mask_h; ++y)
    for (int x = 0; x < mask_w; ++x)
        mask_prob.at<float>(y,x) = 1.0f / (1.0f + std::exp(-logits[y*mask_w+x]));

// Resize + threshold
cv::resize(mask_prob, resized_mask, cv::Size(orig_w, orig_h));
cv::threshold(resized_mask, binary_mask, 0.5f, 255.0f, cv::THRESH_BINARY);
binary_mask.convertTo(binary_mask, CV_8UC1);
```

---

## 12. 模块详解：可视化

### 12.1 调色板

20 种颜色循环使用 (BGR 格式)：

```cpp
static const cv::Scalar PALETTE[] = {
    {56, 56, 255},   // 红
    {151, 157, 255}, // 浅红
    {31, 112, 255},  // 橙
    // ... 共 20 种
};
```

### 12.2 绘制流程

`draw_results()` 对每个检测结果绘制 4 层 (`main.cpp:190-225`)：

```
对每个检测 i (按顺序绘制):
    │
    ├── 1. Mask overlay: 半透明彩色覆盖
    │      cv::addWeighted(原图, 0.5, 纯色图, 0.5, 0, 混合图)
    │      混合图.copyTo(画布, mask)  ← 只在 mask 区域生效
    │
    ├── 2. Mask contour: 轮廓线
    │      cv::findContours(mask) → cv::drawContours(color, 线宽=2)
    │
    ├── 3. Box: 矩形框
    │      cv::rectangle(tl, br, color, 线宽=2)
    │
    └── 4. Score label: 分数标签
           "0.18" 白字 + 彩色背景矩形
           cv::putText(FONT_HERSHEY_SIMPLEX, 0.6)
```

---

## 13. C++ 与 Python 逐步对比

### 13.1 main 函数执行流程对比

| 步骤 | Python (`run_onnx_text_grounding.py`) | C++ (`main.cpp`) |
|------|---------------------------------------|-------------------|
| 参数解析 | `argparse.ArgumentParser` | 手写 `parse_args()` + `Args` 结构体 |
| 加载 tokenizer | `SimpleTokenizer(bpe_path=...)` | `SimpleTokenizer tokenizer(bpe_path)` |
| 加载模型 | `ort.InferenceSession(path)` | `Ort::Session(env, path, opts)` |
| 读取图片 | `PIL.Image.open().convert("RGB")` | `cv::imread()` (BGR) |
| 预处理 | NumPy 数组运算 | OpenCV + 手动 CHW 转换 |
| Tokenize | `tokenizer([text])` | `tokenizer.tokenize({text})` |
| image_encoder | `img_sess.run(None, {"images": ...})` | `run_session(img_sess, ...)` |
| text_encoder | `txt_sess.run(None, {"token_ids": ...})` | `run_session(txt_sess, ...)` |
| decoder | `dec_sess.run(None, {6 inputs})` | `run_session(dec_sess, ...)` |
| Score 过滤 | `scores > threshold` (NumPy broadcast) | `for` 循环比较 |
| NMS | 自实现 NumPy 版 | 自实现 C++ 版 |
| Top-K | `argsort()[::-1][:k]` | `keep.resize(k)` (已排序) |
| Box 转换 | `boxes *= [w,h,w,h]` | 手动乘法 |
| Mask sigmoid | `1/(1+np.exp(-x))` | `1/(1+std::exp(-x))` 逐元素 |
| Mask resize | `cv2.resize(INTER_LINEAR)` | `cv::resize()` |
| 可视化 | `np.where` + `cv2.drawContours` | `cv::addWeighted` + `cv::drawContours` |
| 保存 | `cv2.imwrite()` | `cv::imwrite()` |

### 13.2 数据类型对比

| 数据 | Python | C++ |
|------|--------|-----|
| 图像像素 | `np.float32 [1,3,H,W]` | `std::vector<float>` |
| Token IDs | `np.int64 [1,77]` | `std::vector<int64_t>` |
| Score | `np.float32 [200]` | `const float*` (直接指向 ORT 输出) |
| Box | `np.float32 [200,4]` | `const float*` |
| Mask | `np.float32 [200,288,288]` | `const float*` → `cv::Mat CV_32FC1` |
| Binary mask | `np.bool [H,W]` | `cv::Mat CV_8UC1` (0/255) |

### 13.3 关键差异说明

**1. 图像格式**

Python 使用 PIL (RGB)，C++ 使用 OpenCV (BGR)。C++ 版在预处理时通过 `cv::cvtColor(BGR2RGB)` 转换。

**2. Tensor 内存管理**

Python 的 NumPy 自动管理内存。C++ 的 `Ort::Value::CreateTensor` 不拷贝数据，因此 `std::vector<float>` 必须在 `session.Run()` 返回前保持有效。

**3. 模型间数据传递**

Python 版 `img_sess.run()` 返回 NumPy 数组，直接传给 `dec_sess.run()` 的输入字典。

C++ 版使用 `std::move` 将 `Ort::Value` 对象从 image_encoder 输出移动到 decoder 输入，零拷贝：

```cpp
for (int i = 0; i < 4; ++i) dec_inputs.push_back(std::move(img_outputs[i]));
```

**4. NMS 实现差异**

Python 版使用向量化 NumPy 运算 (一次计算所有 IoU)。C++ 版使用标量循环 + `suppressed` 标志数组，逻辑等价但实现风格不同。

---

## 14. TV 平台交叉编译

### 14.1 WebOS (LG)

WebOS SDK 提供 ARM 交叉编译工具链。CMake 配置示例：

```bash
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/webos-toolchain.cmake \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-aarch64 \
    -DOpenCV_DIR=/path/to/opencv-aarch64/lib/cmake/opencv4
```

### 14.2 Tizen (Samsung)

Tizen Studio 提供交叉编译环境。使用 `tizen build-native` 或 CMake：

```bash
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/tizen-toolchain.cmake \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-armv7l \
    -DOpenCV_DIR=/path/to/opencv-armv7l/lib/cmake/opencv4
```

### 14.3 性能调优建议

| 优化项 | 方法 |
|--------|------|
| 线程数 | `opts.SetIntraOpNumThreads(N)` 匹配 TV CPU 核数 |
| 图优化 | `opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL)` |
| 内存优化 | `opts.EnableMemPattern()` / `opts.EnableCpuMemArena()` |
| 分辨率 | 导出 512×512 或 256×256 模型减少计算量 |
| 量化 | ONNX 模型做 INT8 量化 (需验证精度) |
| Mask 省略 | 如只需 box 检测，可修改 decoder 不输出 mask_logits |

### 14.4 运行时文件清单

部署到 TV 上需要的文件：

```
sam3_infer                                  # 可执行文件
exports_repvit_m0_9/
    image_encoder.onnx                      # 1.3 MB
    text_encoder.onnx                       # 0.5 MB
    decoder.onnx                            # 111.3 MB
sam3/assets/
    bpe_simple_vocab_16e6.txt.gz            # ~1 MB
libonnxruntime.so                           # ONNX Runtime 动态库
libopencv_*.so                              # OpenCV 动态库
libz.so                                     # zlib 动态库
```

总计约 **115 MB** (decoder.onnx 占 96%)。

---

## 15. 常见问题

### Q1: 编译报错 `onnxruntime_cxx_api.h not found`

确保 `ONNXRUNTIME_DIR` 指向正确的 ONNX Runtime 安装目录，该目录下应有 `include/onnxruntime_cxx_api.h`。

```bash
cmake .. -DONNXRUNTIME_DIR=/usr/local/onnxruntime-1.17.0
```

### Q2: 链接报错 `libonnxruntime.so not found`

运行时需要 ONNX Runtime 动态库在 `LD_LIBRARY_PATH` 中：

```bash
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

macOS 上使用 `DYLD_LIBRARY_PATH`：

```bash
export DYLD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$DYLD_LIBRARY_PATH
```

### Q3: C++ 和 Python 的 token IDs 不一致

确认使用的 BPE 词表文件是同一个 `bpe_simple_vocab_16e6.txt.gz`。C++ 版的 `regex_tokenize()` 简化了 Unicode 处理，对于纯 ASCII 文本结果应一致。

### Q4: 分数很低 (< 0.2)

这是 stage1 checkpoint 的正常现象。使用 `--score-threshold 0.1` (默认) 或更低的阈值。完整训练后的 checkpoint 会有更高的分数。

### Q5: 如何更换输入分辨率？

需要同时修改两处：
1. ONNX 导出时设置新分辨率 (参见 `ONNX_RESOLUTION_CHANGE_GUIDE.md`)
2. C++ 推理时传 `--resolution 512` (或 256)

### Q6: 如何只做 box 检测不做 mask？

修改 `main.cpp` 中的后处理部分，跳过 mask sigmoid/resize 步骤。decoder 仍会输出 `mask_logits`，但可以不处理。若要彻底优化，需修改 DecoderWrapper 的导出逻辑。

### Q7: TV 上运行太慢怎么办？

- 降低输入分辨率 (1008→512→256)
- 减少 `intra_op_num_threads`
- 使用 INT8 量化模型
- 考虑 GPU 加速 (如 TV 支持 OpenCL/Vulkan EP)
