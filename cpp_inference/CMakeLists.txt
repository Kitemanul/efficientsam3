cmake_minimum_required(VERSION 3.14)
project(sam3_onnx_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Dependencies ----
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

# ONNX Runtime: set via -DONNXRUNTIME_DIR=/path/to/onnxruntime
if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR
        "ONNXRUNTIME_DIR is not set.\n"
        "Usage: cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..")
endif()

# ---- Executable ----
add_executable(sam3_infer
    main.cpp
    simple_tokenizer.cpp
)

target_include_directories(sam3_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(sam3_infer PRIVATE
    ${OpenCV_LIBS}
    ZLIB::ZLIB
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}
)
