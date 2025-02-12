# Optimizing Attention-Based Flower Classification on the Edge TPU

This repository contains an end-to-end workflow demonstrating how to fine-tune and quantize a MobileNetV2 model—enhanced with an attention mechanism—for efficient deployment on a Coral Edge TPU. The project focuses on data pipeline optimizations and hardware-aware design to achieve low-latency inference on edge devices while maintaining high classification accuracy.

Key Features

	• Data Pipeline and TFRecords
	• Converts images from the TensorFlow Flowers dataset into TFRecord files, enabling faster loading and parallel preprocessing.
	• Uses @tf.function decorators and the tf.data API with AUTOTUNE for efficient batch processing and on-the-fly data augmentation.
	• Model Architecture
	• Builds on MobileNetV2 pre-trained on ImageNet, then adds a GlobalPooling2D layer and a custom attention layer to learn the most relevant features for flower classification.
	• Optimized for TPU execution by using power-of-two layer dimensions and ensuring minimal unsupported operations.
	• Edge TPU Deployment
	• Demonstrates 8-bit quantization using TensorFlow Lite, then compiles the model with the Edge TPU Compiler for on-device inference.
	• Shows how to warm up the Edge TPU to reduce first-inference latency and benchmarks inference speed and accuracy on real test images.
	• Performance Benchmarks
	• Compares baseline vs. TPU-optimized training pipelines in a Google Colab TPU environment, highlighting significant speedups when leveraging multiple TPU cores.
	• Achieves ~90% test accuracy on the Edge TPU, with only a ~1.4% drop from the full-precision baseline.

## Why This Matters

Deploying computer vision models on edge devices can be challenging due to compute constraints. By combining quantization, attention-based architectures, and system-level optimizations (such as TFRecords, parallel processing, and TPU-aware layers), this project illustrates a practical approach to delivering real-time inference with minimal resource usage.

Feel free to explore the notebooks and scripts in this repo for step-by-step instructions on data preparation, model fine-tuning, quantization, and final deployment to the Coral Edge TPU.