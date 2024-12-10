from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import time
import numpy as np
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os


class EdgeTPUInference:
    def __init__(self, model_path):
        """Initialize Edge TPU interpreter with the quantized model"""
        # Measure model loading time
        self.model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB

        # Measure TPU model loading time
        load_start = time.perf_counter()
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.tpu_load_time = time.perf_counter() - load_start

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]

    def preprocess_image(self, image_path):
        """Preprocess image for Edge TPU inference"""
        with Image.open(image_path) as img:
            # Resize to match model's expected input shape
            img = img.resize((self.input_shape[1], self.input_shape[2]), Image.LANCZOS)

            # Convert to numpy array
            input_data = np.asarray(img)
            input_data = input_data.astype("uint8")  # Edge TPU expects uint8

            return input_data

    def run_inference(self, input_data):
        """Run a single inference"""
        common.set_input(self.interpreter, input_data)
        self.interpreter.invoke()
        return classify.get_classes(self.interpreter, top_k=1)[0]

    def benchmark_inference(self, test_image_dir, num_runs=500, warmup=True):
        """Run comprehensive inference benchmark with detailed performance metrics"""
        results = {
            "warmup": warmup,
            "model_metrics": {
                "model_size_mb": self.model_size,
                "tpu_load_time": self.tpu_load_time,
            },
            "inference_times": [],
            "batch_results": [],
            "performance_metrics": {
                "preprocessing_times": [],
                "inference_times": [],
                "invoke_times": [],
                "overhead_times": [],
                "tail_latencies": {},
                "duty_cycle": 0.0,
            },
            "throughput": 0,
            "avg_inference_time": 0,
            "std_inference_time": 0,
        }

        # Get list of test images
        image_paths = list(Path(test_image_dir).glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No jpg images found in {test_image_dir}")

        if warmup:
            # Warm up
            print("Warming up...")
            warmup_image = self.preprocess_image(str(image_paths[0]))
            for _ in range(10):
                self.run_inference(warmup_image)

        # Run benchmark
        print(f"Running benchmark with {num_runs} iterations...")
        total_time = 0
        total_active_time = 0

        for i in range(num_runs):
            # Run inference with detailed timing
            image_path = image_paths[i % len(image_paths)]

            # Measure preprocessing time
            preprocess_start = time.perf_counter()
            input_data = self.preprocess_image(str(image_path))
            preprocess_time = time.perf_counter() - preprocess_start

            # Measure inference components
            inference_start = time.perf_counter()
            common.set_input(self.interpreter, input_data)
            invoke_start = time.perf_counter()
            self.interpreter.invoke()
            invoke_end = time.perf_counter()
            prediction = classify.get_classes(self.interpreter, top_k=1)[0]
            inference_end = time.perf_counter()

            # Calculate timing metrics
            invoke_time = invoke_end - invoke_start
            total_inference_time = inference_end - inference_start
            overhead_time = total_inference_time - invoke_time

            # Record all timing metrics
            results["inference_times"].append(
                float(total_inference_time * 1000)
            )  # Convert to ms
            results["performance_metrics"]["preprocessing_times"].append(
                float(preprocess_time * 1000)
            )
            results["performance_metrics"]["inference_times"].append(
                float(total_inference_time * 1000)
            )
            results["performance_metrics"]["invoke_times"].append(
                float(invoke_time * 1000)
            )
            results["performance_metrics"]["overhead_times"].append(
                float(overhead_time * 1000)
            )

            total_time += total_inference_time
            total_active_time += invoke_time

            # Record detailed results
            results["batch_results"].append(
                {
                    "image": str(image_path),
                    "inference_time_ms": float(total_inference_time * 1000),
                    "preprocessing_time_ms": float(preprocess_time * 1000),
                    "invoke_time_ms": float(invoke_time * 1000),
                    "overhead_time_ms": float(overhead_time * 1000),
                    "class_id": int(prediction.id),
                    "score": float(prediction.score),
                }
            )

        # Calculate basic statistics
        results["throughput"] = float(num_runs / total_time)
        results["avg_inference_time"] = float(np.mean(results["inference_times"]))
        results["std_inference_time"] = float(np.std(results["inference_times"]))

        # Calculate detailed performance statistics
        for metric_name in [
            "preprocessing_times",
            "inference_times",
            "invoke_times",
            "overhead_times",
        ]:
            times = results["performance_metrics"][metric_name]
            results["performance_metrics"][f"{metric_name}_stats"] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "p95": float(np.percentile(times, 95)),
                "p99": float(np.percentile(times, 99)),
            }

        # Calculate duty cycle
        results["performance_metrics"]["duty_cycle"] = float(
            total_active_time / total_time
        )

        # Calculate throughput stability
        inference_times = results["performance_metrics"]["inference_times"]
        results["performance_metrics"]["throughput_stability"] = {
            "coefficient_of_variation": float(
                np.std(inference_times) / np.mean(inference_times)
            ),
            "max_deviation": float(
                np.max(np.abs(inference_times - np.mean(inference_times)))
            ),
        }

        return results

    from pycoral.utils import edgetpu


from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import numpy as np
import time
from pathlib import Path


def evaluate_model_on_test_set(model_path, test_dir, labels_file):
    """Evaluate quantized model on test set using Edge TPU"""
    # Initialize interpreter
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    input_size = input_details[0]["shape"][1]

    # Load labels
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            image_name, label = line.strip().split(",")
            labels[image_name] = int(label)

    # Initialize metrics
    correct = 0
    total = 0
    inference_times = []
    class_correct = np.zeros(5)  # Assuming 5 flower classes
    class_total = np.zeros(5)

    # Process each test image
    for image_file in Path(test_dir).glob("*.jpg"):
        if image_file.name not in labels:
            continue

        # Load and preprocess image
        with Image.open(image_file) as img:
            img = img.resize((input_size, input_size), Image.LANCZOS)
            img = np.asarray(img)

        # Run inference
        start_time = time.perf_counter()
        common.set_input(interpreter, img)
        interpreter.invoke()
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)

        # Get prediction
        classes = classify.get_classes(interpreter, top_k=1)[0]
        predicted_class = classes.id
        true_class = labels[image_file.name]

        # Update metrics
        total += 1
        if predicted_class == true_class:
            correct += 1
            class_correct[true_class] += 1
        class_total[true_class] += 1

    # Calculate metrics
    accuracy = correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    std_inference_time = np.std(inference_times) * 1000

    # Print results
    print("\nEdge TPU Evaluation Results:")
    print("-" * 50)
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Inference Time Std Dev: {std_inference_time:.2f} ms")
    print("\nPer-class Accuracy:")
    for i in range(5):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(
                f"Class {i}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})"
            )

    return {
        "accuracy": accuracy,
        "avg_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "total_correct": correct,
        "total_samples": total,
        "class_correct": class_correct.tolist(),
        "class_total": class_total.tolist(),
        "inference_times": inference_times,
    }


def plot_performance_metrics(results, output_dir="benchmark_plots"):
    """Generate performance visualization plots"""
    Path(output_dir).mkdir(exist_ok=True)

    # Compare warmup vs no warmup inference times
    plt.figure(figsize=(12, 6))
    plt.hist(
        results["warmup"]["performance_metrics"]["inference_times"],
        bins=30,
        alpha=0.7,
        label="With Warmup",
    )
    plt.hist(
        results["no_warmup"]["performance_metrics"]["inference_times"],
        bins=30,
        alpha=0.7,
        label="No Warmup",
    )
    plt.title("Inference Time Distribution")
    plt.xlabel("Time (ms)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"{output_dir}/inference_time_distribution.png")
    plt.close()

    # Time components breakdown
    plt.figure(figsize=(10, 6))
    components = ["preprocessing_times", "invoke_times", "overhead_times"]
    warm_means = [
        np.mean(results["warmup"]["performance_metrics"][c]) for c in components
    ]
    no_warm_means = [
        np.mean(results["no_warmup"]["performance_metrics"][c]) for c in components
    ]

    x = np.arange(len(components))
    width = 0.35

    plt.bar(x - width / 2, warm_means, width, label="With Warmup")
    plt.bar(x + width / 2, no_warm_means, width, label="No Warmup")
    plt.title("Average Time Components")
    plt.ylabel("Time (ms)")
    plt.xticks(x, ["Preprocessing", "Invoke", "Overhead"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_components.png")
    plt.close()

    # Throughput stability over time
    plt.figure(figsize=(12, 6))
    warm_throughput = [
        1000 / t for t in results["warmup"]["performance_metrics"]["inference_times"]
    ]
    no_warm_throughput = [
        1000 / t for t in results["no_warmup"]["performance_metrics"]["inference_times"]
    ]

    plt.plot(warm_throughput, label="With Warmup", alpha=0.7)
    plt.plot(no_warm_throughput, label="No Warmup", alpha=0.7)
    plt.title("Throughput Stability Over Time")
    plt.xlabel("Inference Number")
    plt.ylabel("Throughput (FPS)")
    plt.legend()
    plt.savefig(f"{output_dir}/throughput_stability.png")
    plt.close()

    # Tail latency analysis
    plt.figure(figsize=(12, 6))
    percentiles = range(1, 101)
    warm_percentiles = [
        np.percentile(results["warmup"]["performance_metrics"]["inference_times"], p)
        for p in percentiles
    ]
    no_warm_percentiles = [
        np.percentile(results["no_warmup"]["performance_metrics"]["inference_times"], p)
        for p in percentiles
    ]

    plt.plot(percentiles, warm_percentiles, label="With Warmup")
    plt.plot(percentiles, no_warm_percentiles, label="No Warmup")
    plt.title("Tail Latency Analysis")
    plt.xlabel("Percentile")
    plt.ylabel("Latency (ms)")
    plt.axvline(x=95, color="r", linestyle="--", alpha=0.5, label="P95")
    plt.axvline(x=99, color="g", linestyle="--", alpha=0.5, label="P99")
    plt.legend()
    plt.savefig(f"{output_dir}/tail_latency.png")
    plt.close()

    # Duty cycle comparison
    plt.figure(figsize=(8, 6))
    duty_cycles = [
        results["warmup"]["performance_metrics"]["duty_cycle"],
        results["no_warmup"]["performance_metrics"]["duty_cycle"],
    ]
    plt.bar(["With Warmup", "No Warmup"], duty_cycles)
    plt.title("TPU Duty Cycle")
    plt.ylabel("Duty Cycle (ratio)")
    plt.savefig(f"{output_dir}/duty_cycle.png")
    plt.close()


def main():
    # Initialize inference benchmark
    model_path = "models/tpu-optimized/tpu_optimized_quantized_model_edgetpu.tflite"
    benchmark = EdgeTPUInference(model_path)

    # Run benchmark w/o warmup
    no_warm_results = benchmark.benchmark_inference(
        test_image_dir="test_images", num_runs=1000, warmup=False
    )

    # Run benchmark w/ warmup
    warm_results = benchmark.benchmark_inference(
        test_image_dir="test_images", num_runs=1000, warmup=True
    )

    print("Evaluating Baseline model on test set...")
    test_results = evaluate_model_on_test_set(
        "models/baseline/quantized_model_edgetpu.tflite",
        "test_dataset",
        "test_dataset/labels.txt",
    )

    print("Evaluating TPU Optimized model on test set...")
    test_results = evaluate_model_on_test_set(
        model_path, "test_dataset", "test_dataset/labels.txt"
    )

    # Save results
    Path("benchmark_results").mkdir(exist_ok=True)
    with open("benchmark_results/edge_tpu_inference.json", "w") as f:
        json.dump(
            {
                "test_eval": test_results,
                "warmup": {
                    "summary": {
                        "warmup": warm_results["warmup"],
                        "model_size_mb": float(
                            warm_results["model_metrics"]["model_size_mb"]
                        ),
                        "average_inference_time_ms": float(
                            warm_results["avg_inference_time"]
                        ),
                        "inference_time_std_ms": float(
                            warm_results["std_inference_time"]
                        ),
                        "throughput_fps": float(warm_results["throughput"]),
                    },
                    "detailed_performance_metrics": {
                        "preprocess_times": warm_results["performance_metrics"][
                            "preprocessing_times"
                        ],
                        "inference_times": warm_results["performance_metrics"][
                            "inference_times"
                        ],
                        "invoke_times": warm_results["performance_metrics"][
                            "invoke_times"
                        ],
                        "overhead_times": warm_results["performance_metrics"][
                            "overhead_times"
                        ],
                        "tail_latencies": warm_results["performance_metrics"][
                            "tail_latencies"
                        ],
                        "duty_cycle": warm_results["performance_metrics"]["duty_cycle"],
                    },
                    "detailed_results": warm_results["batch_results"],
                },
                "no_warmup": {
                    "summary": {
                        "warmup": no_warm_results["warmup"],
                        "model_size_mb": float(
                            no_warm_results["model_metrics"]["model_size_mb"]
                        ),
                        "average_inference_time_ms": float(
                            no_warm_results["avg_inference_time"]
                        ),
                        "inference_time_std_ms": float(
                            no_warm_results["std_inference_time"]
                        ),
                        "throughput_fps": float(no_warm_results["throughput"]),
                    },
                    "detailed_performance_metrics": {
                        "preprocess_times": no_warm_results["performance_metrics"][
                            "preprocessing_times"
                        ],
                        "inference_times": no_warm_results["performance_metrics"][
                            "inference_times"
                        ],
                        "invoke_times": no_warm_results["performance_metrics"][
                            "invoke_times"
                        ],
                        "overhead_times": no_warm_results["performance_metrics"][
                            "overhead_times"
                        ],
                        "tail_latencies": no_warm_results["performance_metrics"][
                            "tail_latencies"
                        ],
                        "duty_cycle": no_warm_results["performance_metrics"][
                            "duty_cycle"
                        ],
                    },
                    "detailed_results": no_warm_results["batch_results"],
                },
            },
            f,
            indent=4,
        )

    # Generate plots
    plot_performance_metrics({"warmup": warm_results, "no_warmup": no_warm_results})

    # Print summary
    print("\nBenchmark Results:")
    print("-" * 50)
    print("Model without Warmup:\n")
    print(f"Model Size: {no_warm_results['model_metrics']['model_size_mb']:.2f} mb")
    print(f"First inference latency: {no_warm_results['inference_times'][0]:.2f} ms")
    print(f"Average inference time: {no_warm_results['avg_inference_time']:.2f} ms")
    print(f"Inference time std dev: {no_warm_results['std_inference_time']:.2f} ms")
    print(f"Throughput: {no_warm_results['throughput']:.2f} FPS")

    print("\nWarmed up model:")
    print(f"Model Size: {warm_results['model_metrics']['model_size_mb']:.2f} mb")
    print(f"First inference latency: {warm_results['inference_times'][0]:.2f} ms")
    print(f"Average inference time: {warm_results['avg_inference_time']:.2f} ms")
    print(f"Inference time std dev: {warm_results['std_inference_time']:.2f} ms")
    print(f"Throughput: {warm_results['throughput']:.2f} FPS")


if __name__ == "__main__":
    main()
