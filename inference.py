from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import time
import numpy as np
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt


class EdgeTPUInference:
    def __init__(self, model_path):
        """Initialize Edge TPU interpreter with the quantized model"""
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]

    def preprocess_image(self, image_path):
        """Preprocess image for Edge TPU inference"""
        with Image.open(image_path) as img:
            # Resize and center crop
            img = img.resize((self.input_shape[1], self.input_shape[2]), Image.LANCZOS)

            # Convert to numpy array and normalize
            input_data = np.asarray(img)
            input_data = input_data.astype("uint8")  # Edge TPU expects uint8

            return input_data

    def run_inference(self, input_data):
        """Run a single inference"""
        common.set_input(self.interpreter, input_data)
        self.interpreter.invoke()
        return classify.get_classes(self.interpreter, top_k=1)[0]

    def benchmark_inference(self, test_image_dir, num_runs=100):
        """Run inference benchmark"""
        results = {
            "inference_times": [],
            "batch_results": [],
            "throughput": 0,
            "avg_inference_time": 0,
            "std_inference_time": 0,
        }

        # Get list of test images
        image_paths = list(Path(test_image_dir).glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No jpg images found in {test_image_dir}")

        # Warm up
        print("Warming up...")
        warmup_image = self.preprocess_image(str(image_paths[0]))
        for _ in range(10):
            self.run_inference(warmup_image)

        # Run benchmark
        print(f"Running benchmark with {num_runs} iterations...")
        total_time = 0

        for i in range(num_runs):
            # Cycle through available images
            image_path = image_paths[i % len(image_paths)]
            input_data = self.preprocess_image(str(image_path))

            # Time inference
            start_time = time.perf_counter()
            prediction = self.run_inference(input_data)
            inference_time = time.perf_counter() - start_time

            # Convert numpy types to Python native types
            inference_time_ms = float(inference_time * 1000)
            results["inference_times"].append(inference_time_ms)
            results["batch_results"].append(
                {
                    "image": str(image_path),
                    "inference_time_ms": inference_time_ms,
                    "class_id": int(prediction.id),  # Convert to native Python int
                    "score": float(prediction.score),
                }
            )

            total_time += inference_time

        # Calculate statistics and convert to native Python types
        results["throughput"] = float(num_runs / total_time)
        results["avg_inference_time"] = float(np.mean(results["inference_times"]))
        results["std_inference_time"] = float(np.std(results["inference_times"]))

        return results

    def plot_benchmark_results(self, results, output_dir="benchmark_plots"):
        """Generate visualizations of benchmark results"""
        Path(output_dir).mkdir(exist_ok=True)

        # Inference time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results["inference_times"], bins=30)
        plt.title("Inference Time Distribution")
        plt.xlabel("Inference Time (ms)")
        plt.ylabel("Count")
        plt.savefig(f"{output_dir}/inference_time_distribution.png")
        plt.close()

        # Inference times over runs
        plt.figure(figsize=(12, 6))
        plt.plot(results["inference_times"])
        plt.title("Inference Times Over Runs")
        plt.xlabel("Run Number")
        plt.ylabel("Inference Time (ms)")
        plt.axhline(
            y=results["avg_inference_time"],
            color="r",
            linestyle="--",
            label=f'Average ({results["avg_inference_time"]:.2f}ms)',
        )
        plt.legend()
        plt.savefig(f"{output_dir}/inference_times_sequence.png")
        plt.close()


def main():
    # Initialize inference benchmark
    model_path = "quantized_model_edgetpu.tflite"
    benchmark = EdgeTPUInference(model_path)

    # Run benchmark
    results = benchmark.benchmark_inference(test_image_dir="test_images", num_runs=100)

    # Prepare results for JSON serialization
    json_results = {
        "summary": {
            "average_inference_time_ms": float(results["avg_inference_time"]),
            "inference_time_std_ms": float(results["std_inference_time"]),
            "throughput_fps": float(results["throughput"]),
        },
        "detailed_results": results["batch_results"],
    }

    # Save results
    Path("benchmark_results").mkdir(exist_ok=True)
    with open("benchmark_results/edge_tpu_inference.json", "w") as f:
        json.dump(json_results, f, indent=4)

    # Plot results
    benchmark.plot_benchmark_results(results)

    # Print summary
    print("\nBenchmark Results:")
    print("-" * 50)
    print(f"Average inference time: {results['avg_inference_time']:.2f} ms")
    print(f"Inference time std dev: {results['std_inference_time']:.2f} ms")
    print(f"Throughput: {results['throughput']:.2f} FPS")


if __name__ == "__main__":
    main()
