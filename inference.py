from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import time
import numpy as np
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt


class EdgeTPUMonitor:
    def __init__(self):
        """Initialize paths for Edge TPU monitoring"""
        # On this device, we only have thermal_zone0
        self.thermal_zones = {
            "soc": "/sys/class/thermal/thermal_zone0/temp",  # This represents the system-on-chip temperature
        }

    def read_thermal(self, zone):
        """Read temperature from a thermal zone"""
        try:
            with open(self.thermal_zones[zone], "r") as f:
                # Temperature is reported in millicelsius
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, KeyError):
            return None


class EdgeTPUInference:
    def __init__(self, model_path):
        """Initialize Edge TPU interpreter with the quantized model"""
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]

        # Initialize monitoring
        self.monitor = EdgeTPUMonitor()

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

    def benchmark_inference(self, test_image_dir, num_runs=500):
        """Run comprehensive inference benchmark with detailed performance metrics"""
        results = {
            "inference_times": [],
            "batch_results": [],
            "monitoring": {
                "temperature": {"cpu": [], "tpu": []},
                "power": [],
                "timestamps": [],
            },
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

        # Initial monitoring reading
        start_temp_soc = self.monitor.read_thermal(
            "soc"
        )  # Changed from start_temp_cpu/tpu

        # Warm up
        print("Warming up...")
        warmup_image = self.preprocess_image(str(image_paths[0]))
        for _ in range(10):
            self.run_inference(warmup_image)

        # Run benchmark
        print(f"Running benchmark with {num_runs} iterations...")
        total_time = 0
        total_active_time = 0
        start_time = time.time()

        for i in range(num_runs):
            # Record monitoring data
            current_time = time.time() - start_time
            results["monitoring"]["timestamps"].append(float(current_time))

            # In benchmark_inference method, replace the temperature monitoring part with:
            temp_soc = self.monitor.read_thermal("soc")
            if temp_soc:
                results["monitoring"]["temperature"]["soc"] = []
                results["monitoring"]["temperature"]["soc"].append(float(temp_soc))

            # And update the thermal stats calculation:
            if results["monitoring"]["temperature"]["soc"]:
                results["thermal_stats"] = {
                    "soc": {
                        "avg_temp": float(
                            np.mean(results["monitoring"]["temperature"]["soc"])
                        ),
                        "max_temp": float(
                            np.max(results["monitoring"]["temperature"]["soc"])
                        ),
                        "temp_increase": float(
                            results["monitoring"]["temperature"]["soc"][-1]
                            - start_temp_soc
                            if start_temp_soc
                            else 0
                        ),
                    }
                }

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
                    "temperature_soc": temp_soc,
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

    def measure_performance_metrics(self, input_data):
        """Measure detailed performance metrics for a single inference"""
        metrics = {}

        # Measure preprocessing time
        preprocess_start = time.perf_counter()
        processed_data = self.preprocess_image(input_data)
        metrics["preprocess_time"] = time.perf_counter() - preprocess_start

        # Measure actual inference time
        inference_start = time.perf_counter()
        common.set_input(self.interpreter, processed_data)
        invoke_start = time.perf_counter()
        self.interpreter.invoke()
        invoke_end = time.perf_counter()
        results = classify.get_classes(self.interpreter, top_k=1)
        inference_end = time.perf_counter()

        # Calculate different timing components
        metrics["total_inference_time"] = inference_end - inference_start
        metrics["invoke_time"] = invoke_end - invoke_start
        metrics["overhead_time"] = (
            metrics["total_inference_time"] - metrics["invoke_time"]
        )

        return metrics, results[0]

    def plot_monitoring_results(self, results, output_dir="benchmark_plots"):
        """Plot thermal and power monitoring results"""
        Path(output_dir).mkdir(exist_ok=True)

        # Temperature over time
        plt.figure(figsize=(12, 6))
        timestamps = results["monitoring"]["timestamps"]

        if results["monitoring"]["temperature"]["cpu"]:
            plt.plot(
                timestamps,
                results["monitoring"]["temperature"]["cpu"],
                label="CPU Temperature",
            )
        if results["monitoring"]["temperature"]["tpu"]:
            plt.plot(
                timestamps,
                results["monitoring"]["temperature"]["tpu"],
                label="TPU Temperature",
            )

        plt.title("Temperature Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.savefig(f"{output_dir}/temperature_over_time.png")
        plt.close()

        # Power over time
        if results["monitoring"]["power"]:
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, results["monitoring"]["power"])
            plt.title("Power Consumption Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Power (Watts)")
            plt.savefig(f"{output_dir}/power_over_time.png")
            plt.close()

    def plot_performance_metrics(self, results, output_dir="benchmark_plots"):
        """Generate enhanced performance visualizations"""
        # Latency distribution
        plt.figure(figsize=(12, 6))
        plt.hist(
            results["performance_metrics"]["inference_times"],
            bins=30,
            alpha=0.7,
            label="Total Inference",
        )
        plt.hist(
            results["performance_metrics"]["invoke_times"],
            bins=30,
            alpha=0.7,
            label="TPU Invoke",
        )
        plt.title("Latency Distribution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"{output_dir}/latency_distribution.png")
        plt.close()

        # Time components breakdown
        plt.figure(figsize=(10, 6))
        components = ["preprocessing_times", "invoke_times", "overhead_times"]
        means = [np.mean(results["performance_metrics"][c]) for c in components]
        plt.bar(components, means)
        plt.title("Average Time Components")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_components.png")
        plt.close()


def main():
    # Initialize inference benchmark
    model_path = "models/quantized_model_edgetpu.tflite"
    benchmark = EdgeTPUInference(model_path)

    # Run benchmark
    results = benchmark.benchmark_inference(test_image_dir="test_images", num_runs=500)

    # Save results
    Path("benchmark_results").mkdir(exist_ok=True)
    with open("benchmark_results/edge_tpu_inference.json", "w") as f:
        json.dump(
            {
                "summary": {
                    "average_inference_time_ms": float(results["avg_inference_time"]),
                    "inference_time_std_ms": float(results["std_inference_time"]),
                    "throughput_fps": float(results["throughput"]),
                    "thermal_stats": results.get("thermal_stats", {}),
                    "power_stats": results.get("power_stats", {}),
                },
                "detailed_results": results["batch_results"],
            },
            f,
            indent=4,
        )

    # Plot results
    benchmark.plot_monitoring_results(results)

    # Print summary
    print("\nBenchmark Results:")
    print("-" * 50)
    print(f"Average inference time: {results['avg_inference_time']:.2f} ms")
    print(f"Inference time std dev: {results['std_inference_time']:.2f} ms")
    print(f"Throughput: {results['throughput']:.2f} FPS")

    if "thermal_stats" in results:
        print("\nThermal Results:")
        if "tpu" in results["thermal_stats"]:
            print(
                f"Max TPU Temperature: {results['thermal_stats']['tpu']['max_temp']:.1f}°C"
            )
            print(
                f"TPU Temperature Increase: {results['thermal_stats']['tpu']['temp_increase']:.1f}°C"
            )

    if "power_stats" in results:
        print("\nPower Results:")
        print(f"Average Power: {results['power_stats']['avg_power']:.2f}W")
        print(f"Total Energy: {results['power_stats']['total_energy']:.2f}Ws")


if __name__ == "__main__":
    main()
