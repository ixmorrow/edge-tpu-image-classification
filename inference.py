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
        # Thermal zones paths
        self.thermal_zones = {
            "cpu": "/sys/class/thermal/thermal_zone0/temp",
            "tpu": "/sys/class/thermal/thermal_zone1/temp",
        }

        # Power monitoring paths for Coral Dev Board
        self.power_paths = {
            "voltage": "/sys/bus/i2c/devices/0-0040/in1_input",
            "current": "/sys/bus/i2c/devices/0-0040/curr1_input",
        }

    def read_thermal(self, zone):
        """Read temperature from a thermal zone"""
        try:
            with open(self.thermal_zones[zone], "r") as f:
                # Temperature is reported in millicelsius
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, KeyError):
            return None

    def read_power(self):
        """Read power consumption"""
        try:
            with open(self.power_paths["voltage"], "r") as f:
                voltage = float(f.read().strip()) / 1000.0  # Convert to V
            with open(self.power_paths["current"], "r") as f:
                current = float(f.read().strip()) / 1000.0  # Convert to A
            return voltage * current  # Power in Watts
        except FileNotFoundError:
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

    def benchmark_inference(self, test_image_dir, num_runs=100):
        """Run inference benchmark with monitoring"""
        results = {
            "inference_times": [],
            "batch_results": [],
            "monitoring": {
                "temperature": {"cpu": [], "tpu": []},
                "power": [],
                "timestamps": [],
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
        start_temp_cpu = self.monitor.read_thermal("cpu")
        start_temp_tpu = self.monitor.read_thermal("tpu")
        start_power = self.monitor.read_power()

        # Warm up
        print("Warming up...")
        warmup_image = self.preprocess_image(str(image_paths[0]))
        for _ in range(10):
            self.run_inference(warmup_image)

        # Run benchmark
        print(f"Running benchmark with {num_runs} iterations...")
        total_time = 0
        start_time = time.time()

        for i in range(num_runs):
            # Record monitoring data
            current_time = time.time() - start_time
            results["monitoring"]["timestamps"].append(float(current_time))

            temp_cpu = self.monitor.read_thermal("cpu")
            temp_tpu = self.monitor.read_thermal("tpu")
            power = self.monitor.read_power()

            if temp_cpu:
                results["monitoring"]["temperature"]["cpu"].append(float(temp_cpu))
            if temp_tpu:
                results["monitoring"]["temperature"]["tpu"].append(float(temp_tpu))
            if power:
                results["monitoring"]["power"].append(float(power))

            # Run inference
            image_path = image_paths[i % len(image_paths)]
            input_data = self.preprocess_image(str(image_path))

            inference_start = time.perf_counter()
            prediction = self.run_inference(input_data)
            inference_time = time.perf_counter() - inference_start

            results["inference_times"].append(float(inference_time * 1000))
            results["batch_results"].append(
                {
                    "image": str(image_path),
                    "inference_time_ms": float(inference_time * 1000),
                    "class_id": int(prediction.id),
                    "score": float(prediction.score),
                    "temperature_cpu": temp_cpu,
                    "temperature_tpu": temp_tpu,
                    "power": power,
                }
            )

            total_time += inference_time

        # Calculate statistics
        results["throughput"] = float(num_runs / total_time)
        results["avg_inference_time"] = float(np.mean(results["inference_times"]))
        results["std_inference_time"] = float(np.std(results["inference_times"]))

        # Calculate thermal and power statistics
        if results["monitoring"]["temperature"]["cpu"]:
            results["thermal_stats"] = {
                "cpu": {
                    "avg_temp": float(
                        np.mean(results["monitoring"]["temperature"]["cpu"])
                    ),
                    "max_temp": float(
                        np.max(results["monitoring"]["temperature"]["cpu"])
                    ),
                    "temp_increase": float(
                        results["monitoring"]["temperature"]["cpu"][-1] - start_temp_cpu
                        if start_temp_cpu
                        else 0
                    ),
                }
            }
        if results["monitoring"]["temperature"]["tpu"]:
            results["thermal_stats"]["tpu"] = {
                "avg_temp": float(np.mean(results["monitoring"]["temperature"]["tpu"])),
                "max_temp": float(np.max(results["monitoring"]["temperature"]["tpu"])),
                "temp_increase": float(
                    results["monitoring"]["temperature"]["tpu"][-1] - start_temp_tpu
                    if start_temp_tpu
                    else 0
                ),
            }
        if results["monitoring"]["power"]:
            results["power_stats"] = {
                "avg_power": float(np.mean(results["monitoring"]["power"])),
                "max_power": float(np.max(results["monitoring"]["power"])),
                "total_energy": float(
                    np.sum(results["monitoring"]["power"])
                    * (total_time / len(results["monitoring"]["power"]))
                ),  # Watt-seconds
            }

        return results

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
