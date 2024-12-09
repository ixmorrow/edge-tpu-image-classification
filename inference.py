import time
import os
from pathlib import Path
import numpy as np
from PIL import Image
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import tensorflow_datasets as tfds
import pandas as pd
from tabulate import tabulate


class EdgeTPUTester:
    def __init__(self, model_path, dataset_name="tf_flowers"):
        # Initialize the Edge TPU interpreter
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Load dataset info for label mapping
        _, info = tfds.load(dataset_name, with_info=True)
        self.labels = info.features["label"].names

    def prepare_image(self, image_path):
        """Prepare image for inference"""
        img = Image.open(image_path)
        img = img.resize((224, 224), Image.LANCZOS)
        return np.asarray(img).astype("uint8")

    def run_inference(self, image):
        """Run inference and time it"""
        start_time = time.perf_counter()

        # Set input and run inference
        common.set_input(self.interpreter, image)
        self.interpreter.invoke()

        # Get results
        classes = classify.get_classes(self.interpreter, top_k=3)

        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        return classes, inference_time

    def test_batch(self, test_image_dir):
        """Run tests on a batch of images"""
        results = []

        for image_path in Path(test_image_dir).glob("*.jpg"):
            image = self.prepare_image(image_path)
            predictions, inf_time = self.run_inference(image)

            # Get top prediction
            top_pred = predictions[0]

            results.append(
                {
                    "image": image_path.name,
                    "predicted_label": self.labels[top_pred.id],
                    "confidence": f"{top_pred.score*100:.2f}%",
                    "inference_time_ms": f"{inf_time:.2f}",
                    "top_3_predictions": [
                        (self.labels[p.id], f"{p.score*100:.2f}%") for p in predictions
                    ],
                }
            )

        return results


def create_test_set(output_dir, num_images=10):
    """Create a test set from TF Flowers dataset"""
    # Load some test images from the dataset
    dataset = tfds.load("tf_flowers", split="train[:{}]".format(num_images))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save images
    for i, example in enumerate(dataset):
        image = example["image"].numpy()
        image = Image.fromarray(image)
        image.save(os.path.join(output_dir, f"test_image_{i}.jpg"))


def display_results(results):
    """Display results in a formatted table"""
    # Create summary table
    summary_data = [
        [r["image"], r["predicted_label"], r["confidence"], r["inference_time_ms"]]
        for r in results
    ]

    print("\n=== Inference Results ===")
    print(
        tabulate(
            summary_data,
            headers=["Image", "Prediction", "Confidence", "Time (ms)"],
            tablefmt="grid",
        )
    )

    # Calculate and display statistics
    inf_times = [float(r["inference_time_ms"]) for r in results]
    print(f"\nPerformance Statistics:")
    print(f"Average inference time: {np.mean(inf_times):.2f} ms")
    print(f"Std dev inference time: {np.std(inf_times):.2f} ms")
    print(f"Max inference time: {np.max(inf_times):.2f} ms")
    print(f"Min inference time: {np.min(inf_times):.2f} ms")

    # Display detailed predictions
    print("\n=== Detailed Top-3 Predictions ===")
    for r in results:
        print(f"\nImage: {r['image']}")
        for label, conf in r["top_3_predictions"]:
            print(f"  {label}: {conf}")


def main():
    # Set up paths
    TEST_DIR = "test_images"
    MODEL_PATH = "models/quantized_model_edgetpu.tflite"

    # Create test set
    print("Creating test set...")
    create_test_set(TEST_DIR)

    # Initialize tester
    print("Initializing Edge TPU tester...")
    tester = EdgeTPUTester(MODEL_PATH)

    # Run tests
    print("Running inference tests...")
    results = tester.test_batch(TEST_DIR)

    # Display results
    display_results(results)


if __name__ == "__main__":
    main()
