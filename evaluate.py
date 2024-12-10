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
