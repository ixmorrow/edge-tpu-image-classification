from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import os
import time


def test_inference(interpreter, image_path):
    """Run a single inference and return results"""
    # Open and resize image
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.LANCZOS)

    # Convert to numpy array
    input_data = img

    # Time the inference
    start = time.perf_counter()

    # Run inference
    common.set_input(interpreter, input_data)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    inference_time = (time.perf_counter() - start) * 1000  # ms

    return classes[0].id, classes[0].score, inference_time


def main():
    # Load model
    MODEL_PATH = "models/quantized_model_edgetpu.tflite"
    TEST_DIR = "test_images"  # Directory containing your test images

    interpreter = edgetpu.make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    # Test each image in the directory
    print("Image | Class ID | Confidence | Time (ms)")
    print("-" * 40)

    total_time = 0
    count = 0

    for image_name in os.listdir(TEST_DIR):
        if image_name.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(TEST_DIR, image_name)
            class_id, score, inf_time = test_inference(interpreter, image_path)

            print(f"{image_name} | {class_id} | {score:.2f} | {inf_time:.2f}")

            total_time += inf_time
            count += 1

    if count > 0:
        avg_time = total_time / count
        print(f"\nAverage inference time: {avg_time:.2f} ms")


if __name__ == "__main__":
    main()
