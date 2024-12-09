
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
import multiprocessing
import os

class BaseTrainingPipeline:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.history = {}
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.quantized_model = None
        # Add batch size as class attribute for benchmarking
        self.batch_size = 32
        self.image_size = (224, 224)

    def create_model(self):  # Remove num_classes parameter as it's a class attribute
        """Create a fine-tunable MobileNetV2 model"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom classification head
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(base_model.input, outputs)
        
        # Compile model here to match benchmark requirements
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model  # Return model instead of setting self.model

    def preprocess_data(self, image, label):
        """Preprocess images for MobileNetV2"""
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, self.image_size)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    def create_dataset(self, dataset, is_training=True):  # Add is_training parameter to match TPU pipeline
        """Create an optimized tf.data pipeline"""
        AUTOTUNE = tf.data.AUTOTUNE
        
        dataset = dataset.map(self.preprocess_data, num_parallel_calls=AUTOTUNE)
        
        if is_training:
            dataset = dataset.cache().shuffle(1000)
        
        return dataset.batch(self.batch_size).prefetch(AUTOTUNE)

    def train(self, train_dataset, val_dataset, epochs=5):  # Modified to match benchmark interface
        """Train the model using an efficient training loop"""
        model = self.create_model()
        
        # Use multiple workers for training
        train_workers = min(multiprocessing.cpu_count(), 4)
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            workers=train_workers,
            use_multiprocessing=True
        )
        
        # Return results in the same format as TPU pipeline
        return {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }

    def quantize_model(self, model, train_dataset):  # Add train_dataset parameter
        """Convert model to TFLite format with quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Representative dataset for quantization
        def representative_dataset():
            for data in train_dataset.take(100):
                yield [tf.dtypes.cast(data[0], tf.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Convert the model
        quantized_tflite_model = converter.convert()
        
        # Save the quantized model
        with open('quantized_model.tflite', 'wb') as f:
            f.write(quantized_tflite_model)
        
        return quantized_tflite_model  # Return instead of setting self.quantized_model
