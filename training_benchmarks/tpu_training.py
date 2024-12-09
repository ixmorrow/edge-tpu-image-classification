
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
import os
import json
from datetime import datetime

class TPUTrainingPipeline:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.history = {}
        
        # Initialize TPU strategy
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            self.strategy = tf.distribute.TPUStrategy(resolver)
            print(f"Training on TPU: {resolver.cluster_spec().as_dict()}")
        except ValueError:
            print("No TPU detected. Using CPU/GPU strategy")
            self.strategy = tf.distribute.MirroredStrategy()
        
        print(f'Number of replicas: {self.strategy.num_replicas_in_sync}')
        
        # TPU-optimized batch size
        self.batch_size = 128 * self.strategy.num_replicas_in_sync
        self.image_size = (224, 224)
        
    def create_model(self):
        """Create model within TPU strategy scope"""
        with self.strategy.scope():
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            
            base_model.trainable = False
            
            x = layers.GlobalAveragePooling2D()(base_model.output)
            
            # TPU-optimized attention mechanism
            attention = layers.Dense(512, activation='relu')(x)
            attention = layers.Dense(x.shape[-1], activation='sigmoid')(attention)
            attention_output = layers.Multiply()([x, attention])
            
            # TPU-friendly layer sizes (powers of two)
            x = layers.Dense(512, activation='relu')(attention_output)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(base_model.input, outputs)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model

    def create_dataset(self, dataset, is_training=True):
        """Create TPU-optimized dataset pipeline"""
        AUTOTUNE = tf.data.AUTOTUNE
        
        @tf.function
        def preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.image_size)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return image, label
        
        @tf.function
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            return image, label
        
        dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
        
        if is_training:
            dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
            dataset = dataset.shuffle(10000)
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
    
    def train(self, train_dataset, val_dataset, epochs=10):
        """Train the model using TPU-optimized settings"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        model = self.create_model()
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Return results in same format as base pipeline for fair comparison
        return {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
