import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5
MODEL_FILENAME = "plant_disease_model.h5"
LABELS_FILENAME = "class_labels.npy"
NUM_CLASSES = 38


def build_datasets(data_dir):
    train_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(class_names)} in {data_dir}."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune)
        .cache()
        .prefetch(autotune)
    )
    val_ds = (
        val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune)
        .cache()
        .prefetch(autotune)
    )

    return train_ds, val_ds, class_names


def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def fine_tune_model(model, base_model, learning_rate=1e-5):
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def save_artifacts(model, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, MODEL_FILENAME)
    labels_path = os.path.join(output_dir, LABELS_FILENAME)

    model.save(model_path)
    np.save(labels_path, np.array(class_names))
    print(f"Saved model to {model_path}")
    print(f"Saved class labels to {labels_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease classifier using MobileNetV2.")
    parser.add_argument(
        "data_dir",
        help="Path to the PlantVillage-style dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save the trained model and class labels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds, val_ds, class_names = build_datasets(args.data_dir)

    model, base_model = build_model()
    compile_model(model, learning_rate=1e-3)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        verbose=1,
    )

    fine_tune_model(model, base_model, learning_rate=1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        verbose=1,
    )

    save_artifacts(model, class_names, args.output_dir)


if __name__ == "__main__":
    main()
