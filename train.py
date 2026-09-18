import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

MODEL_DIR = "models"

RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# GPU / CPU CHECK
# ============================================================

print("\n" + "=" * 60)
print("DEVICE CHECK")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")

    for gpu in gpus:
        print(gpu)

else:

    print("No GPU detected.")
    print("Training will use CPU.")


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

print("\n" + "=" * 60)
print("CHECKING DATASET")
print("=" * 60)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError(
        f"Validation directory not found: {VALIDATION_DIR}"
    )


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=RANDOM_SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\n" + "=" * 60)
print("CLASS INFORMATION")
print("=" * 60)

print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(f"{i}: {class_name}")

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# VERIFY CLASSES
# ============================================================

expected_classes = ["c1", "c2", "c3"]

if class_names != expected_classes:

    raise ValueError(
        f"\nExpected classes: {expected_classes}"
        f"\nFound classes: {class_names}"
    )

print("\nClass verification successful.")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(class_names_path, "w") as f:

    json.dump(
        class_names,
        f,
        indent=4
    )

print(
    "\nClass names saved to:",
    class_names_path
)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

print("\nCreating data augmentation pipeline...")

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),

    ],
    name="data_augmentation"
)


# ============================================================
# LOAD EFFICIENTNET-B3
# ============================================================

print("\n" + "=" * 60)
print("LOADING EFFICIENTNET-B3")
print("=" * 60)

base_model = tf.keras.applications.EfficientNetB3(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

print("\nEfficientNet-B3 loaded successfully.")

print(
    "Total EfficientNet layers:",
    len(base_model.layers)
)


# ============================================================
# STAGE 1
# FREEZE BASE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STAGE 1 SETUP")
print("=" * 60)

base_model.trainable = False

print(
    "EfficientNet-B3 backbone: FROZEN"
)


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="input_image"
)


# Data augmentation
x = data_augmentation(inputs)


# EfficientNet-B3
x = base_model(
    x,
    training=False
)


# Global average pooling
x = layers.GlobalAveragePooling2D()(x)


# Dropout
x = layers.Dropout(
    0.30
)(x)


# Classification layer
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classifier"
)(x)


# Create final model
model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="EfficientNetB3_LubVision"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# STAGE 1 COMPILE
# ============================================================

print("\nCompiling Stage 1...")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# STAGE 1 CALLBACKS
# ============================================================

checkpoint_stage1 = ModelCheckpoint(

    filepath=os.path.join(
        MODEL_DIR,
        "best_model_stage1.keras"
    ),

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping_stage1 = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr_stage1 = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

history1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=[

        checkpoint_stage1,

        early_stopping_stage1,

        reduce_lr_stage1

    ]
)


# ============================================================
# STAGE 2
# FINE-TUNING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING EFFICIENTNET-B3")
print("=" * 60)


# Unfreeze EfficientNet
base_model.trainable = True


# ============================================================
# FREEZE EARLY LAYERS
# ============================================================

FINE_TUNE_FROM = len(base_model.layers) - 50


for layer in base_model.layers[:FINE_TUNE_FROM]:

    layer.trainable = False


print(
    "\nFine-tuning from layer:",
    FINE_TUNE_FROM
)


trainable_layers = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)


print(
    "Trainable EfficientNet layers:",
    trainable_layers
)


print(
    "Frozen EfficientNet layers:",
    len(base_model.layers) - trainable_layers
)


# ============================================================
# RECOMPILE FOR FINE-TUNING
# ============================================================

print("\nRecompiling model for fine-tuning...")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# FINE-TUNING CALLBACKS
# ============================================================

checkpoint_fine = ModelCheckpoint(

    filepath=os.path.join(
        MODEL_DIR,
        "best_model_finetuned.keras"
    ),

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping_fine = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr_fine = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-8,

    verbose=1
)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[

        checkpoint_fine,

        early_stopping_fine,

        reduce_lr_fine

    ]
)


# ============================================================
# LOAD BEST FINE-TUNED MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model_finetuned.keras"
)


if os.path.exists(best_model_path):

    model = tf.keras.models.load_model(
        best_model_path
    )

    print(
        "\nBest fine-tuned model loaded successfully."
    )

else:

    print(
        "\nFine-tuned checkpoint not found."
    )

    print(
        "Using current model."
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_b3_final.keras"
)

model.save(
    final_model_path
)

print(
    "\nFinal model saved to:"
)

print(
    final_model_path
)


# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

accuracy = (

    history1.history["accuracy"]

    +

    history2.history["accuracy"]

)


val_accuracy = (

    history1.history["val_accuracy"]

    +

    history2.history["val_accuracy"]

)


loss = (

    history1.history["loss"]

    +

    history2.history["loss"]

)


val_loss = (

    history1.history["val_loss"]

    +

    history2.history["val_loss"]

)


# ============================================================
# ACCURACY GRAPH
# ============================================================

print("\nGenerating accuracy graph...")

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.axvline(

    x=len(
        history1.history["accuracy"]
    ) - 1,

    linestyle="--",

    label="Fine Tuning Starts"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "EfficientNet-B3 Accuracy"
)

plt.legend()

plt.grid()

plt.tight_layout()

accuracy_path = os.path.join(
    MODEL_DIR,
    "accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

print("\nGenerating loss graph...")

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.axvline(

    x=len(
        history1.history["loss"]
    ) - 1,

    linestyle="--",

    label="Fine Tuning Starts"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "EfficientNet-B3 Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()

loss_path = os.path.join(
    MODEL_DIR,
    "loss.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.show()


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


loss_value, accuracy_value = model.evaluate(
    validation_dataset,
    verbose=1
)


print(
    f"\nValidation Loss: {loss_value:.4f}"
)

print(
    f"Validation Accuracy: {accuracy_value:.4f}"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)


y_true = []

y_pred = []


for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(8, 6)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=class_names,

    yticklabels=class_names

)


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.title(
    "EfficientNet-B3 Confusion Matrix"
)

plt.tight_layout()


confusion_matrix_path = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=300
)

plt.show()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    digits=4
)


print("\n")

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    MODEL_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(report)


print(
    "\nClassification report saved to:"
)

print(
    report_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {

    "stage1": history1.history,

    "stage2": history2.history,

    "final_validation_accuracy":
        float(accuracy_value),

    "final_validation_loss":
        float(loss_value)

}


history_path = os.path.join(
    MODEL_DIR,
    "training_history.json"
)


with open(
    history_path,
    "w"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)


print("\nModel:")
print(final_model_path)


print("\nClasses:")
print(class_names)


print("\nValidation Accuracy:")
print(
    f"{accuracy_value:.4f}"
)


print("\nValidation Loss:")
print(
    f"{loss_value:.4f}"
)


print("\nResults saved inside:")
print(
    os.path.abspath(MODEL_DIR)
)


print("\nFiles generated:")

print("  - efficientnet_b3_final.keras")
print("  - best_model_stage1.keras")
print("  - best_model_finetuned.keras")
print("  - class_names.json")
print("  - accuracy.png")
print("  - loss.png")
print("  - confusion_matrix.png")
print("  - classification_report.txt")
print("  - training_history.json")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)