import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"

CLASS_NAMES = ["c1", "c2", "c3"]

# Optimization settings
INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 5e-6

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 25

# IMPORTANT:
# Original model fine-tuned last 50 layers.
# This optimized version fine-tunes last 100 layers.
FINE_TUNE_LAYERS = 100

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )


# ============================================================
# LOAD TRAIN DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

print("\nTraining classes:", train_dataset.class_names)
print("Validation classes:", validation_dataset.class_names)
print("Test classes:", test_dataset.class_names)

if train_dataset.class_names != CLASS_NAMES:
    raise ValueError(
        f"\nTraining class mismatch!"
        f"\nExpected: {CLASS_NAMES}"
        f"\nFound: {train_dataset.class_names}"
    )

if validation_dataset.class_names != CLASS_NAMES:
    raise ValueError(
        f"\nValidation class mismatch!"
        f"\nExpected: {CLASS_NAMES}"
        f"\nFound: {validation_dataset.class_names}"
    )

if test_dataset.class_names != CLASS_NAMES:
    raise ValueError(
        f"\nTest class mismatch!"
        f"\nExpected: {CLASS_NAMES}"
        f"\nFound: {test_dataset.class_names}"
    )

print("\nClass verification successful.")


# ============================================================
# DATASET PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.10
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD EFFICIENTNET-B3
# ============================================================

print("\n" + "=" * 70)
print("LOADING EFFICIENTNET-B3")
print("=" * 70)

base_model = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

print(
    "\nTotal EfficientNet-B3 layers:",
    len(base_model.layers)
)


# ============================================================
# STAGE 1
# FREEZE ENTIRE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS - STAGE 1
# ============================================================

stage1_best_model = os.path.join(
    MODEL_DIR,
    "best_model_stage1.keras"
)

stage1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    stage1_best_model,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

stage1_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

stage1_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# COMPILE STAGE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 TRAINING")
print("=" * 70)

history_stage1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        stage1_checkpoint,
        stage1_early_stopping,
        stage1_reduce_lr
    ],
    verbose=1
)


# ============================================================
# LOAD BEST STAGE 1 MODEL
# ============================================================

if os.path.exists(stage1_best_model):

    model = tf.keras.models.load_model(
        stage1_best_model
    )

    print(
        "\nBest Stage 1 model loaded successfully."
    )


# ============================================================
# STAGE 2
# FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 FINE-TUNING")
print("=" * 70)

base_model = model.get_layer(
    "efficientnetb3"
)

base_model.trainable = True


# ============================================================
# FREEZE ALL BUT LAST 100 LAYERS
# ============================================================

total_layers = len(base_model.layers)

freeze_until = max(
    0,
    total_layers - FINE_TUNE_LAYERS
)

print(
    "\nTotal base-model layers:",
    total_layers
)

print(
    "Fine-tuning last:",
    FINE_TUNE_LAYERS,
    "layers"
)

print(
    "Freezing first:",
    freeze_until,
    "layers"
)


for layer in base_model.layers:

    layer.trainable = False


for layer in base_model.layers[freeze_until:]:

    layer.trainable = True


# ============================================================
# IMPORTANT:
# KEEP BATCH NORMALIZATION LAYERS FROZEN
# ============================================================

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_layers = sum(
    1
    for layer in model.layers
    if layer.trainable
)

print(
    "\nTrainable model layers:",
    trainable_layers
)


# ============================================================
# COMPILE STAGE 2
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS - STAGE 2
# ============================================================

best_finetuned_model = os.path.join(
    MODEL_DIR,
    "best_model_optimized.keras"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_finetuned_model,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history_stage2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# LOAD BEST FINE-TUNED MODEL
# ============================================================

if os.path.exists(best_finetuned_model):

    model = tf.keras.models.load_model(
        best_finetuned_model
    )

    print(
        "\nBest optimized model loaded successfully."
    )


# ============================================================
# SAVE FINAL OPTIMIZED MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_b3_optimized.keras"
)

model.save(
    final_model_path
)

print(
    "\nOptimized model saved to:",
    final_model_path
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(
    class_names_path,
    "w"
) as f:

    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )

print(
    "Class names saved to:",
    class_names_path
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print(
    f"\nValidation Loss: {validation_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []

for images, labels in test_dataset:

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


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(
        len(CLASS_NAMES)
    )
)

print("\n" + "=" * 70)
print("TEST CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE TEST CONFUSION MATRIX
# ============================================================

confusion_matrix_path = os.path.join(
    MODEL_DIR,
    "optimized_test_confusion_matrix.png"
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.title(
    "Optimized EfficientNet-B3 Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    confusion_matrix_path,
    dpi=300
)

plt.close()

print(
    "\nConfusion matrix saved to:",
    confusion_matrix_path
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print("\n" + "=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    MODEL_DIR,
    "optimized_test_classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(report)

print(
    "\nClassification report saved to:",
    report_path
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)

print(
    "\nModel:",
    "EfficientNet-B3"
)

print(
    "Image size:",
    IMAGE_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Fine-tuned layers:",
    FINE_TUNE_LAYERS
)

print(
    "Fine-tuning learning rate:",
    FINE_TUNE_LEARNING_RATE
)

print(
    f"\nValidation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nPrevious test accuracy: 74.16%"
)

print(
    f"Change: "
    f"{(test_accuracy * 100) - 74.16:+.2f} percentage points"
)

print("\n" + "=" * 70)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 70)