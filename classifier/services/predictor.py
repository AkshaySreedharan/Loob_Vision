import json
import threading
from pathlib import Path
import numpy as np
from django.conf import settings

# Human readable display mapping according to requirements
HUMAN_CLASS_MAPPING = {
    "c1": "Fresh",
    "c2": "Semi Degraded",
    "c3": "Fully Degraded"
}

# Fallback class order if class_names.json is unavailable
DEFAULT_CLASSES = ["c1", "c2", "c3"]


class LubricantPredictor:
    """
    Singleton predictor service for EfficientNet-B3 model.
    Loads the Keras model once at Django startup/predictor initialization.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LubricantPredictor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.model = None
            self.class_names = DEFAULT_CLASSES
            self.model_path = Path(settings.MODEL_PATH)
            self.class_names_path = Path(settings.CLASS_NAMES_PATH)
            self.load_model()
            self._initialized = True

    def _build_model_architecture(self):
        """Reconstructs exact EfficientNet-B3 model architecture as defined in train_optimized.py."""
        import tensorflow as tf

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ], name="data_augmentation")

        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights=None,
            input_shape=(300, 300, 3)
        )

        inputs = tf.keras.Input(shape=(300, 300, 3))
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.30)(x)
        outputs = tf.keras.layers.Dense(len(self.class_names), activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        return model

    def load_model(self):
        """Loads class names and Keras v3 model weights into memory."""
        import tensorflow as tf

        # 1. Load class names
        if self.class_names_path.exists():
            try:
                with open(self.class_names_path, 'r') as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(f"[LubricantPredictor] Warning: Failed to load class names JSON: {e}")

        # 2. Load model / weights
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        print(f"[LubricantPredictor] Loading model from {self.model_path}...")
        try:
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            print("[LubricantPredictor] Model loaded directly via load_model.")
        except Exception as err:
            print(f"[LubricantPredictor] Direct load_model notice ({err}). Reconstructing architecture & loading weights...")
            self.model = self._build_model_architecture()
            self.model.load_weights(str(self.model_path))
            print("[LubricantPredictor] Trained model weights loaded successfully into architecture.")


    def predict(self, img_batch: np.ndarray):
        """
        Executes model inference on preprocessed image batch (1, 300, 300, 3).
        Returns prediction dictionary.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        predictions = self.model.predict(img_batch, verbose=0)
        probabilities = predictions[0]

        # Existing prediction probabilities (Index 0: c1 Fresh, Index 1: c2 Semi, Index 2: c3 Fully)
        p_c1 = float(probabilities[0]) if len(probabilities) > 0 else 0.0
        p_c2 = float(probabilities[1]) if len(probabilities) > 1 else 0.0
        p_c3 = float(probabilities[2]) if len(probabilities) > 2 else 0.0

        # Calculate degradation score: (P(c2) * 50) + (P(c3) * 100)
        raw_degradation_score = (p_c2 * 50.0) + (p_c3 * 100.0)
        clamped_degradation_score = max(0.0, min(100.0, raw_degradation_score))
        degradation_score = round(clamped_degradation_score, 1)

        top_index = int(np.argmax(probabilities))
        class_code = self.class_names[top_index] if top_index < len(self.class_names) else f"c{top_index+1}"
        condition = HUMAN_CLASS_MAPPING.get(class_code, class_code)
        confidence = float(probabilities[top_index]) * 100.0

        # Debug logging
        print(f"[DEBUG Predictor] Probabilities -> c1(Fresh): {p_c1:.4f}, c2(Semi): {p_c2:.4f}, c3(Fully): {p_c3:.4f}")
        print(f"[DEBUG Predictor] Calculated Degradation Score: {degradation_score}")

        return {
            "class_code": class_code,
            "condition": condition,
            "confidence": round(confidence, 1),
            "degradation_score": degradation_score
        }


def get_predictor():
    """Helper function to obtain the singleton predictor instance."""
    return LubricantPredictor()
