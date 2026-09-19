import json
import threading
from pathlib import Path
import numpy as np
import onnxruntime as ort
from django.conf import settings

# Human readable display mapping according to requirements
HUMAN_CLASS_MAPPING = {
    "c1": "Fresh",
    "c2": "Semi Degraded",
    "c3": "Fully Degraded"
}

# Fallback class order if class_names.json is unavailable
DEFAULT_CLASSES = ["c1", "c2", "c3"]


def _softmax(x: np.ndarray) -> np.ndarray:
    """Applies numerically stable softmax along the last axis if input is logits."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LubricantPredictor:
    """
    Singleton ONNX Runtime predictor service for EfficientNet-B3 lubricant condition analysis.
    Loads the ONNX model once during initialization and reuses the session for all inference calls.
    Contains NO TensorFlow or Keras dependencies.
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


            self.session = None
            self.input_name = None
            self.output_name = None
            self.class_names = DEFAULT_CLASSES
            self.model_path = Path(settings.MODEL_PATH)
            self.class_names_path = Path(settings.CLASS_NAMES_PATH)
            self.load_model()
            self._initialized = True

    def load_model(self):
        """Loads class names and initializes the ONNX Runtime InferenceSession."""
        # 1. Load class names JSON
        if self.class_names_path.exists():
            try:
                with open(self.class_names_path, 'r', encoding='utf-8') as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(f"[ONNX LubricantPredictor] Warning: Failed to load class names JSON: {e}")

        # 2. Load ONNX model
        if not self.model_path.exists():
            raise FileNotFoundError(f"[ONNX LubricantPredictor] ONNX model file not found at: {self.model_path}")

        print(f"[ONNX LubricantPredictor] Initializing ONNX Runtime Session from {self.model_path}...")
        
        # Configure ONNX Runtime session options for efficient serverless execution
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts)
        
        # Dynamically obtain ONNX input and output tensor names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[ONNX LubricantPredictor] Model loaded successfully.")
        print(f"  Input Tensor Name:  '{self.input_name}'")
        print(f"  Output Tensor Name: '{self.output_name}'")

    def predict(self, img_batch: np.ndarray) -> dict:
        """
        Executes model inference on preprocessed image batch (1, 300, 300, 3) float32.
        Returns prediction dictionary with class_code, condition, confidence, and degradation_score.
        """
        if self.session is None:
            raise RuntimeError("[ONNX LubricantPredictor] ONNX session is not initialized.")

        # Ensure correct input shape and float32 dtype
        if img_batch.dtype != np.float32:
            img_batch = img_batch.astype(np.float32)

        # Execute ONNX Runtime inference
        raw_outputs = self.session.run([self.output_name], {self.input_name: img_batch})[0]
        probabilities = raw_outputs[0]

        # Check if output is already probabilities (sum ≈ 1.0) or raw logits
        prob_sum = float(np.sum(probabilities))
        if not (0.98 <= prob_sum <= 1.02) or np.any(probabilities < 0.0):
            probabilities = _softmax(probabilities)

        # Probabilities per class (Index 0: c1 Fresh, Index 1: c2 Semi Degraded, Index 2: c3 Fully Degraded)
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
        print(f"[ONNX Predictor] Probabilities -> c1(Fresh): {p_c1:.4f}, c2(Semi): {p_c2:.4f}, c3(Fully): {p_c3:.4f}")
        print(f"[ONNX Predictor] Predicted Class: {class_code} ({condition}), Confidence: {confidence:.1f}%, Degradation Score: {degradation_score}")

        return {
            "class_code": class_code,
            "condition": condition,
            "confidence": round(confidence, 1),
            "degradation_score": degradation_score
        }


def get_predictor() -> LubricantPredictor:
    """Helper function to obtain the singleton predictor instance."""
    return LubricantPredictor()
