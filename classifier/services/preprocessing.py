import io
import numpy as np
from PIL import Image

# Allowed MIME types / formats
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP'}
IMAGE_SIZE = (300, 300)


def preprocess_image(image_bytes: bytes):
    """
    Safely decodes, validates, and preprocesses input image bytes for EfficientNet-B3 inference.
    Returns (preprocessed_array, None) on success or (None, error_string) on failure.
    """
    if not image_bytes:
        return None, "No image data received."

    try:
        # Open image using Pillow
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream)

        # Verify format
        if img.format and img.format.upper() not in ALLOWED_FORMATS:
            return None, f"Unsupported image format: {img.format}. Please upload a JPEG, PNG, or WEBP image."

        # Ensure image is RGB (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image to target input size (300, 300)
        resample_filter = getattr(Image, 'Resampling', Image).BILINEAR
        img_resized = img.resize(IMAGE_SIZE, resample_filter)

        # Convert to float32 numpy array with values in [0.0, 255.0]
        img_array = np.array(img_resized, dtype=np.float32)

        # Expand batch dimension to shape (1, 300, 300, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        return img_batch, None

    except Exception as e:
        return None, "Corrupted or invalid image file. Please provide a valid image."
