/**
 * Modular Camera Controller Class
 * Encapsulates browser MediaDevices / getUserMedia API.
 * Keeps camera handling decoupled from the prediction & UI logic.
 */
class CameraController {
    constructor() {
        this.stream = null;
        this.videoElement = null;
        this.onErrorCallback = null;
    }

    /**
     * Initializes and starts the camera stream on the target video element.
     * @param {HTMLVideoElement} videoElement 
     * @param {Function} onErrorCallback 
     * @returns {Promise<boolean>}
     */
    async start(videoElement, onErrorCallback = null) {
        this.videoElement = videoElement;
        this.onErrorCallback = onErrorCallback;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this._handleError("Camera API is not supported by your browser or secure context (HTTPS/localhost required).");
            return false;
        }

        try {
            const constraints = {
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();
            return true;
        } catch (err) {
            let errorMsg = "Unable to access camera.";

            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                errorMsg = "Camera access permission was denied. Please allow camera access in your browser settings.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                errorMsg = "No camera hardware detected on your device.";
            } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                errorMsg = "Camera is currently in use by another application.";
            } else {
                errorMsg = `Camera error: ${err.message || 'Unknown camera issue.'}`;
            }

            this._handleError(errorMsg);
            return false;
        }
    }

    /**
     * Captures current frame from video stream to a PNG/JPEG Blob.
     * @param {HTMLCanvasElement} canvasElement 
     * @returns {Promise<Blob|null>}
     */
    captureBlob(canvasElement) {
        return new Promise((resolve) => {
            if (!this.videoElement || !this.stream) {
                resolve(null);
                return;
            }

            const width = this.videoElement.videoWidth || 640;
            const height = this.videoElement.videoHeight || 480;

            canvasElement.width = width;
            canvasElement.height = height;

            const ctx = canvasElement.getContext('2d');
            ctx.drawImage(this.videoElement, 0, 0, width, height);

            canvasElement.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.92);
        });
    }

    /**
     * Stops the active video stream tracks.
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
    }

    _handleError(message) {
        console.warn("[CameraController]", message);
        if (typeof this.onErrorCallback === 'function') {
            this.onErrorCallback(message);
        }
    }
}

// Global camera instance
window.cameraController = new CameraController();
