/**
 * LUBVISION Camera Controller
 * Encapsulates browser MediaDevices / getUserMedia / enumerateDevices API.
 * Automatically detects and selects the first available video camera input device.
 */
class CameraController {
    constructor() {
        this.stream = null;
        this.videoElement = null;
        this.selectedDeviceId = null;
        this.onErrorCallback = null;
    }

    /**
     * Initializes camera permissions, enumerates available devices, selects the first camera,
     * and starts the live video stream.
     * @param {HTMLVideoElement} videoElement 
     * @param {Function} onErrorCallback 
     * @returns {Promise<boolean>}
     */
    async start(videoElement, onErrorCallback = null) {
        this.videoElement = videoElement;
        this.onErrorCallback = onErrorCallback;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this._handleError("Camera API is not supported by your browser or current context (HTTPS or localhost required).");
            return false;
        }

        try {
            // 1. Initial permission request & video stream acquisition
            let initialConstraints = {
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            let tempStream = await navigator.mediaDevices.getUserMedia(initialConstraints);

            // 2. Enumerate available video input devices
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');

            // Release temporary stream before starting targeted device stream if needed
            tempStream.getTracks().forEach(track => track.stop());

            if (videoDevices.length === 0) {
                this._handleError("No camera devices detected on your device.");
                return false;
            }

            // 3. Automatically select the first detected video device
            const firstDevice = videoDevices[0];
            this.selectedDeviceId = firstDevice.deviceId;

            console.log(`[CameraController] Found ${videoDevices.length} camera(s). Auto-selecting first camera: "${firstDevice.label || firstDevice.deviceId}"`);

            // 4. Start targeted video stream with selected deviceId
            const finalConstraints = {
                video: {
                    deviceId: { exact: this.selectedDeviceId },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(finalConstraints);
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();

            return true;

        } catch (err) {
            console.warn("[CameraController] Targeted camera start failed, attempting fallback:", err);

            // Fallback: Attempt general video getUserMedia if exact deviceId failed
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                this.videoElement.srcObject = this.stream;
                await this.videoElement.play();
                return true;
            } catch (fallbackErr) {
                let errorMsg = "Camera unavailable.";

                if (fallbackErr.name === "NotAllowedError" || fallbackErr.name === "PermissionDeniedError") {
                    errorMsg = "Camera access permission was denied. Please allow camera access in browser settings.";
                } else if (fallbackErr.name === "NotFoundError" || fallbackErr.name === "DevicesNotFoundError") {
                    errorMsg = "No camera hardware detected on your device.";
                } else if (fallbackErr.name === "NotReadableError" || fallbackErr.name === "TrackStartError") {
                    errorMsg = "Camera is currently in use by another application.";
                } else {
                    errorMsg = `Camera unavailable: ${fallbackErr.message || 'Unable to start camera stream.'}`;
                }

                this._handleError(errorMsg);
                return false;
            }
        }
    }

    /**
     * Captures current frame from video stream to a JPEG Blob.
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
     * Stops active video stream tracks.
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (e) {}
            });
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
