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
            // 1. Initial permission request & video stream acquisition targeting rear camera
            let tempStream = null;
            try {
                tempStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { exact: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
            } catch (exactErr) {
                tempStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
            }

            // 2. Enumerate available video input devices
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');

            if (videoDevices.length === 0) {
                if (tempStream) tempStream.getTracks().forEach(track => track.stop());
                this._handleError("No camera devices detected on your device.");
                return false;
            }

            if (videoDevices.length === 1) {
                this.stream = tempStream;
                this.videoElement.srcObject = this.stream;
                await this.videoElement.play();
                return true;
            }

            // 3. Automatically select the back / environment camera device
            let backDevice = videoDevices.find(device => {
                const label = (device.label || '').toLowerCase();
                return label.includes('back') || label.includes('rear') || label.includes('environment') || label.includes('outward') || label.includes('facing back');
            });

            if (!backDevice) {
                // On mobile devices, index 0 is front camera, last index is back camera
                backDevice = videoDevices[videoDevices.length - 1];
            }

            this.selectedDeviceId = backDevice.deviceId;
            console.log(`[CameraController] Found ${videoDevices.length} camera(s). Auto-selecting back camera: "${backDevice.label || backDevice.deviceId}"`);

            // Release temporary stream before starting targeted back camera stream
            tempStream.getTracks().forEach(track => track.stop());

            // 4. Start targeted video stream with selected back camera deviceId
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

            const video = this.videoElement;
            const vw = video.videoWidth || 640;
            const vh = video.videoHeight || 480;

            const renderW = video.clientWidth || vw;
            const renderH = video.clientHeight || vh;

            // Compute scaling factor under CSS object-fit: cover
            const scale = Math.max(renderW / vw, renderH / vh);

            // Square ROI size (65% of viewport width) in intrinsic video pixels
            const roiSize = (renderW * 0.65) / scale;

            // Center of video stream
            const cx = vw / 2;
            const cy = vh / 2;

            // Top-left coordinates of ROI crop box
            const cropX = Math.max(0, cx - (roiSize / 2));
            const cropY = Math.max(0, cy - (roiSize / 2));
            const cropW = Math.min(vw - cropX, roiSize);
            const cropH = Math.min(vh - cropY, roiSize);

            const outputSize = Math.round(Math.min(cropW, cropH));

            canvasElement.width = outputSize;
            canvasElement.height = outputSize;

            const ctx = canvasElement.getContext('2d');
            ctx.drawImage(
                video,
                cropX, cropY, cropW, cropH,
                0, 0, outputSize, outputSize
            );

            canvasElement.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.95);
        });
    }

    /**
     * Checks if torch/flashlight feature is available on active video track.
     * @returns {boolean}
     */
    hasTorchCapability() {
        if (!this.stream) return false;
        const track = this.stream.getVideoTracks()[0];
        if (!track) return false;

        if (typeof track.getCapabilities === 'function') {
            const capabilities = track.getCapabilities();
            return Boolean(capabilities && capabilities.torch);
        }
        return false;
    }

    /**
     * Toggles the hardware flashlight/torch state ON or OFF.
     * @param {boolean} enabled 
     * @returns {Promise<boolean>}
     */
    async setTorch(enabled) {
        if (!this.stream) return false;
        const track = this.stream.getVideoTracks()[0];
        if (!track) return false;

        try {
            await track.applyConstraints({
                advanced: [{ torch: Boolean(enabled) }]
            });
            this.isTorchActive = Boolean(enabled);
            return true;
        } catch (err) {
            console.warn("[CameraController] Failed to toggle flashlight:", err);
            return false;
        }
    }

    /**
     * Stops active video stream tracks.
     */
    stop() {
        this.isTorchActive = false;
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
