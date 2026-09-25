/**
 * LUBVISION Mobile Camera Application Script
 * Enforces strict Camera Lifecycle Management:
 * REAR CAMERA STREAM -> CAPTURE FRAME -> STOP CAMERA -> PREVIEW -> [ RETAKE / ANALYZE ]
 * Reuses existing Django POST /api/predict/ endpoint.
 */
document.addEventListener('DOMContentLoaded', () => {

    const mobileVideo = document.getElementById('mobile-video');
    const mobileImagePreview = document.getElementById('mobile-image-preview');
    const mobileCanvas = document.getElementById('mobile-canvas');
    const mobileHud = document.getElementById('mobile-hud');
    const processingOverlay = document.getElementById('mobile-processing-overlay');

    const liveControls = document.getElementById('mobile-live-controls');
    const captureControls = document.getElementById('mobile-capture-controls');
    const btnTakePhoto = document.getElementById('btn-mobile-take-photo');
    const btnRetake = document.getElementById('btn-mobile-retake');
    const btnAnalyze = document.getElementById('btn-mobile-analyze');

    const resultsSection = document.getElementById('mobile-results-section');
    const conditionBadge = document.getElementById('mobile-condition-badge');
    const conditionName = document.getElementById('mobile-condition-name');
    const confidenceValue = document.getElementById('mobile-confidence-value');
    const degradationScoreNum = document.getElementById('mobile-degradation-score-num');
    const degradationBarFill = document.getElementById('mobile-degradation-bar-fill');
    const degradationLevelText = document.getElementById('mobile-degradation-level-text');
    const btnReset = document.getElementById('btn-mobile-reset');

    const btnTorch = document.getElementById('btn-mobile-torch');
    const torchStatusText = document.getElementById('mobile-torch-status');

    // Camera Stream & Capture State
    let cameraStream = null;
    let capturedBlob = null;
    let currentPreviewUrl = null;
    let isTorchOn = false;

    /**
     * Dedicated Function to Stop Camera Stream Completely & Reset Video Element
     */
    function stopCamera() {
        setTorchState(false);
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (e) {
                    console.warn('[LUBVISION Mobile] Error stopping camera track:', e);
                }
            });
            cameraStream = null;
        }

        if (mobileVideo) {
            try {
                mobileVideo.pause();
            } catch (e) {}
            mobileVideo.srcObject = null;
        }
        if (btnTorch) btnTorch.classList.add('hidden');
    }

    /**
     * Dedicated Async Function to Start Camera Stream cleanly
     */
    async function startCamera() {
        // 1. Always stop any existing stream first
        stopCamera();

        // 2. Hardware release settling delay to prevent mobile OS lockups
        await new Promise(resolve => setTimeout(resolve, 200));

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const errorMsg = "MediaDevices getUserMedia API is not supported by your browser or secure context (HTTPS required).";
            console.error("Camera error:", new Error(errorMsg));
            alert(errorMsg);
            return false;
        }

        try {
            // 3. Request camera with rear environment preference
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

            // 4. Enumerate devices to identify explicit back/rear camera hardware
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');

            if (videoDevices.length <= 1) {
                cameraStream = tempStream;
            } else {
                let backCamera = videoDevices.find(d => {
                    const label = (d.label || '').toLowerCase();
                    return label.includes('back') || label.includes('rear') || label.includes('environment') || label.includes('outward') || label.includes('facing back');
                });

                if (!backCamera) {
                    // On iOS/Android, index 0 is front/selfie camera, last index is back camera
                    backCamera = videoDevices[videoDevices.length - 1];
                }

                if (backCamera && backCamera.deviceId) {
                    tempStream.getTracks().forEach(track => track.stop());
                    cameraStream = await navigator.mediaDevices.getUserMedia({
                        video: { deviceId: { exact: backCamera.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
                        audio: false
                    });
                } else {
                    cameraStream = tempStream;
                }
            }

            if (mobileVideo) {
                mobileVideo.srcObject = cameraStream;
                await mobileVideo.play();
            }

            // Initialize Flashlight / Torch Control
            initTorchControl();
            return true;

        } catch (error) {
            handleCameraError(error);
            return false;
        }
    }

    /**
     * Initializes Flashlight / Torch button visibility based on camera hardware capabilities
     */
    function initTorchControl() {
        if (!btnTorch) return;
        isTorchOn = false;
        updateTorchUI(false);

        if (!cameraStream) {
            btnTorch.classList.add('hidden');
            return;
        }

        const track = cameraStream.getVideoTracks()[0];
        if (!track) {
            btnTorch.classList.add('hidden');
            return;
        }

        // Check if track supports capabilities and torch
        const capabilities = (typeof track.getCapabilities === 'function') ? track.getCapabilities() : {};
        if (capabilities && capabilities.torch) {
            btnTorch.classList.remove('hidden');
        } else if (capabilities && 'torch' in capabilities) {
            btnTorch.classList.remove('hidden');
        } else {
            // Show torch button for rear cameras on mobile as fallback attempt
            btnTorch.classList.remove('hidden');
        }
    }

    /**
     * Toggles hardware torch ON or OFF
     */
    async function toggleTorch() {
        if (!cameraStream) return;
        const targetState = !isTorchOn;
        const success = await setTorchState(targetState);
        if (success) {
            isTorchOn = targetState;
            updateTorchUI(isTorchOn);
        } else {
            console.warn('[LUBVISION Mobile] Flashlight toggle rejected by hardware.');
        }
    }

    async function setTorchState(enable) {
        if (!cameraStream) return false;
        const track = cameraStream.getVideoTracks()[0];
        if (!track) return false;

        try {
            await track.applyConstraints({
                advanced: [{ torch: Boolean(enable) }]
            });
            return true;
        } catch (err) {
            return false;
        }
    }

    function updateTorchUI(active) {
        if (!btnTorch) return;
        if (active) {
            btnTorch.classList.add('active');
            if (torchStatusText) torchStatusText.textContent = "FLASH ON";
        } else {
            btnTorch.classList.remove('active');
            if (torchStatusText) torchStatusText.textContent = "FLASH OFF";
        }
    }

    if (btnTorch) {
        btnTorch.addEventListener('click', (e) => {
            e.preventDefault();
            toggleTorch();
        });
    }

    /**
     * Explicit Camera Error Logging & Categorization
     */
    function handleCameraError(error) {
        console.error("Camera error:", error);

        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            console.warn('[LUBVISION Mobile] Camera access permission denied by user.');
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            console.warn('[LUBVISION Mobile] No camera device hardware found.');
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            console.warn('[LUBVISION Mobile] Camera hardware in use by another app or locked.');
        } else if (error.name === 'AbortError') {
            console.warn('[LUBVISION Mobile] Camera stream request aborted.');
        } else if (error.name === 'SecurityError') {
            console.warn('[LUBVISION Mobile] Camera security restriction.');
        }
    }

    /**
     * Helper to Capture One Frame from Video Stream to JPEG Blob
     */
    function captureFrameBlob() {
        return new Promise((resolve) => {
            if (!mobileVideo || !cameraStream || mobileVideo.readyState < 2) {
                resolve(null);
                return;
            }

            const width = mobileVideo.videoWidth || 640;
            const height = mobileVideo.videoHeight || 480;

            mobileCanvas.width = width;
            mobileCanvas.height = height;

            const ctx = mobileCanvas.getContext('2d');
            ctx.drawImage(mobileVideo, 0, 0, width, height);

            mobileCanvas.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.92);
        });
    }

    // Initialize Camera Stream on load
    startCamera();

    // 1. CAPTURE Handler (Captures 1 frame, stops camera completely, shows preview)
    btnTakePhoto.addEventListener('click', async () => {
        const blob = await captureFrameBlob();
        if (blob) {
            setCapturedState(blob);
        } else {
            console.warn('[LUBVISION Mobile] Frame capture yielded null, attempting camera restart.');
            await startCamera();
        }
    });

    function setCapturedState(blob) {
        capturedBlob = blob;

        // Stop camera hardware stream completely
        stopCamera();

        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
        }
        currentPreviewUrl = URL.createObjectURL(blob);

        mobileImagePreview.src = currentPreviewUrl;
        mobileImagePreview.classList.remove('hidden');
        if (mobileVideo) mobileVideo.classList.add('hidden');
        if (mobileHud) mobileHud.classList.add('hidden');

        const titleText = document.getElementById('mobile-title-text');
        if (titleText) titleText.textContent = "PREVIEW PHOTO";

        liveControls.classList.add('hidden');
        captureControls.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    }

    // 2. RETAKE Handler (Clears preview, stops old stream, waits, restarts camera)
    btnRetake.addEventListener('click', async () => {
        await resetToLiveCamera();
    });

    // 3. ANALYZE Handler (Sends captured Blob to existing Django POST /api/predict/)
    btnAnalyze.addEventListener('click', async () => {
        if (!capturedBlob) {
            alert("No photo captured for analysis.");
            return;
        }

        showProcessing(true);

        const formData = new FormData();
        formData.append('image', capturedBlob, 'mobile_sample.jpg');

        try {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': typeof CSRF_TOKEN !== 'undefined' ? CSRF_TOKEN : ''
                },
                body: formData
            });

            const data = await response.json();
            showProcessing(false);

            if (data.success) {
                renderResult(data.condition, data.confidence, data.degradation_score);
            } else {
                alert(`Analysis Error: ${data.error || 'Failed to process lubricant sample.'}`);
            }
        } catch (err) {
            showProcessing(false);
            console.error('[LUBVISION Mobile] Network error sending image:', err);
            alert("Network error: Unable to connect to LUBVISION prediction server.");
        }
    });

    function showProcessing(show) {
        if (show) {
            processingOverlay.classList.remove('hidden');
        } else {
            processingOverlay.classList.add('hidden');
        }
    }

    // 4. Render Prediction Result
    function renderResult(condition, confidence, degradationScore) {
        resultsSection.classList.remove('hidden');

        conditionName.textContent = condition;
        confidenceValue.textContent = `${confidence.toFixed(1)}%`;

        conditionBadge.className = 'condition-badge';
        const condLower = condition.toLowerCase();
        if (condLower.includes('fresh')) {
            conditionBadge.classList.add('fresh');
        } else if (condLower.includes('semi')) {
            conditionBadge.classList.add('semi');
        } else {
            conditionBadge.classList.add('fully');
        }

        const score = Math.max(0, Math.min(100, Number(degradationScore)));
        if (degradationScoreNum) {
            degradationScoreNum.textContent = score.toFixed(1);
        }

        if (degradationBarFill) {
            degradationBarFill.style.width = `${score}%`;
            degradationBarFill.className = 'degradation-bar-fill';
            if (score <= 33.0) {
                degradationBarFill.classList.add('level-low');
            } else if (score <= 66.0) {
                degradationBarFill.classList.add('level-moderate');
            } else {
                degradationBarFill.classList.add('level-high');
            }
        }

        if (degradationLevelText) {
            degradationLevelText.className = 'degradation-level';
            if (score <= 33.0) {
                degradationLevelText.textContent = "Low Degradation";
                degradationLevelText.classList.add('low');
            } else if (score <= 66.0) {
                degradationLevelText.textContent = "Moderate Degradation";
                degradationLevelText.classList.add('moderate');
            } else {
                degradationLevelText.textContent = "High Degradation";
                degradationLevelText.classList.add('high');
            }
        }

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    // 5. Reset / New Sample Handler
    if (btnReset) {
        btnReset.addEventListener('click', async () => {
            await resetToLiveCamera();
        });
    }

    async function resetToLiveCamera() {
        capturedBlob = null;
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
            currentPreviewUrl = null;
        }

        mobileImagePreview.classList.add('hidden');
        resultsSection.classList.add('hidden');

        const titleText = document.getElementById('mobile-title-text');
        if (titleText) titleText.textContent = "TAKE LUBRICANT PHOTO";

        captureControls.classList.add('hidden');
        liveControls.classList.remove('hidden');

        if (mobileVideo) mobileVideo.classList.remove('hidden');
        if (mobileHud) mobileHud.classList.remove('hidden');

        // Restart camera cleanly
        await startCamera();
    }

    // 6. Page Lifecycle Event Listeners (Release camera on hide/unload)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopCamera();
        } else if (!capturedBlob && liveControls && !liveControls.classList.contains('hidden')) {
            startCamera();
        }
    });

    window.addEventListener('pagehide', () => {
        stopCamera();
    });

    window.addEventListener('unload', () => {
        stopCamera();
    });
});
