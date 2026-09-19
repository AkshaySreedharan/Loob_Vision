/**
 * LUBVISION Main Application Script
 * Implements CAMERA FIRST -> CAPTURE / UPLOAD -> PREVIEW -> ANALYZE workflow.
 * Integrates with Django ONNX Predictor API (/api/predict/).
 */
document.addEventListener('DOMContentLoaded', () => {

    // DOM Elements
    const splashScreen = document.getElementById('splash-screen');
    const viewTitleText = document.getElementById('view-title-text');
    const previewContainer = document.getElementById('preview-container');
    const cameraVideo = document.getElementById('camera-video');
    const imagePreview = document.getElementById('image-preview');
    const captureCanvas = document.getElementById('capture-canvas');
    const cameraHud = document.getElementById('camera-hud');
    const cameraError = document.getElementById('camera-error');
    const cameraErrorText = document.getElementById('camera-error-text');
    const processingOverlay = document.getElementById('processing-overlay');
    
    // Control Bars
    const liveControls = document.getElementById('live-controls');
    const captureControls = document.getElementById('capture-controls');

    // Buttons
    const btnMobileScan = document.getElementById('btn-mobile-scan');
    const btnCapture = document.getElementById('btn-capture');
    const btnRetake = document.getElementById('btn-retake');
    const btnAnalyze = document.getElementById('btn-analyze');
    const fileInput = document.getElementById('file-input');
    
    // Results DOM Elements
    const resultsSection = document.getElementById('results-section');
    const conditionBadge = document.getElementById('condition-badge');
    const conditionName = document.getElementById('condition-name');
    const confidenceValue = document.getElementById('confidence-value');
    const btnReset = document.getElementById('btn-reset');

    // State Variables
    let currentCameraActive = false;
    let capturedBlob = null;
    let currentPreviewUrl = null;
    let currentFilename = "lubricant_sample.jpg";

    // 1. Splash Screen Dismissal
    setTimeout(() => {
        if (splashScreen) {
            splashScreen.classList.add('fade-out');
        }
    }, 1200);

    // 2. Camera Initialization (Automatic first camera selection)
    async function initCamera() {
        if (!cameraVideo) return;

        cameraError.classList.add('hidden');
        cameraHud.classList.remove('hidden');

        const success = await window.cameraController.start(cameraVideo, (errorMsg) => {
            showCameraError(errorMsg);
        });

        currentCameraActive = success;
        if (success) {
            cameraVideo.classList.remove('hidden');
        }
    }

    function showCameraError(message) {
        currentCameraActive = false;
        cameraHud.classList.add('hidden');
        if (cameraVideo) cameraVideo.classList.add('hidden');
        cameraErrorText.textContent = message || "Camera is currently unavailable.";
        cameraError.classList.remove('hidden');
    }

    // Auto-start camera when page loads
    initCamera();

    // 3. CAPTURE Button Handler (Freezes camera frame, shows preview, waits for ANALYZE)
    btnCapture.addEventListener('click', async () => {
        if (!currentCameraActive) {
            alert("Camera is not active. Please use the 'Upload Image' button to select an image from your device.");
            return;
        }

        const blob = await window.cameraController.captureBlob(captureCanvas);
        if (!blob) {
            alert("Failed to capture frame from camera. Please try again.");
            return;
        }

        capturedBlob = blob;
        currentFilename = "captured_lubricant_sample.jpg";

        showImagePreview(capturedBlob, "CAPTURE PREVIEW");
    });

    // 4. Secondary Upload Button Handler (File Picker -> Preview -> Waits for ANALYZE)
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        capturedBlob = file;
        currentFilename = file.name || "uploaded_sample.jpg";

        showImagePreview(capturedBlob, "UPLOADED IMAGE PREVIEW");
        
        // Reset file input value so re-selecting the same file fires change event
        fileInput.value = '';
    });

    // Helper: Show captured/uploaded image preview in container
    function showImagePreview(blobOrFile, titleText) {
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
        }
        currentPreviewUrl = URL.createObjectURL(blobOrFile);

        imagePreview.src = currentPreviewUrl;
        imagePreview.classList.remove('hidden');
        cameraVideo.classList.add('hidden');
        cameraHud.classList.add('hidden');
        if (previewContainer) previewContainer.classList.add('captured-mode');

        viewTitleText.textContent = titleText;
        liveControls.classList.add('hidden');
        captureControls.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    }

    // 5. RETAKE Button Handler (Discards current preview, returns to live camera)
    btnRetake.addEventListener('click', () => {
        resetToLiveCamera();
    });

    // 6. ANALYZE IMAGE Button Handler (Sends preview image to Django prediction API)
    btnAnalyze.addEventListener('click', async () => {
        if (!capturedBlob) {
            alert("No captured or uploaded image available for analysis.");
            return;
        }

        await sendImageForPrediction(capturedBlob, currentFilename);
    });

    // 7. Prediction API Communication
    async function sendImageForPrediction(imageBlobOrFile, filename) {
        showProcessing(true);

        const formData = new FormData();
        formData.append('image', imageBlobOrFile, filename);

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
                alert(`Analysis Error: ${data.error || 'Failed to analyze sample.'}`);
            }

        } catch (err) {
            showProcessing(false);
            alert("Network error: Unable to connect to LUBVISION prediction server.");
        }
    }

    function showProcessing(show) {
        if (show) {
            processingOverlay.classList.remove('hidden');
        } else {
            processingOverlay.classList.add('hidden');
        }
    }

    // 8. Render Prediction Result & Degradation Score Bar
    function renderResult(condition, confidence, degradationScore) {
        resultsSection.classList.remove('hidden');

        conditionName.textContent = condition || "Unknown";
        confidenceValue.textContent = `${Number(confidence || 0).toFixed(1)}%`;

        // Style status badge
        conditionBadge.className = 'condition-badge';
        const condLower = String(condition || '').toLowerCase();
        if (condLower.includes('fresh')) {
            conditionBadge.classList.add('fresh');
        } else if (condLower.includes('semi')) {
            conditionBadge.classList.add('semi');
        } else {
            conditionBadge.classList.add('fully');
        }

        // Clamp degradation score to range [0.0, 100.0]
        const score = Math.max(0.0, Math.min(100.0, Number(degradationScore || 0.0)));

        // Update score display text
        const degradationScoreNum = document.getElementById('degradation-score-num');
        if (degradationScoreNum) {
            degradationScoreNum.textContent = score.toFixed(1);
        }

        // Update visual progress bar width & level indicator
        const fill = document.getElementById('degradation-bar-fill');
        if (fill) {
            fill.style.width = `${score}%`;
            fill.className = 'degradation-bar-fill';
            if (score <= 33.0) {
                fill.classList.add('level-low');
            } else if (score <= 66.0) {
                fill.classList.add('level-moderate');
            } else {
                fill.classList.add('level-high');
            }
        }

        // Update interpretation level text
        const degradationLevelText = document.getElementById('degradation-level-text');
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

        if (window.innerWidth <= 768) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // 9. NEW ANALYSIS Handler
    btnReset.addEventListener('click', () => {
        resetToLiveCamera();
    });

    function resetToLiveCamera() {
        capturedBlob = null;
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
            currentPreviewUrl = null;
        }

        imagePreview.classList.add('hidden');
        resultsSection.classList.add('hidden');
        if (previewContainer) previewContainer.classList.remove('captured-mode');

        // Reset degradation score bar and numbers
        const fill = document.getElementById('degradation-bar-fill');
        if (fill) fill.style.width = "0%";
        const degradationScoreNum = document.getElementById('degradation-score-num');
        if (degradationScoreNum) degradationScoreNum.textContent = "0.0";
        const degradationLevelText = document.getElementById('degradation-level-text');
        if (degradationLevelText) {
            degradationLevelText.textContent = "Low Degradation";
            degradationLevelText.className = "degradation-level low";
        }

        viewTitleText.textContent = "LIVE CAMERA PREVIEW";
        captureControls.classList.add('hidden');
        liveControls.classList.remove('hidden');

        if (currentCameraActive) {
            cameraVideo.classList.remove('hidden');
            cameraHud.classList.remove('hidden');
        } else {
            initCamera();
        }
    }

    // 10. Mobile QR Code Modal Handler
    const qrModal = document.getElementById('qr-modal');
    const btnQrClose = document.getElementById('btn-qr-close');
    const qrCodeBox = document.getElementById('qr-code-box');
    const qrUrlInput = document.getElementById('qr-url-input');
    const btnCopyUrl = document.getElementById('btn-copy-url');

    if (btnMobileScan) {
        btnMobileScan.addEventListener('click', (e) => {
            e.preventDefault();
            openQrModal();
        });
    }

    if (btnQrClose) {
        btnQrClose.addEventListener('click', () => {
            closeQrModal();
        });
    }

    if (qrModal) {
        qrModal.addEventListener('click', (e) => {
            if (e.target === qrModal) closeQrModal();
        });
    }

    async function openQrModal() {
        if (!qrModal) return;
        qrModal.classList.remove('hidden');

        try {
            const resp = await fetch('/api/mobile-url/');
            const data = await resp.json();
            const targetUrl = data.mobile_url || window.location.href + 'mobile-upload/';
            qrUrlInput.value = targetUrl;

            const qrResp = await fetch(`/api/qr/?url=${encodeURIComponent(targetUrl)}`);
            if (qrResp.ok) {
                const svgText = await qrResp.text();
                qrCodeBox.innerHTML = svgText;
            } else {
                qrCodeBox.innerHTML = `<span class="qr-error-text">Failed to load QR code.</span>`;
            }
        } catch (e) {
            qrCodeBox.innerHTML = `<span class="qr-error-text">Unable to connect for QR generation.</span>`;
        }
    }

    function closeQrModal() {
        if (qrModal) qrModal.classList.add('hidden');
    }

    if (btnCopyUrl) {
        btnCopyUrl.addEventListener('click', () => {
            if (!qrUrlInput.value) return;
            navigator.clipboard.writeText(qrUrlInput.value).then(() => {
                const origText = btnCopyUrl.innerHTML;
                btnCopyUrl.innerHTML = "Copied!";
                setTimeout(() => { btnCopyUrl.innerHTML = origText; }, 2000);
            });
        });
    }
});
