/**
 * LUBVISION Main Application Script
 * Implements CAPTURE -> PREVIEW -> ANALYZE workflow.
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
    
    // Results DOM
    const resultsSection = document.getElementById('results-section');
    const conditionBadge = document.getElementById('condition-badge');
    const conditionName = document.getElementById('condition-name');
    const confidenceValue = document.getElementById('confidence-value');
    const btnReset = document.getElementById('btn-reset');

    // State Variables
    let currentCameraActive = false;
    let capturedBlob = null;
    let currentPreviewUrl = null;

    // 1. Splash Screen Handler
    setTimeout(() => {
        if (splashScreen) {
            splashScreen.classList.add('fade-out');
        }
    }, 1800);

    // 2. Initialize Camera Stream
    async function initCamera() {
        cameraError.classList.add('hidden');
        cameraHud.classList.remove('hidden');

        const success = await window.cameraController.start(cameraVideo, (errorMsg) => {
            showCameraError(errorMsg);
        });

        currentCameraActive = success;
    }

    function showCameraError(message) {
        currentCameraActive = false;
        cameraHud.classList.add('hidden');
        cameraErrorText.textContent = message;
        cameraError.classList.remove('hidden');
    }

    initCamera();

    // 3. CAPTURE Handler (Freezes frame, shows preview, DOES NOT call API)
    btnCapture.addEventListener('click', async () => {
        if (!currentCameraActive) {
            alert("Camera is not active. Please use the 'Upload Image' button to select an image from your device.");
            return;
        }

        const blob = await window.cameraController.captureBlob(captureCanvas);
        if (!blob) {
            alert("Failed to capture camera frame. Please try again.");
            return;
        }

        capturedBlob = blob;

        // Revoke previous object URL if any
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
        }
        currentPreviewUrl = URL.createObjectURL(capturedBlob);

        // Display Captured Frame Preview
        imagePreview.src = currentPreviewUrl;
        imagePreview.classList.remove('hidden');
        cameraVideo.classList.add('hidden');
        cameraHud.classList.add('hidden');
        if (previewContainer) previewContainer.classList.add('captured-mode');

        // Update UI Title and Control Buttons
        viewTitleText.textContent = "CAPTURE PREVIEW";
        liveControls.classList.add('hidden');
        captureControls.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    });

    // 4. RETAKE Handler (Discards frame, returns to live camera, DOES NOT call API)
    btnRetake.addEventListener('click', () => {
        resetToLiveCamera();
    });

    // 5. ANALYZE IMAGE Handler (Sends captured image to Django prediction API)
    btnAnalyze.addEventListener('click', async () => {
        if (!capturedBlob) {
            alert("No captured image available for analysis.");
            return;
        }

        await sendImageForPrediction(capturedBlob, "captured_lubricant_sample.jpg");
    });

    // 6. Simple Upload Button Handler (Direct File Upload Analysis)
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        capturedBlob = file;

        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
        }
        currentPreviewUrl = URL.createObjectURL(file);

        // Display preview image
        imagePreview.src = currentPreviewUrl;
        imagePreview.classList.remove('hidden');
        cameraVideo.classList.add('hidden');
        cameraHud.classList.add('hidden');
        if (previewContainer) previewContainer.classList.add('captured-mode');

        viewTitleText.textContent = "UPLOADED IMAGE PREVIEW";
        liveControls.classList.add('hidden');
        captureControls.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        // Automatically analyze uploaded file directly
        await sendImageForPrediction(file, file.name);

        // Clear file input so re-selecting same file fires change event
        fileInput.value = '';
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
                alert(`Analysis Error: ${data.error || 'Failed to process image.'}`);
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

    // 8. Render Prediction Result
    function renderResult(condition, confidence, degradationScore) {
        // Unhide results section so DOM elements are computed in layout
        resultsSection.classList.remove('hidden');

        conditionName.textContent = condition;
        confidenceValue.textContent = `${confidence.toFixed(1)}%`;

        // Style status badge
        conditionBadge.className = 'condition-badge';
        const condLower = condition.toLowerCase();
        if (condLower.includes('fresh')) {
            conditionBadge.classList.add('fresh');
        } else if (condLower.includes('semi')) {
            conditionBadge.classList.add('semi');
        } else {
            conditionBadge.classList.add('fully');
        }

        // Extract and clamp degradation score
        const score = Math.max(0, Math.min(100, Number(degradationScore)));

        // Update score text (e.g. 63.5 / 100)
        const degradationScoreNum = document.getElementById('degradation-score-num');
        if (degradationScoreNum) {
            degradationScoreNum.textContent = score.toFixed(1);
        }

        // Update progress bar width and level styling
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

        // Update interpretation text
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

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    // 9. Reset / New Analysis Handler
    btnReset.addEventListener('click', () => {
        resetToLiveCamera();
    });

    function resetToLiveCamera() {
        // Discard captured blob and preview URL
        capturedBlob = null;
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
            currentPreviewUrl = null;
        }

        // Hide preview and results
        imagePreview.classList.add('hidden');
        resultsSection.classList.add('hidden');
        if (previewContainer) previewContainer.classList.remove('captured-mode');

        // Restore UI Title and Live Camera Controls
        viewTitleText.textContent = "LIVE CAMERA PREVIEW";
        captureControls.classList.add('hidden');
        liveControls.classList.remove('hidden');

        // Restore Camera Video Stream
        if (currentCameraActive) {
            cameraVideo.classList.remove('hidden');
            cameraHud.classList.remove('hidden');
        } else {
            initCamera();
        }
    }

    // 10. Mobile Scan QR Modal Handler
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
            if (e.target === qrModal) {
                closeQrModal();
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && qrModal && !qrModal.classList.contains('hidden')) {
            closeQrModal();
        }
    });

    /**
     * Mobile QR Code Configuration:
     * - LOCAL DEVELOPMENT: Configure MOBILE_BASE_URL=http://<PC-LAN-IP>:8000 in your environment.
     *   Ensure both your host PC and mobile phone are connected to the SAME Wi-Fi network.
     * - PRODUCTION / Vercel: Configure MOBILE_BASE_URL=https://<production-domain>.
     * - Encodes: MOBILE_BASE_URL + "/mobile-upload/"
     */
    async function openQrModal() {
        if (!qrModal) return;

        // Resolve mobile upload URL
        let mobileUrl = '';
        const clientOrigin = window.location.origin.replace(/\/+$/, '');
        const isLoopback = ['127.0.0.1', 'localhost', '0.0.0.0'].includes(window.location.hostname.toLowerCase());

        if (isLoopback && typeof SERVER_MOBILE_UPLOAD_URL !== 'undefined' && SERVER_MOBILE_UPLOAD_URL) {
            mobileUrl = SERVER_MOBILE_UPLOAD_URL;
        } else {
            mobileUrl = clientOrigin + "/mobile-upload/";
        }

        // Prevent loopback IP/localhost from being encoded if server LAN URL is available
        if (isLoopback && (mobileUrl.includes('127.0.0.1') || mobileUrl.includes('localhost'))) {
            try {
                const res = await fetch('/api/mobile-url/');
                if (res.ok) {
                    const data = await res.json();
                    if (data.mobile_url) {
                        mobileUrl = data.mobile_url;
                    }
                }
            } catch (e) {
                console.warn('[LUBVISION QR] Could not fetch server LAN URL via API:', e);
            }
        }

        // Log exact URL encoded in QR code
        console.log('[LUBVISION QR] Encoding mobile URL into QR code:', mobileUrl);
        
        if (qrUrlInput) {
            qrUrlInput.value = mobileUrl;
        }

        qrModal.classList.remove('hidden');

        // Clear and regenerate QR code every time modal opens
        if (qrCodeBox) {
            qrCodeBox.innerHTML = '';

            try {
                if (typeof QRCode === 'function') {
                    new QRCode(qrCodeBox, {
                        text: mobileUrl,
                        width: 220,
                        height: 220,
                        colorDark: "#000000",
                        colorLight: "#ffffff",
                        correctLevel: QRCode.CorrectLevel ? QRCode.CorrectLevel.M : 0
                    });
                } else {
                    console.warn('[LUBVISION QR] QRCode client library not found, fetching fallback SVG from /api/qr/');
                    const qrApiUrl = `/api/qr/?url=${encodeURIComponent(mobileUrl)}&t=${Date.now()}`;
                    const response = await fetch(qrApiUrl);
                    if (response.ok) {
                        qrCodeBox.innerHTML = await response.text();
                    } else {
                        throw new Error(`Server returned status ${response.status}`);
                    }
                }
            } catch (err) {
                console.error('[LUBVISION QR] Error generating QR code:', err);
                qrCodeBox.innerHTML = '<span class="qr-loading-text" style="color: #ef4444;">Error generating QR code. See browser console for details.</span>';
            }
        }
    }

    function closeQrModal() {
        if (qrModal) {
            qrModal.classList.add('hidden');
        }
    }

    if (btnCopyUrl && qrUrlInput) {
        btnCopyUrl.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(qrUrlInput.value);
                const originalHTML = btnCopyUrl.innerHTML;
                btnCopyUrl.innerHTML = '✓ Copied!';
                btnCopyUrl.style.borderColor = 'var(--color-fresh)';
                btnCopyUrl.style.color = 'var(--color-fresh)';

                setTimeout(() => {
                    btnCopyUrl.innerHTML = originalHTML;
                    btnCopyUrl.style.borderColor = '';
                    btnCopyUrl.style.color = '';
                }, 2000);
            } catch (err) {
                qrUrlInput.select();
                document.execCommand('copy');
                alert('URL copied to clipboard!');
            }
        });
    }

});
