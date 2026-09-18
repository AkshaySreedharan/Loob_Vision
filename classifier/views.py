import os
import socket
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import ensure_csrf_cookie
from classifier.services.preprocessing import preprocess_image
from classifier.services.predictor import get_predictor


def get_lan_ip():
    """Returns host machine's primary local area network (LAN) IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def get_mobile_upload_url(request=None):
    """
    Resolves mobile upload URL for QR code generation.

    Configuration Note:
      1. For LOCAL DEVELOPMENT:
         Set MOBILE_BASE_URL environment variable to your computer's local LAN IPv4 address:
           MOBILE_BASE_URL=http://<PC-LAN-IP>:8000  (e.g., MOBILE_BASE_URL=http://192.168.1.50:8000)
         Both your host PC and mobile phone MUST be connected to the SAME Wi-Fi network.
         If MOBILE_BASE_URL is not explicitly set in dev, the server auto-detects the host LAN IP.

      2. For PRODUCTION / Vercel:
         Set MOBILE_BASE_URL environment variable to your deployed HTTPS domain:
           MOBILE_BASE_URL=https://<production-domain>  (e.g., MOBILE_BASE_URL=https://lubvision.vercel.app)

      3. The generated QR URL is always formatted as:
         MOBILE_BASE_URL + "/mobile-upload/"
    """
    env_url = os.environ.get('MOBILE_BASE_URL', '').strip()
    if env_url:
        base = env_url.rstrip('/')
        return f"{base}/mobile-upload/"

    if request is not None:
        host = request.get_host()
        scheme = request.scheme
        hostname = host.split(':')[0].lower()
        port = host.split(':')[1] if ':' in host else ''
        port_suffix = f":{port}" if port else ""

        if hostname in ('127.0.0.1', 'localhost', '0.0.0.0'):
            lan_ip = get_lan_ip()
            if lan_ip and lan_ip not in ('127.0.0.1', '0.0.0.0'):
                return f"{scheme}://{lan_ip}{port_suffix}/mobile-upload/"

        return f"{scheme}://{host}/mobile-upload/"

    lan_ip = get_lan_ip()
    return f"http://{lan_ip}:8000/mobile-upload/"


@ensure_csrf_cookie
def index_view(request):
    """Renders the main LUBVISION industrial single-page UI."""
    mobile_url = get_mobile_upload_url(request)
    return render(request, 'classifier/index.html', {'mobile_upload_url': mobile_url})


@ensure_csrf_cookie
def mobile_view(request):
    """Renders the dedicated mobile rear-camera sample scan interface."""
    return render(request, 'classifier/mobile.html')


@require_GET
def get_mobile_url_api(request):
    """API endpoint returning the resolved mobile upload URL."""
    mobile_url = get_mobile_upload_url(request)
    return JsonResponse({'mobile_url': mobile_url})


@require_POST
def predict_api(request):
    """
    API Endpoint for lubricant image analysis.
    Accepts image file in multipart/form-data under key 'image'.
    """
    if 'image' not in request.FILES:
        return JsonResponse({
            'success': False,
            'error': 'No image file provided. Please select or capture an image.'
        }, status=400)

    image_file = request.FILES['image']

    # Enforce 10MB file size limit
    if image_file.size > 10 * 1024 * 1024:
        return JsonResponse({
            'success': False,
            'error': 'Image file size exceeds the 10MB limit.'
        }, status=400)

    try:
        image_bytes = image_file.read()
        img_batch, error_msg = preprocess_image(image_bytes)

        if error_msg or img_batch is None:
            return JsonResponse({
                'success': False,
                'error': error_msg or 'Failed to preprocess image.'
            }, status=400)

        # Run inference using singleton predictor
        predictor = get_predictor()
        result = predictor.predict(img_batch)

        return JsonResponse({
            'success': True,
            'class_code': result['class_code'],
            'condition': result['condition'],
            'confidence': result['confidence'],
            'degradation_score': result['degradation_score']
        })

    except Exception as e:
        # User-friendly error response without exposing stack traces
        return JsonResponse({
            'success': False,
            'error': 'An internal error occurred while processing the image.'
        }, status=500)


def generate_qr_api(request):
    """
    Generates a high-contrast SVG QR code encoding the current application host URL.
    Does not hardcode localhost or 127.0.0.1; uses LAN IP or MOBILE_BASE_URL.
    """
    import io
    import qrcode
    import qrcode.image.svg

    # Dynamic target URL resolution
    target_url = request.GET.get('url', '').strip()
    if not target_url:
        target_url = get_mobile_upload_url(request)

    try:
        factory = qrcode.image.svg.SvgImage
        img = qrcode.make(target_url, image_factory=factory, box_size=10)
        stream = io.BytesIO()
        img.save(stream)
        svg_data = stream.getvalue().decode('utf-8')

        return HttpResponse(svg_data, content_type='image/svg+xml')
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

