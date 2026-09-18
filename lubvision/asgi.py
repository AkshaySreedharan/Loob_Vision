"""
ASGI config for lubvision project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lubvision.settings')
application = get_asgi_application()
