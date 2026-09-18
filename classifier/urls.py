from django.urls import path
from classifier import views

app_name = 'classifier'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('mobile-upload/', views.mobile_view, name='mobile_upload'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/qr/', views.generate_qr_api, name='qr_api'),
    path('api/mobile-url/', views.get_mobile_url_api, name='mobile_url_api'),
]
