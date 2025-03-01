# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.chatbot_interface, name='chatbot_interface')
# ]
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_interface, name='chatbot_interface'),
    # Add other URL patterns if you have any
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# # Serve static files during development
# if settings.DEBUG:
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
