from django.contrib import admin
from django.urls import path, include  # Import the include function

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),  # Include the URL configuration for your app
]
