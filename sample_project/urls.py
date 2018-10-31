
from django.contrib import admin
from django.urls import path
from api.views import *



urlpatterns = [
    path('admin/', admin.site.urls),
    path('get_data/',GET_APIView.as_view()),
    path('get_job/',job_APIView.as_view()),
    path('abc/',abc.as_view()),
]
