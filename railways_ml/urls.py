from django.urls import path

from railways_ml.views import RunML

urlpatterns = [
    path('ml/', RunML.as_view(), name='run-ml'),
]
