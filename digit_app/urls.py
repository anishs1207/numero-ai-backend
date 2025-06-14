from django.urls import path
from .views import DigitPredictView

urlpatterns = [
    path("predict-digit/", DigitPredictView.as_view(), name="predict-digit"),
]
