from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image
from .digit_model import predict_digit

class DigitPredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file = request.FILES.get('image')
        if not file:
            return Response({"error": "No image uploaded"}, status=400)

        try:
            image = Image.open(file).convert("L")
            digit, confidence, all_scores = predict_digit(image)

            return Response({
                "digit": digit,
                "confidence": confidence,
                "all_scores": all_scores
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)
