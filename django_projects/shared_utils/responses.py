# Creado por Lucy
# Fecha: 26/02/2024

from rest_framework.response import Response
from rest_framework import status

def standard_response(data, message="", code=status.HTTP_200_OK, success=True):
    return Response({
        "success": success,
        "data": data,
        "message": message,
        "code": code
    }, status=code)
