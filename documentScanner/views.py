import sys
sys.path.append("..")
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .modalClassifer import imageTester
from .scanner import myKadScanner;
from .drivingLicenseMal import scanImage

import json
import pybase64
import base64
# from base64 import decodestring


def handle_uploaded_file(f, fileName):
    with open(fileName, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def index(request):
    if request.method == 'POST':
        received_json_data = json.loads(request.body)
        file_name = "DocumentScannerImageFile.jpg"

        with open(file_name, "wb") as f:
            f.write(pybase64.b64decode(received_json_data['base64']))

        image_type = imageTester(file_name)
        if image_type == "MyKad":
            texts = myKadScanner(file_name)

            image = open(texts['imgName'], 'rb')
            image_read = image.read()
            image_64_encode = pybase64.b64encode(image_read)
            image_encoded = image_64_encode.decode('utf-8')

            image_face = open(texts['faceImg'], 'rb')
            image_read_face = image_face.read()
            image_64_encode_face = pybase64.b64encode(image_read_face)
            image_encoded_face = image_64_encode_face.decode('utf-8')

            return JsonResponse({'data': texts, "type": "MyKad","base64": image_encoded, "faceBase64": image_encoded_face })
        elif image_type == "Driving License":
            texts = scanImage(file_name)

            image = open(texts['imgName'], 'rb')
            image_read = image.read()
            image_64_encode = pybase64.b64encode(image_read)
            image_encoded = image_64_encode.decode('utf-8')

            image_face = open(texts['faceImg'], 'rb')
            image_read_face = image_face.read()
            image_64_encode_face = pybase64.b64encode(image_read_face)
            image_encoded_face = image_64_encode_face.decode('utf-8')

            return JsonResponse({'data': texts, "type": "Driving License", "base64": image_encoded, "faceBase64": image_encoded_face  })
        # texts = scanImage(file_name)
        print("File Extraction Completed")
        # print(texts);
        # return JsonResponse({ 'data': texts, })
        # image = cv2.imread(file_name);
        # cv2.imshow("Object", image);
        # cv2.waitKey(0)

        # if form.is_valid():
        #     handle_uploaded_file(request.FILES['file'])
        #     return HttpResponseRedirect('/success/url/')
    else:
        print("No Post Method")
    return HttpResponse("Please use Post mehtod with form Data.")