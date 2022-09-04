from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2;
from .drivingLicenseMal import  scanImage
from django.http import JsonResponse

def handle_uploaded_file(f, fileName):
    with open(fileName, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def index(request):
    if request.method == 'POST':
        file = request.FILES['file']
        print(file);
        file_name = "SampleDriving.jpg"
        handle_uploaded_file(file, file_name);
        texts = scanImage(file_name)
        print("File Extraction Completed")
        print(texts);
        return JsonResponse({ 'data': texts, })
        # image = cv2.imread(file_name);
        # cv2.imshow("Object", image);
        # cv2.waitKey(0)

        # if form.is_valid():
        #     handle_uploaded_file(request.FILES['file'])
        #     return HttpResponseRedirect('/success/url/')
    else:
        print("No Post Method")
    return HttpResponse("Please use Post mehtod with form Data.")