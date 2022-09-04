from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .scanner import  myKadScanner
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
        file_name = "SampleMyKad.jpg"
        handle_uploaded_file(file, file_name);
        texts = myKadScanner(file_name)
        print("File Extraction Completed")
        print(texts);
        return JsonResponse({ 'data': texts, })
    else:
        print("No Post Method")
    return HttpResponse("Please use Post method with form Data.")