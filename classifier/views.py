from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .modalClassifer import imageTester

def handle_uploaded_file(f, fileName):
    with open(fileName, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def index(request):
    if request.method == 'POST':
        file = request.FILES['file']
        print(file);
        file_name = "SampleClassificationDocument.jpg"
        handle_uploaded_file(file, file_name);
        document_type = imageTester(file_name);
        print(type);
        return JsonResponse({ 'type': document_type })
    else:
        print("No Post Method")
    return HttpResponse("You have visited wrong link.")

