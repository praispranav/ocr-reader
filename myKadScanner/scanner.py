import cv2;
import numpy as np;
import pytesseract;
import os;
import random
import face_recognition
from PIL import Image, ImageEnhance, ImageFilter

original_train_image = "vajra/myKadSample.jpg"
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe";

per = 25;
custom_config_text = r'--oem 3 --psm 6 outputbase digits'
custom_config_number = r'--oem 3 --psm 6 digits'

roi = [

    [(14, 90), (281, 135), 'number', "IC No"],

    [(8, 264), (413, 317), 'text', "Name"],
    # [(9, 318), (374, 431), 'text', "Address"],
    #
    # [(480, 372), (708, 429), 'text', "Gender"],
]

def myKadScanner(fileName):
    ocrImage = cv2.imread(original_train_image);
    h, w, c = ocrImage.shape

    orb = cv2.ORB_create(20000);
    kp1, des1 = orb.detectAndCompute(ocrImage, None)

    img2 = cv2.imread(fileName);
    kp2, des2 = orb.detectAndCompute(img2, None);

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good = matches[:int(len(matches) * (per / 100))]

    dstPoints = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    srcPoints = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img2, M, (w, h))
    file_name = f'media/images/myKad/{random.randint(100000, 999999999)}License.jpg'
    to_json = {
        'imgName': file_name
    }
    # cv2.imshow("Final", imgScan)
    cv2.imwrite(to_json['imgName'], imgScan)
    print(f'Image Saved {file_name}')

    # imgScan = cv2.imread(to_json['imgName'])
    # cv2.imshow("Enhanced Image", imgScan)
    gray = cv2.cvtColor(imgScan, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)

    imgShow = imgScan.copy();
    imgMask = np.zeros_like(imgShow);

    for x, r in enumerate(roi):
        cv2.rectangle(imgScan, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), thickness=1)
        # imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = gray[r[0][1]:r[1][1], r[0][0]: r[1][0]]
        # cv2.imshow(str(x), imgCrop);

        if r[2] == 'text':
            h, w = imgCrop.shape
            imgCrop = cv2.resize(imgCrop, (w*3, h*3))
            to_json[r[3]] = pytesseract.image_to_string(imgCrop)
        if r[2] == 'number':
            h, w = imgCrop.shape
            imgCrop = cv2.resize(imgCrop, (w*3, h*3))
            custom_oem = r'digits --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'
            to_json[r[3]] = pytesseract.image_to_string(imgCrop, config=custom_oem)

    print(to_json)
    fc = face_recognition.load_image_file(file_name)
    fc = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)

    fc_loc = face_recognition.face_locations(fc)[0]
    print(fc_loc)
    cv2.rectangle(fc, (fc_loc[3], fc_loc[0]), (fc_loc[1], fc_loc[2]), (255, 0, 255), thickness=2)
    croppedFace = fc[fc_loc[0]: fc_loc[2], fc_loc[3]: fc_loc[1]]
    fc_img_name = f'media/images/faces/{random.randint(100000, 999999999)}Face.jpg'
    cv2.imwrite(fc_img_name, croppedFace)
    to_json['faceImg'] = fc_img_name
    return to_json


