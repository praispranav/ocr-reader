from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
import numpy as np;

model = load_model("best_model.h5")
print("Modal Loaded");

def imageTester(file_name):

    path = file_name
    img = load_img(path, target_size=(256, 256))

    i = img_to_array(img)
    i = preprocess_input(i)

    input_arr = np.array([i])

    pred = np.argmax(model.predict(input_arr))
    if pred == 0:
        return "Driving License"
    elif pred == 1:
        return "MyKad"
    else:
        return ""