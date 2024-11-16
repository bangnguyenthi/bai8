import cv2
import numpy as np

def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print("Dự đoán: Chó")
    else:
        print("Dự đoán: Mèo")
