import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array


model = tf.keras.models.load_model("pill_model.h5")


class_names = [
    "multivitamin",
    "paracetamol",
    "vitamin_c"
]


img = load_img(
    "test.jpg",
    target_size=(224, 224)
)


img_array = img_to_array(img)


img_array = img_array / 255.0


img_array = np.expand_dims(img_array, axis=0)


prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)

medicine = class_names[predicted_class]

confidence = np.max(prediction) * 100

medicine_info = {

    "paracetamol": {
        "use": "Fever and pain relief",
        "dosage": "500 mg",
        "warning": "Do not exceed prescribed dose"
    },

    "vitamin_c": {
        "use": "Immunity support",
        "dosage": "500 mg daily",
        "warning": "Take after food"
    },

    "multivitamin": {
        "use": "Nutritional supplement",
        "dosage": "1 tablet daily",
        "warning": "Do not overdose"
    }
}

print("\n===================================")
print("     MEDICINE DETECTION RESULT")
print("===================================")

print(f"\nDetected Medicine : {medicine}")
print(f"Confidence         : {confidence:.2f}%")

print(f"\nUse      : {medicine_info[medicine]['use']}")
print(f"Dosage   : {medicine_info[medicine]['dosage']}")
print(f"Warning  : {medicine_info[medicine]['warning']}")

print("\n===================================")