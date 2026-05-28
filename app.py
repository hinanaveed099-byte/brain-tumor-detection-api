# ============================================================
# FILE NAME: app.py
# FLASK API FOR BRAIN TUMOR DETECTION
# ============================================================

from flask import Flask, request, jsonify

import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    "clean_brain_tumor_model.keras",
    compile=False
)

print("\n====================================")
print("✅ MODEL LOADED SUCCESSFULLY")
print("====================================")

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")

def home():

    return jsonify({

        "message": "Brain Tumor Detection API Running Successfully"

    })

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])

def predict():

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        return jsonify({

            "success": False,
            "message": "No image uploaded"

        })

    file = request.files["image"]

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    image_path = "temp.jpg"

    file.save(image_path)

    # ========================================================
    # PREPROCESS
    # ========================================================

    processed_img = preprocess_image(image_path)

    if processed_img is None:

        return jsonify({

            "success": False,
            "message": "Invalid image"

        })

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(processed_img)[0][0]

    # ========================================================
    # NO TUMOR
    # ========================================================

    if prediction < 0.5:

        confidence = float((1 - prediction) * 100)

        result = "NO BRAIN TUMOR"

        severity = "NORMAL"

        suggestion = "Brain MRI looks normal."

    # ========================================================
    # TUMOR
    # ========================================================

    else:

        confidence = float(prediction * 100)

        result = "BRAIN TUMOR DETECTED"

        # ====================================================
        # SEVERITY
        # ====================================================

        if confidence < 70:

            severity = "MILD"

        elif confidence < 90:

            severity = "MODERATE"

        else:

            severity = "HIGH"

        suggestion = "Consult neurologist for further analysis."

    # ========================================================
    # JSON RESPONSE
    # ========================================================

    return jsonify({

        "success": True,
        "prediction": result,
        "confidence": round(confidence, 2),
        "severity": severity,
        "suggestion": suggestion

    })

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",
        port=5000,
        debug=True

    )
