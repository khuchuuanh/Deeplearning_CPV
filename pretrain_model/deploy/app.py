from flask import Flask, request, render_template, jsonify
import io 
from flask_cors import CORS

from predict import *

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])

def predict_label():
    if request.method == "GET":
        return render_template("index.html", value="Image")
    if request.method == "POST":
        if "file" not in request.files:
            return "Image not uploaded"
        
        file = request.files["file"].read()

        try:
            img = Image.open(io.BytesIO(file))
        except IOError:
            return jsonify(predictions = "Chưa tải ảnh mà lại đi bắt người ta dự đoán nhỉ ??")

        img = img.convert("RGB")

        label = predict(img)

        return jsonify(predictions = label)

if __name__ == "__main__":
    app.run(debug=True)
