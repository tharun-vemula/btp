from flask import Flask, render_template, request
import tensorflow as tf
from os.path import join, dirname
import numpy as np
import joblib
from pickle import load

filename = join(dirname(__file__), "models", "mos", "mos_predictor.pkl")
model = joblib.load(filename)

new_model = tf.keras.models.load_model("models/avg")

scaler = load(open("models/scaler/scaler.pkl", "rb"))

THRESHOLD = 3

app = Flask(__name__)


@app.route("/")
def hello_word():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def find_mos():
    ans = request.form.to_dict()
    raw_features = [
        float(ans["bandwidth"]),
        float(ans["ramclockSpeed"]),
        float(ans["screenMHz"]),
        float(ans["screenDimension"]),
        float(ans["stepping"]),
        float(ans["nbitrate"]),
        float(ans["complexity"]),
        float(ans["delay"]),
        float(ans["jitter"]),
        float(ans["complexityClass"]),
        float(ans["plugType"]),
        float(ans["resolution"]),
        float(ans["Mhzmoy"]),
        float(ans["Mhzavg"]),
        float(ans["nbr"]),
        float(ans["processor"]),
    ]
    features = [np.array(raw_features)]
    prediction = model.predict(features)
    mos = prediction[0]

    if mos < THRESHOLD:
        return render_template("neg-result.html", mos=mos)
    else:
        return render_template("pos-result.html", mos=mos)


@app.route("/get-avg", methods=["POST"])
def find_bitrate():
    payload = np.array([float(x) for x in request.form.values()])
    transformed_payload = scaler.transform(payload.reshape(1, 19))
    payload_reshaped = transformed_payload.reshape(1, 19, 1)
    print(payload_reshaped)
    prediction = new_model.predict(payload_reshaped)
    print(prediction)
    avg = prediction[0][0]
    return render_template("result.html", avg=avg)


if __name__ == "__main__":
    app.run(debug=True)
