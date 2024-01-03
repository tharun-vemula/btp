from flask import Flask, render_template, request

from os.path import join, dirname
import numpy as np
from joblib import Parallel, delayed
import joblib

filename = join(dirname(__file__), "model", "mos_predictor.pkl")
model = joblib.load(filename)

fileNameOfModel = join(dirname(__file__), "model", "bitrate-predictor.pkl")
bit_rate_predictor = joblib.load(fileNameOfModel)

THRESHOLD = 3

app = Flask(__name__)


@app.route("/")
def hello_word():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def find_price():
    ans = request.form.to_dict()
    raw_features = [
        int(ans["bandwidth"]),
        int(ans["ramclockSpeed"]),
        int(ans["screenMHz"]),
        int(ans["screenDimension"]),
        int(ans["stepping"]),
        int(ans["nbitrate"]),
        int(ans["complexity"]),
        int(ans["delay"]),
        int(ans["jitter"]),
        int(ans["complexityClass"]),
        int(ans["plugType"]),
        int(ans["resolution"]),
        int(ans["Mhzmoy"]),
        int(ans["Mhzavg"]),
        int(ans["nbr"]),
        int(ans["processor"]),
    ]
    features = [np.array(raw_features)]
    prediction = model.predict(features)
    mos = prediction[0]

    if mos < THRESHOLD:
        return render_template("neg-result.html", mos=mos)
    else:
        return render_template("pos-result.html", mos=mos)


@app.route("/get-avg", methods=["POST"])
def find_price():
    payload = np.array(int(x) for x in request.form.values())
    payload_reshaped = payload.reshape(1, 19, 1)
    prediction = model.predict(payload_reshaped)
    avg = prediction[0][0]
    return render_template("result.html", avg=avg)


if __name__ == "__main__":
    app.run(debug=True)
