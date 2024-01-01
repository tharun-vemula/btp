from flask import Flask, render_template, request

from os.path import join, dirname
import numpy as np
from joblib import Parallel, delayed
import joblib

filename = join(dirname(__file__), "model", "mos_predictor.pkl")
model = joblib.load(filename)


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
    return render_template("result.html", rent=mos)


if __name__ == "__main__":
    app.run(debug=True)
