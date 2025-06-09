from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import mlib

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

@app.route("/")
def home():
    html = f"<h3> Predict the height from the weight of a player </h3>"
    html += f"<p>Use the endpoint <code>/predict</code> with a POST request to get the height prediction.</p>"
    return html.format(format)


@app.route("/predict", methods=["POST"])
def predict_height():
    """Endpoint to predict height based on weight"""
    json_playload = request.get_json()
    LOG.info(f"Received payload: %s", json_playload)
    prediction = mlib.predict(json_playload["weight"])
    return jsonify({"Prediction" :  prediction})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)