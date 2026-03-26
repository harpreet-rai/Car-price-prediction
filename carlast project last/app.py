from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("car_price_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    km_driven = int(request.form["km_driven"])
    fuel = int(request.form["fuel"])
    seller_type = int(request.form["seller_type"])
    transmission = int(request.form["transmission"])
    owner = int(request.form["owner"])

    prediction = model.predict([[year, km_driven,
                                 fuel,
                                 seller_type,
                                 transmission,
                                 owner]])

    output = round(prediction[0], 2)

    return render_template("index.html",
                           prediction_text=f"Estimated Car Price: ₹ {output}")


if __name__ == "__main__":
    app.run(debug=True)