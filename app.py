from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib

# Load classifer
classifier = joblib.load("stroke_predictor.model")

# Load standard scaler
standard_scaler = joblib.load("standard_scaler.model")

# Define Flask app
app = Flask(__name__)

# Define index
@app.route("/")
def index():
    return render_template("index.html")

# Create API endpoint based on HTML form
@app.route("/api/predict")
def predict():
    age = float(request.args.get("age")) or 0
    average_glucose_level = float(request.args.get("agl")) or 0
    # bmi = float(request.args.get("bmi")) or 0

    # We one-hot-encoded the categorical columns
    # Therefore we create a mapper for each of these features,
    #  to "convert" the users input to the correct format
    work_type_mapper = {
        "self_employed": [1,0,0],
        "children": [0,1,0],
        "employer_employed": [0,0,1]
    }
    work_type = request.args.get("work_type")

    # In our initial data analysis we noticed that,
    # the smoking_status data does not reflect statements made in a scientific journal
    # on the effects of smoking on having a stroke
    # We therefore opted not to use smoking_status as a feature
    # We leave it commented it out here,
    # so that we may return at a later time if we decide to use this in some way
    smoking_status_mapper = {
        "formerly_smoked": [0, 0, 1],
        "never_smoked": [0, 1, 0],
        "smokes": [1, 0, 0]
    }
    # smoking_status_mapper = {
    #     "currently_smokes": [1, 0],
    #     "doesnt_currently_smoke": [0, 1]
    # }
    smoking_status = request.args.get("smoking_status")

    hypertension_mapper = {
        0: [1, 0],
        1: [0, 1]
    }
    hypertension = request.args.get("hypertension") or 0
    if hypertension:
        hypertension = 1
    else:
        hypertension = 0
    
    heart_disease_mapper = {
        0: [1, 0],
        1: [0, 1]
    }
    heart_disease = request.args.get("heart_disease") or 0
    if heart_disease:
        heart_disease = 1
    else:
        heart_disease = 0

    ever_married_mapper = {
        0: [1, 0],
        1: [0, 1]
    }
    ever_married = request.args.get("ever_married") or 0
    if ever_married:
        ever_married = 1
    else:
        ever_married = 0

    # Create a list of the user's given features
    # We one-hot-encoded the categorical columns,
    # so we must use a mapper to pass the user's single input
    # to each relevant column

    # In our initial data analysis we noticed that,
    # the smoking_status data does not reflect statements made in a scientific journal
    # on the effects of smoking on having a stroke
    # We therefore opted not to use smoking_status as a feature
    # We leave it commented it out here,
    # so that we may return at a later time if we decide to use this in some way
    features = [[age,
                 average_glucose_level,
                #  bmi,
                 work_type_mapper[work_type][0],
                 work_type_mapper[work_type][1],
                 work_type_mapper[work_type][2],
                 smoking_status_mapper[smoking_status][0],
                 smoking_status_mapper[smoking_status][1],
                 smoking_status_mapper[smoking_status][2],
                 hypertension_mapper[hypertension][0],
                 hypertension_mapper[hypertension][1],
                 heart_disease_mapper[heart_disease][0],
                 heart_disease_mapper[heart_disease][1],
                 ever_married_mapper[ever_married][0],
                 ever_married_mapper[ever_married][1]]]

    # Scale features
    scaled_features = standard_scaler.transform(features)

    # Make a prediction based on the user's input
    prediction = classifier.predict(scaled_features)
    # Determine prediction probability
    prediction_probability = classifier.predict_proba(scaled_features)

    # print(100*"#")
    # print(features)
    # print(100*"#")

    # Return a JSON giving the prediction and probability thereof
    return jsonify({"stroke_prediction": prediction.tolist(),
                    "stroke_prediction_probability": prediction_probability.tolist(),
                    "features": features})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)