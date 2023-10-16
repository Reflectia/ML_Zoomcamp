import pickle
from flask import Flask
from flask import request
from flask import jsonify

dv_file = 'dv.bin'
model_file = 'model1.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('credit_scoring')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    credit_scoring = y_pred >= 0.5

    result = {
        "credit_scoring_probability": float(y_pred),
        "credit_scoring": bool(credit_scoring)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)


