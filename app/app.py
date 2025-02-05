from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load Model & Scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data["features"]).reshape(1, -1)  
        input_scaled = scaler.transform(input_data)  
        prediction = model.predict(input_scaled)
        return jsonify({"income_prediction": ">$50K" if prediction[0] == 1 else "<=$50K"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
