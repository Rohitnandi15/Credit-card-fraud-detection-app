from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the form data
        features = [float(request.form[f]) for f in ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']]

        # Ensure the data is correctly shaped
        data = np.array([features])

        # Scale the features
        data_scaled = scaler.transform(data)

        # Get prediction probability
        prob = model.predict_proba(data_scaled)[0][1]

        # Set threshold and make prediction (fraud if prob > threshold)
        threshold = 0.3
        prediction = int(prob > threshold)

        # Output the prediction probability for debugging
        print(f"Prediction probability: {prob}")

        # Determine result text
        result = 'Legitimate ✅' if prediction == 0 else 'Fraudulent ❌'

        # Return the result to the template
        return render_template('index.html', prediction_text=f"Prediction: {result} (prob: {prob:.2f})")

    except Exception as e:
        # If an error occurs, display it
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
