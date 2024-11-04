from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input data from the form
        data = [float(x) for x in request.form.values()]
        
        # Ensure that the input data is in the correct shape for the model
        features = np.array([data])  # Create a 2D array with a single sample

        # Make prediction
        prediction = model.predict(features)
        prediction_probability = model.predict_proba(features)

        # Determine diabetes status based on the prediction
        diabetes_status = "You are diabetic" if prediction[0] == 1 else "You are not diabetic"

        # Return the result to the HTML page
        return render_template('index.html', prediction=diabetes_status, probability=prediction_probability[0][1])

    except Exception as e:
        # Handle any exceptions and return an error message
        return render_template('index.html', prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)



    
