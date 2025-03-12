from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('web.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Retrieve the input values from the form
    sepal_length = float(request.form['Sepal_Length'])
    sepal_width = float(request.form['Sepal_Width'])
    petal_length = float(request.form['Petal_Length'])
    petal_width = float(request.form['Petal_Width'])

    # Prepare the input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    
    # Make the prediction
    prediction = model.predict(input_data)

    # Get the predicted class
    prediction_text = f"Flower Species is  {prediction[0]}"

    return render_template('web.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
