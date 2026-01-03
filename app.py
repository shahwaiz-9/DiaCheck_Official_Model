from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Model (simple is with out scaling)
# Model1 (with scaling)


# Load the diabetes prediction model
with open('Diabetes_model1.pkl', 'rb') as file:
    model = pickle.load(file)


# Route for web application (Initial page)
@app.route('/')
def home():
    return render_template('index.html')


# Route for web application

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])
        
        # Create feature array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            result = "Diabetic"
            result_class = "diabetic"
        else:
            result = "Normal"
            result_class = "normal"
        
        return render_template('index.html', prediction=result, result_class=result_class)
    
    except Exception as e:
        return render_template('index.html', error=str(e))


# Api route for application https request

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        # Try to get JSON
        data = request.get_json(force=True)  # <-- force=True ensures Flask parses JSON even if headers missing

        features = np.array([[
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["blood_pressure"]),
            float(data["skin_thickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["diabetes_pedigree"]),
            float(data["age"])
        ]])

        prediction = int(model.predict(features)[0])

        return jsonify({
            "prediction": prediction,
            "result": "Diabetic" if prediction == 1 else "Normal"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



# Api route for application https request

if __name__ == '__main__':
    app.run(debug=True)
