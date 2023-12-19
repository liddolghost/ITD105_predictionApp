from flask import Flask, render_template, url_for, request, jsonify
import os
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/drug')
def drug():
    return render_template('drug.html')

@app.route('/price')
def price():
    return render_template('price.html')  

# Get the absolute path to the model files
drug_model_path = os.path.join('datasets', 'drugclassmodel.pkl')
regress_model_path = os.path.join('datasets', 'priceregmodel.pkl')

# Load the models
drug_model = joblib.load(drug_model_path)
regress_model = joblib.load(regress_model_path)

# Mapping for the drug labels
drug_labels = {4: 'drugA', 3: 'drugB', 2: 'drugC', 1: 'drugX', 0: 'DrugY'}

#prediction route
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bp = int(request.form['bp'])
        cholesterol = int(request.form['cholesterol'])
        na_to_k = float(request.form['na_to_k'])

        # Make a prediction
        #input_data = np.array([[age, sex, bp, cholesterol, na_to_k]])
        # Convert data to a format suitable for prediction
        #input_data = np.array([float(request.form['age']),
        #               float(request.form['sex']),
        #               float(request.form['bp']),
        #               float(request.form['cholesterol']),
        #               float(request.form['na_to_k'])]).reshape(1, -1)
        
        # Make a prediction
        input_data = np.array([[age, sex, bp, cholesterol, na_to_k]])
        prediction = drug_model.predict(input_data)[0]
        #predicted_drug = drug_labels[prediction[0]]


        #prediction = model.predict(input_data)
        #predicted_drug = drug_labels[prediction[0]] ikaw lang pala ang sagabal hayp ka

        return render_template('result.html', drug=prediction)
    

# Load the saved MLPRegressor model
model_filename = 'datasets/priceregmodel.pkl'
loaded_model = joblib.load(model_filename)

# Load the training data and get feature names after one-hot encoding
train_filename = 'datasets/regression/train.csv'
train_df = pd.read_csv(train_filename)

# Features and target variable
X_train = train_df[['battery_power', 'clock_speed', 'dual_sim', 'four_g', 'mobile_wt', 'n_cores', 'px_height', 'px_width', 
                    'ram', 'talk_time', 'touch_screen', 'wifi']]
Y_train = train_df['price_range']

# One-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=['dual_sim', 'four_g', 'touch_screen', 'wifi'], drop_first=True)

# Feature names after one-hot encoding
feature_names = X_train_encoded.columns.tolist()

@app.route("/predict_regression", methods=['POST'])
def predict_regression():
    if request.method == 'POST':
        # Get user input from the form
        battery_power = float(request.form['battery_power'])
        clock_speed = float(request.form['clock_speed'])
        dual_sim = int(request.form['dual_sim'])
        four_g = int(request.form['four_g'])
        mobile_wt = float(request.form['mobile_wt'])
        n_cores = float(request.form['n_cores'])
        px_height = float(request.form['px_height'])
        px_width = float(request.form['px_width'])
        ram = float(request.form['ram'])
        talk_time = float(request.form['talk_time'])
        touch_screen = int(request.form['touch_screen'])
        wifi = int(request.form['wifi'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'battery_power': [battery_power],
            'clock_speed': [clock_speed],
            'dual_sim': [dual_sim],
            'four_g': [four_g],
            'mobile_wt': [mobile_wt],
            'n_cores': [n_cores],
            'px_height': [px_height],
            'px_width': [px_width],
            'ram': [ram],
            'talk_time': [talk_time],
            'touch_screen': [touch_screen],
            'wifi': [wifi],
        })

        # Perform the same preprocessing steps as in the Jupyter Notebook
        # One-hot encoding for categorical variables
        input_data_encoded = pd.get_dummies(input_data, columns=['dual_sim', 'four_g', 'touch_screen', 'wifi'], drop_first=True)

        # Ensure feature names match
        input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)

        # Make a prediction using the loaded and now fitted model
        prediction = loaded_model.predict(input_data_encoded)[0]
        
        # Round the prediction to the nearest whole number
        rounded_prediction = round(prediction)

        # Convert the prediction to positive
        rounded_prediction = abs(rounded_prediction)
        
        # Create a mapping for the price ranges
        price_range_mapping = {
            0: '< Php 1000',
            1: 'Php 1000 - Php 5000',
            2: 'Php 5000 - Php 15000',
            3: '> Php 15000'
        }

        # Get the equivalent price range for the rounded prediction
        equivalent_prediction = price_range_mapping.get(rounded_prediction, 'Unknown')

        return render_template('priceresult.html', price=equivalent_prediction)


if __name__ == '__main__':
    app.run(debug=True)
