import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import random

# Initialize Flask app
app = Flask(__name__)

# Load the datasets
condition_data_path = 'condition_data.csv'  # Parameter Kondisi
data_path = 'data.csv'  # Dataset

condition_data = pd.read_csv(condition_data_path)
data = pd.read_csv(data_path)

# Copy the condition_data for preprocessing
df = condition_data.copy()

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'weather', 'situation', 'fashion_style', 'upperwear', 'bottomwear', 'footwear']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features (X) and target variables (y)
X = df[['gender', 'weather', 'situation', 'fashion_style']]
y_upperwear = df['upperwear']
y_bottomwear = df['bottomwear']
y_footwear = df['footwear']

# Split data into training and testing sets
X_train, X_test, y_train_upperwear, y_test_upperwear = train_test_split(X, y_upperwear, test_size=0.2, random_state=42)
X_train, X_test, y_train_bottomwear, y_test_bottomwear = train_test_split(X, y_bottomwear, test_size=0.2, random_state=42)
X_train, X_test, y_train_footwear, y_test_footwear = train_test_split(X, y_footwear, test_size=0.2, random_state=42)

# Initialize RandomForest Classifier
model_upperwear = RandomForestClassifier(random_state=42)
model_bottomwear = RandomForestClassifier(random_state=42)
model_footwear = RandomForestClassifier(random_state=42)

# Train the models
model_upperwear.fit(X_train, y_train_upperwear)
model_bottomwear.fit(X_train, y_train_bottomwear)
model_footwear.fit(X_train, y_train_footwear)

def apply_rules(upperwear_options, bottomwear_options, footwear_options, gender, weather, situation, fashion_style):
    # Get the condition data that matches the input parameters
    matching_conditions = condition_data[
        (condition_data['gender'] == gender) &
        (condition_data['weather'] == weather) &
        (condition_data['situation'] == situation) &
        (condition_data['fashion_style'] == fashion_style)
    ]
    
    if not matching_conditions.empty:
        # Get the valid categories from the condition_data
        valid_upperwear = matching_conditions['upperwear'].str.split(';').values[0]
        valid_bottomwear = matching_conditions['bottomwear'].str.split(';').values[0]
        valid_footwear = matching_conditions['footwear'].str.split(';').values[0]

        # Filter the predicted options to only include valid categories
        upperwear_options = [option for option in upperwear_options if option in valid_upperwear]
        bottomwear_options = [option for option in bottomwear_options if option in valid_bottomwear]
        footwear_options = [option for option in footwear_options if option in valid_footwear]

    return upperwear_options, bottomwear_options, footwear_options

@app.route('/recommend', methods=['POST'])
def recommend_outfit():
    # Get input parameters from the POST request
    input_data = request.json
    gender = input_data['gender']
    weather = input_data['weather']
    situation = input_data['situation']
    style = input_data['fashion_style']
    n_options = input_data.get('n_options', 10)  # Default to 3 options if not provided
    
    # Encode the input parameters
    encoded_input = [
        label_encoders['gender'].transform([gender])[0],
        label_encoders['weather'].transform([weather])[0],
        label_encoders['situation'].transform([situation])[0],
        label_encoders['fashion_style'].transform([style])[0],
    ]
    
    # Get predictions probabilities
    upperwear_probs = model_upperwear.predict_proba([encoded_input])[0]
    bottomwear_probs = model_bottomwear.predict_proba([encoded_input])[0]
    footwear_probs = model_footwear.predict_proba([encoded_input])[0]
    
    # Get the top n options
    upperwear_top_n = upperwear_probs.argsort()[-n_options:][::-1]
    bottomwear_top_n = bottomwear_probs.argsort()[-n_options:][::-1]
    footwear_top_n = footwear_probs.argsort()[-n_options:][::-1]
    
    # Decode the predictions to get category names
    upperwear_options = label_encoders['upperwear'].inverse_transform(upperwear_top_n)
    bottomwear_options = label_encoders['bottomwear'].inverse_transform(bottomwear_top_n)
    footwear_options = label_encoders['footwear'].inverse_transform(footwear_top_n)

    # Apply rule-based filtering based on condition_data.csv
    upperwear_options, bottomwear_options, footwear_options = apply_rules(
        upperwear_options, bottomwear_options, footwear_options, gender, weather, situation, style
    )
    
    # Choose base URL based on gender
    if (gender == 'male'):
        up_url = "https://ik.imagekit.io/dededwi/dataset_image/male/Uppearwear/"
        bot_url = "https://ik.imagekit.io/dededwi/dataset_image/male/Bottomwear/"
        foot_url = "https://ik.imagekit.io/dededwi/dataset_image/male/Footwear/"
    else:
        up_url = "https://ik.imagekit.io/dededwi/dataset_image/female/Upperwear/"
        bot_url = "https://ik.imagekit.io/dededwi/dataset_image/female/Bottomwear/"
        foot_url = "https://ik.imagekit.io/dededwi/dataset_image/female/Footwear/"
    
    # Find corresponding filenames in the data.csv file
    upperwear_files = data[(data['gender'] == gender) & (data['category'].isin(upperwear_options))]['file_name'].tolist()
    bottomwear_files = data[(data['gender'] == gender) & (data['category'].isin(bottomwear_options))]['file_name'].tolist()
    footwear_files = data[(data['gender'] == gender) & (data['category'].isin(footwear_options))]['file_name'].tolist()

    # Shuffle the lists before slicing the top n options
    random.shuffle(upperwear_files)
    random.shuffle(bottomwear_files)
    random.shuffle(footwear_files)
    
    # Construct full URLs and wrap them in lists
    upperwear_urls = [[up_url + filename] for filename in upperwear_files]
    bottomwear_urls = [[bot_url + filename] for filename in bottomwear_files]
    footwear_urls = [[foot_url + filename] for filename in footwear_files]
    
    return jsonify({
        'upperwear': upperwear_urls[:n_options],
        'bottomwear': bottomwear_urls[:n_options],
        'footwear': footwear_urls[:n_options]
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(host="192.168.100.32", port=5000)
