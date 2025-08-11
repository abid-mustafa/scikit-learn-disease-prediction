import os
from dotenv import load_dotenv
load_dotenv()
model_type = os.getenv("MODEL_TYPE", "rfc")  # default to 'rfc' if not set
port = int(os.getenv("PORT", 5500))  # default to 5500 if not set
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from flask import Flask, jsonify, redirect, render_template, request, url_for

app = Flask(__name__, static_folder='static')

def load_symptoms_dict(csv_path="data/training.csv"):
    df = pd.read_csv(csv_path)
    # Assume symptoms are all columns except last (target)
    symptoms = df.columns[:-1].str.lower()
    return {symptom: idx for idx, symptom in enumerate(symptoms)}

def load_diseases_list(csv_path="data/description.csv"):
    df = pd.read_csv(csv_path)
    # Assuming 'Disease' and 'ID' columns exist, or just map disease names to IDs if you have them
    return {idx: disease for idx, disease in enumerate(df['Disease'].unique())}

symptoms_dict = load_symptoms_dict()
diseases_list = load_diseases_list()

print(symptoms_dict)

def initializeDTC():
    # Load the dataset
    data = pd.read_csv("data/training.csv")

    # Separate features (X) and target variable (y)
    # Convert column names to lowercase
    data.columns = [col.lower() for col in data.columns]
    X = data.iloc[:, :-1]  # Select all columns except the last one
    y = data.iloc[:, -1]   # Select only the last column
    
    # Convert categorical features to numeric using one-hot encoding
    # Initialize LabelEncoder and fit it to the target variable
    le = LabelEncoder()
    le.fit(y)
    Y = le.transform(y)

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Call DecisionTreeClassifier constructor, fit and predict 
    dtc = DecisionTreeClassifier(ccp_alpha=0.01)
    dtc = dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)

    # Call evaluate model function that, save metrics to txt file
    evaluateModel(y_test, predictions, "dtc")

    # save model in pkl file
    with open("models/dtc.pkl", "wb") as f:
        pickle.dump({'model': dtc, 'le': le}, f)

    # return model instance
    return dtc, le

def initializeRFC():
    # Load the dataset
    data = pd.read_csv("data/training.csv")

    # Separate features (X) and target variable (y)
    # Convert column names to lowercase
    data.columns = [col.lower() for col in data.columns]
    X = data.iloc[:, :-1]  # Select all columns except the last one
    y = data.iloc[:, -1]   # Select only the last column

    # Convert categorical features to numeric using one-hot encoding
    # Initialize LabelEncoder and fit it to the target variable
    le = LabelEncoder()
    le.fit(y)
    Y = le.transform(y)

    print(Y)

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Call RandomForestClassifier constructor, fit and predict 
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc = rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)

    # Call evaluate model function that, save metrics to txt file
    evaluateModel(y_test, predictions, "rfc")

    # save model in pkl file
    with open("models/rfc.pkl", "wb") as f:
        pickle.dump({'model': rfc, 'le': le}, f)

    # return model instance
    return rfc, le

def getPrediction(user_symptoms, model_name):
    model_path = f"models/{model_name}.pkl"

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data['model']
        le = data['le']
    else:
        if model_name == "dtc":
            model, le = initializeDTC()
        elif model_name == "rfc":
            model, le = initializeRFC()

    numeric_pred = model.predict(user_symptoms)[0]
    disease = le.inverse_transform([numeric_pred])[0]
    return disease

def getDescription(disease):
    # get description data from csv file
    description = pd.read_csv("data/description.csv")

    # get description for predicted disease
    desc = description[description['Disease'] == disease]['Description']
    desc = " ".join([w for w in desc])
    return desc

def getPrecautions(disease):
    # get precautions data from csv file
    precautions = pd.read_csv("data/precaution.csv")

    # get precautions for predicted disease
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    return pre

def getMedications(disease):
    # get medications data from csv file
    medications = pd.read_csv("data/medications.csv")

    # get medications for predicted disease
    med = medications[medications['Disease'] == disease]['Medication']
    med = [med for med in med.values]
    return med

def evaluateModel(y_true, y_pred, model):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Get classification report
    classification_rep = classification_report(y_true, y_pred)

    # Get Confusion Matrix
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred))

    output_dir = "evaluation_reports"
    os.makedirs(output_dir, exist_ok=True)  # ensure directory exists
    filepath = os.path.join(output_dir, model + ".txt")

    # Open a text file in write mode
    with open(filepath, "w") as file:
        # Write accuracy to the file
        file.write(f'Accuracy: {accuracy}\n\n')
        
        # Write classification report to the file
        file.write("Classification Report:\n")
        file.write(classification_rep)
        file.write("\n\n")
        
        # Write confusion matrix to the file
        file.write("Confusion Matrix:\n")
        file.write(conf_matrix.to_string(index=False))  # Writing confusion matrix without row indices

def processInput(input):
    # Initialize processed_input as an array of size of total number of symptoms with zeros 
    processed_input = np.zeros(len(symptoms_dict))
    
    # Change value to 1 if the symptom is in user input list
    for item in input:
        processed_input[symptoms_dict[item.lower()]] = 1
    return processed_input.reshape(1, -1)

def clean_symptom_input(input_str):
    return ' '.join(input_str.lower().strip().split())

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    # get user symptoms input from front end
    data = request.get_json()
    symptoms = data.get('symptoms')
    print(symptoms)

    # Split the user's input into a list of symptoms (assuming they are comma-separated)
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    # Remove any extra characters, if any
    user_symptoms = [clean_symptom_input(s) for s in user_symptoms]

    # Check user input if it matches the symptoms in dataset 
    for s in user_symptoms:
        if s not in symptoms_dict:
            message = "Please make sure the spellings are correct"
            return jsonify({'success': False,'message': message}), 400
    
    # Process user symptoms to 1s and 0s
    processed_symptoms = processInput(set(user_symptoms))

    # Get predicted disease, change second argument to rfc for Random forest or dtc for Decision Tree
    disease = getPrediction(processed_symptoms, model_type)

    # Get description of predicted disease
    description = getDescription(disease)

    # Get precautions of predicted disease
    precautions = list(getPrecautions(disease))
    print(disease)
    print(precautions)

    my_precautions = []
    for i in precautions[0]:
        my_precautions.append(i.capitalize())

    # Get medications of predicted disease
    medications = getMedications(disease)
    print(medications)

    # Send data to front end
    return jsonify({'success': True,'disease': disease, 'description' : description, 'precautions': my_precautions, 'medications': medications})

@app.route('/diseases')
# render diseasesPage
def disease():
    # Read the CSV file
    data = pd.read_csv("data/description.csv")
    # Access the 'Disease' column as a list
    disease_list = data['Disease'].tolist()

    # Access the 'Description' column as a list
    description_list = data['Description'].tolist()

    # Send data to front end
    return render_template("diseasesPage.html", diseases = diseases_list, descriptions = description_list)

@app.route('/about')
def about():
# render aboutPage
    return render_template("aboutPage.html")

@app.route('/')
def index():
    # Render homePage
    return render_template("homePage.html")

@app.route('/display/<filename>')
def display_image(filename):
    # get logo.png
    return redirect(url_for('static', filename= 'images/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)