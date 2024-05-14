import os
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

symptoms_dict = {'itching': 0, 'skin rash': 1, 'nodal skin eruptions': 2, 'continuous sneezing': 3, 'shivering': 4, 'chills': 5, 'joint pain': 6, 'stomach pain': 7,'acidity': 8, 'ulcers on tongue': 9, 'muscle wasting': 10, 'vomiting': 11, 'burning micturition': 12, 'spotting  urination': 13, 'fatigue': 14, 'weight gain': 15, 'anxiety': 16, 'cold hands and feets': 17, 'mood swings': 18, 'weight loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches in throat': 22, 'irregular sugar level': 23, 'cough': 24, 'high fever': 25, 'sunken eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish skin': 32, 'dark urine': 33, 'nausea': 34, 'loss of appetite': 35, 'pain behind the eyes': 36, 'back pain': 37, 'constipation': 38, 'abdominal pain': 39, 'diarrhoea': 40, 'mild fever': 41, 'yellow urine': 42, 'yellowing of eyes': 43, 'acute liver failure': 44, 'swelling of stomach': 45, 'swelled lymph nodes': 46, 'malaise': 47, 'blurred and distorted vision': 48, 'phlegm': 49, 'throat irritation': 50, 'redness of eyes': 51, 'sinus pressure': 52, 'runny nose': 53, 'congestion': 54, 'chest pain': 55, 'weakness in limbs': 56, 'fast heart rate': 57, 'pain during bowel movements': 58, 'pain in anal region': 59, 'bloody stool': 60, 'irritation in anus': 61, 'neck pain': 62, 'dizziness': 63, 'cramps': 64, 'bruising': 65, 'obesity': 66, 'swollen legs': 67, 'swollen blood vessels': 68, 'puffy face and eyes': 69, 'enlarged thyroid': 70, 'brittle nails': 71, 'swollen extremeties': 72, 'excessive hunger': 73, 'extra marital contacts': 74, 'drying and tingling lips': 75, 'slurred speech': 76, 'knee pain': 77, 'hip joint pain': 78, 'muscle weakness': 79, 'stiff neck': 80, 'swelling joints': 81, 'movement stiffness': 82, 'spinning movements': 83, 'loss of balance': 84, 'unsteadiness': 85, 'weakness of one body side': 86, 'loss of smell': 87, 'bladder discomfort': 88, 'foul smell of urine': 89, 'continuous feel of urine': 90, 'passage of gases': 91, 'internal itching': 92, 'toxic look (typhos)': 93, 'depression': 94, 'irritability': 95, 'muscle pain': 96, 'altered sensorium': 97, 'red spots over body': 98, 'belly pain': 99, 'abnormal menstruation': 100, 'dischromic  patches': 101, 'watering from eyes': 102, 'increased appetite': 103, 'polyuria': 104, 'family history': 105, 'mucoid sputum': 106, 'rusty sputum': 107, 'lack of concentration': 108, 'visual disturbances': 109, 'receiving blood transfusion': 110, 'receiving unsterile injections': 111, 'coma': 112, 'stomach bleeding': 113, 'distention of abdomen': 114, 'history of alcohol consumption': 115, 'fluid overload.1': 116, 'blood in sputum': 117, 'prominent veins on calf': 118, 'palpitations': 119, 'painful walking': 120, 'pus filled pimples': 121, 'blackheads': 122, 'scurring': 123, 'skin peeling': 124, 'silver like dusting': 125, 'small dents in nails': 126, 'inflammatory nails': 127, 'blister': 128, 'red sore around nose': 129, 'yellow crust ooze': 130}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def initializeDTC():
    # Load the dataset
    data = pd.read_csv("Data/Training.csv")

    # Separate features (X) and target variable (y)
    X = data.iloc[:, :-1]  # Select all columns except the last one
    y = data.iloc[:, -1]   # Select only the last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    dtc = DecisionTreeClassifier(ccp_alpha=0.01)
    dtc = dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)
    
    evaluateModel(y_test, predictions, "dt")

    le = LabelEncoder()
    le.fit(y)
    Y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    dtc = DecisionTreeClassifier(ccp_alpha=0.01)
    dtc = dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame({'Predictions': predictions})

    pickle.dump(dtc, open("models/dtc.pkl", "wb"))

    return dtc

def initializeRF():
    # Load the dataset
    data = pd.read_csv("Data/Training.csv")

    # Separate features (X) and target variable (y)
    X = data.iloc[:, :-1]  # Select all columns except the last one
    y = data.iloc[:, -1]   # Select only the last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf = rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    evaluateModel(y_test, predictions, "rf")

    le = LabelEncoder()
    le.fit(y)
    Y = le.transform(y)

    print(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf = rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame({'Predictions': predictions})

    pickle.dump(rf, open("models/rf.pkl", "wb"))

    return rf

def getPrediction(user_symptoms, model):
    check_file = os.path.exists("models/" + model + ".pkl")

    if check_file:
        print('file '+ model)
        with open("models/"  + model + ".pkl", "rb") as file:
            model = pickle.load(file)
    else:
        model = initializeDTC()

    prediction = diseases_list[model.predict(user_symptoms)[0]]
    return prediction

def getDescription(disease):
    description = pd.read_csv("Data/description.csv")

    desc = description[description['Disease'] == disease]['Description']
    desc = " ".join([w for w in desc])
    # print(desc)
    return desc

def getPrecautions(disease):
    precautions = pd.read_csv("Data/precaution.csv")

    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    # print(pre)
    return pre

def getMedications(disease):
    medications = pd.read_csv("Data/medications.csv")

    med = medications[medications['Disease'] == disease]['Medication']
    med = [med for med in med.values]
    # print(med)
    return med

def evaluateModel(y_true, y_pred, model):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    classification_rep = classification_report(y_true, y_pred)

    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred))

    # Open a text file in write mode
    with open(model + ".txt", "w") as file:
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
    processed_input = np.zeros(len(symptoms_dict))
    
    for item in input:
        processed_input[symptoms_dict[item]] = 1
    return processed_input.reshape(1, -1)

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    data = request.get_json()
    symptoms = data.get('symptoms')
    print(symptoms)


    # Split the user's input into a list of symptoms (assuming they are comma-separated)
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    # Remove any extra characters, if any
    user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

    for s in user_symptoms:
        if s not in symptoms_dict:
            message = "Please make sure the spellings are correct"
            return jsonify({'success': False,'message': message})
        
    processed_symptoms = processInput(set(user_symptoms))
    disease = getPrediction(processed_symptoms, "rf")

    description = getDescription(disease)

    precautions = list(getPrecautions(disease))
    print(disease)
    print(precautions)

    my_precautions = []
    for i in precautions[0]:
        my_precautions.append(i.capitalize())

    medications = getMedications(disease)
    print(medications)

    return jsonify({'success': True,'disease': disease, 'description' : description, 'precautions': my_precautions, 'medications': medications})

@app.route('/report')
def report():
    return render_template("reportPage.html")

@app.route('/diseases')
def disease():
    # Read the CSV file
    data = pd.read_csv("Data/description.csv")
    # Access the 'Disease' column as a list
    disease_list = data['Disease'].tolist()

    # Access the 'Description' column as a list
    description_list = data['Description'].tolist()

    # Print both lists
    # print("Disease List:", disease_list)
    # print("Description List:", description_list)
    return render_template("diseasesPage.html", diseases = diseases_list, descriptions = description_list)

@app.route('/about')
def about():
    return render_template("aboutPage.html")

@app.route('/home')
def contact():
    return render_template("homePage.html")

@app.route('/display/<filename>')
def display_image(filename):
    print(filename)
    return redirect(url_for('static', filename= 'images/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500, debug=True)