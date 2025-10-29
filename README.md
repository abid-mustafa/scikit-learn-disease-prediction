# Disease Prediction Web App using Scikit-learn

## Description

A web application that predicts diseases based on user-reported symptoms.

This is a disease prediction web application developed as a part of our 3rd Year Artificial Intelligence Project. Built using Flask for the backend, HTML, CSS, and JavaScript for the frontend, and Scikit-learn for machine learning, it offers predictions, disease descriptions, precautions, and medication suggestions.

---

## Features

* Predicts disease from a list of symptoms.
* Two ML models: Decision Tree (default) and Random Forest.
* Disease descriptions, precautions, and medication suggestions.
* Dynamic frontend connected via Flask APIs.
* Includes model evaluation reports (accuracy, confusion matrix, classification report).

---

## Demo

### Home Page
<kbd> ![Home Page](https://github.com/user-attachments/assets/3735a368-0faf-4735-8825-b44d00fedce6) </kbd>

---

### About Page
<kbd> ![About Page](https://github.com/user-attachments/assets/888c816e-d278-472f-94c5-f80843fa801b) </kbd>

---

### Diseases Page
<kbd> ![Diseases Page](https://github.com/user-attachments/assets/f4f8c7bc-4e59-4dd5-aad1-e85a6852e916) </kbd>

---

### Live Demo
<kbd> ![Live Demo](https://github.com/user-attachments/assets/a931f3d4-f599-40a5-8731-097584bcad33) </kbd>

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/abid-mustafa/scikit-learn-disease-prediction.git
   cd scikit-learn-disease-prediction
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On MacOS/Linux: source venv/bin/activate
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable**
   Create a `.env` file:

   ```env
   MODEL_TYPE=dtc
   PORT=5500
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

   Access the app at `http://localhost:5500`.

---

## Usage

* Navigate to the homepage and enter your symptoms as a comma-separated list.
* The system returns the most likely disease, its description, possible medications, and recommended precautions.
* Explore a list of all supported diseases via the `/diseases` route.

---

## API Endpoints

| Endpoint          | Method | Description                          |
| ----------------- | ------ | ------------------------------------ |
| `/`               | GET    | Homepage                             |
| `/diagnosis`      | POST   | Accepts symptoms, returns prediction |
| `/diseases`       | GET    | Lists all diseases and descriptions  |
| `/about`          | GET    | About the project                    |
| `/display/<file>` | GET    | Serves static images                 |

**POST `/diagnosis` Payload Example:**

```json
{
  "symptoms": "fever, headache, fatigue, cough"
}
```

**POST `/diagnosis` Response Example:**

```json
{
  "success": true,
  "disease": "Common Cold",
  "description": "A viral infection of your nose and throat (upper respiratory tract).",
  "precautions": [
    "Drink plenty of fluids",
    "Take rest",
    "Use nasal sprays",
    "Consult doctor if symptoms worsen"
  ],
  "medications": ["Paracetamol", "Cough Syrup"]
}
```

---

## Configuration

| Variable     | Description                            | Default |
| ------------ | -------------------------------------- | ------- |
| `MODEL_TYPE` | Machine learning model: `dtc` or `rfc` | `dtc`   |
| `PORT`       | Port for Flask server                  | `5500`  |

Modify in `.env` file to switch between models or change port.

---

## Dependencies

Required Python packages (see `requirements.txt`):

* Flask
* numpy
* pandas
* scikit-learn
* python-dotenv

---
## Data Files (CSV)

| Filename          | Description                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `training.csv`    | Symptom columns with binary features and corresponding disease labels for model training.            |
| `description.csv` | Disease names mapped to their detailed descriptions for display.                                     |
| `precaution.csv`  | Precautionary measures associated with each disease.                                                 |
| `medications.csv` | Recommended medications for each disease.                                                            |
| `dataset.csv`     | Raw/example dataset including disease names and symptom presence, used for exploration or reference. |

All CSV files are located in the `data/` directory and are required for training, prediction, and information retrieval in the application.

---

## Troubleshooting

* **Model file not found:** If `models/dtc.pkl` or `models/rfc.pkl` don't exist, they will be auto-generated on first request from training data.
* **Incorrect symptom name:** Input symptoms must exactly match dataset keys (`symptoms_dict`). Spelling and spacing matters.

---

## Contributors

* **Abid Mustafa** - Backend, Machine Learning Integration
* **Abrar Shah** - Frontend (HTML, CSS, JavaScript)

---

## LICENSE
For academic use only. Not licensed for production. Contact project authors for inquiries.
