# Disease Prediction Web App using Scikit-learn

## Introduction

This is a disease prediction web application developed as a part of our 3rd Year Artificial Intelligence Project. The system uses supervised machine learning models - **Decision Tree Classifier** and **Random Forest Classifier** - to predict diseases based on symptoms input by the user. The backend is implemented using **Flask**, and the frontend was developed using **HTML**, **CSS**, and **JavaScript**.

Random Forest outperformed the Decision Tree in accuracy and robustness, as documented in the model evaluation text files.

---

## Features

- Predicts disease from a list of symptoms.
- Two ML models: Decision Tree and Random Forest (default).
- Disease descriptions, precautions, and medication suggestions.
- Dynamic frontend connected via Flask APIs.
- Includes model evaluation reports (accuracy, confusion matrix, classification report).

---

## Demo

<p align="center">
 <img src="https://github.com/user-attachments/assets/c018ed5d-e819-413c-b417-2d8cb242c42f" alt="Disease Prediction Demo" width="600"/>
</p>

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abid-mustafa/scikit-learn-disease-prediction.git
   cd scikit-learn-disease-prediction
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable**
   Create a `.env` file:
   ```env
   MODEL_TYPE=rf
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

   Access the app at `http://localhost:5500`.

---

## Usage

- Navigate to the homepage and enter your symptoms as a comma-separated list.
- The system returns the most likely disease, its description, possible medications, and recommended precautions.
- You can explore a list of all supported diseases via the `/diseases` route.

---

## API Endpoints

| Endpoint          | Method | Description                            |
|-------------------|--------|----------------------------------------|
| `/`               | GET    | Homepage                               |
| `/diagnosis`      | POST   | Accepts symptoms, returns prediction   |
| `/diseases`       | GET    | Lists all diseases and descriptions    |
| `/about`          | GET    | About the project                      |
| `/display/<file>` | GET    | Serves static images                   |

**POST `/diagnosis` Payload Example:**
```json
{
  "symptoms": "headache, fatigue, nausea"
}
```

---

## ⚙️ Configuration

Environment variables used:

| Variable     | Description                            | Default |
|--------------|----------------------------------------|---------|
| `MODEL_TYPE` | Machine learning model: `dt` or `rf`   | `dt`    |

You can modify this in the `.env` file to switch between models.

---

## Project Structure

```
.
├── app.py
├── models/
│   ├── dt.pkl
│   └── rf.pkl
├── data/
│   ├── training.csv
│   ├── description.csv
│   ├── precaution.csv
│   └── medications.csv
├── templates/
│   ├── homePage.html
│   ├── diseasesPage.html
│   └── aboutPage.html
├── static/
│   └── images/
├── rf.txt
├── dt.txt
├── .env
└── requirements.txt
```

---

## Dependencies

Required Python packages (see `requirements.txt`):

- Flask
- numpy
- pandas
- scikit-learn
- python-dotenv

---

## Examples

### Input
```
Symptoms: ["fever", "headache", "fatigue", "cough"]
```

### Output
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

## Troubleshooting

- **Model file not found:** If `models/dt.pkl` or `models/rf.pkl` doesn't exist, it will be auto-generated on first request using training data.
- **Incorrect symptom name:** Ensure input symptoms match keys in the dataset (`symptoms_dict`). Spelling matters.
- **Port already in use:** Change the port in `app.py` (`app.run(port=5500)`) if necessary.

---

## Contributors

- **S. M. Abid Mustafa** - Backend, Machine Learning Integration
- **Abrar Shah** - https://github.com/sAbrarShah - Frontend (HTML, CSS, JavaScript) 

---

## License

This project is intended for academic purposes and not licensed for production use. For inquiries, please contact the project authors.
