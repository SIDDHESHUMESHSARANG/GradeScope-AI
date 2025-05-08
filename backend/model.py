import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import numpy as np

class MarksPredictor:
    def __init__(self, csv_file="semester_data.csv"):
        # Get the absolute path to the backend directory
        self.backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(self.backend_dir, csv_file)
        self.model = LinearRegression()
        self.initialize_data_file()

    def initialize_data_file(self):
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=['semester', 'marks', 'subject', 'year'])
            df.to_csv(self.csv_file, index=False)

    def get_all_semesters(self):
        if not os.path.exists(self.csv_file):
            return []
        data = pd.read_csv(self.csv_file)
        # Convert DataFrame to list of dictionaries with native Python types
        return data.astype({
            'semester': 'int',
            'marks': 'int',
            'subject': 'str',
            'year': 'int'
        }).to_dict('records')

    def add_semester_data(self, semester_data):
        df = pd.DataFrame([semester_data])
        if os.path.exists(self.csv_file):
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_file, index=False)

    def delete_semester(self, semester):
        if not os.path.exists(self.csv_file):
            return False
        
        data = pd.read_csv(self.csv_file)
        if len(data) == 0:
            return False
        
        # Find and remove the semester
        data = data[data['semester'] != semester]
        
        # Save the updated data
        data.to_csv(self.csv_file, index=False)
        return True

    def train_model(self):
        if not os.path.exists(self.csv_file):
            return None
        
        data = pd.read_csv(self.csv_file)
        if len(data) == 0:
            return None
        
        X = data[["semester"]]
        y = data["marks"]
        self.model.fit(X, y)
        return True

    def predict_next_semester(self):
        if not os.path.exists(self.csv_file):
            return None
        
        data = pd.read_csv(self.csv_file)
        if len(data) == 0:
            return None
        
        future_semester = [[len(data) + 1]]
        predicted_marks = self.model.predict(future_semester)[0]
        # Convert numpy float to Python float
        return float(round(predicted_marks, 2))

    def get_semester_stats(self):
        if not os.path.exists(self.csv_file):
            return None
        
        data = pd.read_csv(self.csv_file)
        if len(data) == 0:
            return None
        
        # Convert numpy types to Python native types
        return {
            "average": float(round(data["marks"].mean(), 2)),
            "highest": int(data["marks"].max()),
            "lowest": int(data["marks"].min()),
            "total_semesters": int(len(data))
        }

# Create an instance of the model for use in Flask
predictor = MarksPredictor()