import pandas as pd
from sklearn.linear_model import LinearRegression
import os

class MarksPredictor:
    def __init__(self, csv_file="semester_data.csv"):
        self.csv_file = csv_file
        self.model = LinearRegression()
        self.initialize_data_file()

    def initialize_data_file(self):
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=['semester', 'marks', 'subject', 'year'])
            df.to_csv(self.csv_file, index=False)

    def get_all_semesters(self):
        if not os.path.exists(self.csv_file):
            return []
        return pd.read_csv(self.csv_file).to_dict('records')

    def add_semester_data(self, semester_data):
        df = pd.DataFrame([semester_data])
        if os.path.exists(self.csv_file):
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_file, index=False)

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
        return round(predicted_marks, 2)

    def get_semester_stats(self):
        if not os.path.exists(self.csv_file):
            return None
        
        data = pd.read_csv(self.csv_file)
        if len(data) == 0:
            return None
        
        return {
            "average": round(data["marks"].mean(), 2),
            "highest": round(data["marks"].max(), 2),
            "lowest": round(data["marks"].min(), 2),
            "total_semesters": len(data)
        }

# Create an instance of the model for use in Flask
predictor = MarksPredictor()