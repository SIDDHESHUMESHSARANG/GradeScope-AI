from flask import Flask, request, render_template, jsonify
from model import predictor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    try:
        semester_data = {
            "semester": int(request.form.get("semester")),
            "marks": int(request.form.get("marks")),
            "subject": request.form.get("subject", "General"),
            "year": request.form.get("year", "2024")
        }
        
        predictor.add_semester_data(semester_data)
        return jsonify({"status": "success", "message": "Data saved successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/predict')
def predict():
    try:
        predictor.train_model()
        predicted_marks = predictor.predict_next_semester()
        stats = predictor.get_semester_stats()
        
        if predicted_marks is None:
            return jsonify({"status": "error", "message": "No data available for prediction"})
            
        return jsonify({
            "status": "success",
            "predicted_marks": predicted_marks,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/semesters')
def get_semesters():
    try:
        semesters = predictor.get_all_semesters()
        return jsonify({"status": "success", "data": semesters})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)