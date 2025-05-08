from flask import Flask, request, render_template, jsonify
from model import predictor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

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
        
        logger.debug(f"Submitting semester data: {semester_data}")
        predictor.add_semester_data(semester_data)
        return jsonify({"status": "success", "message": "Data saved successfully!"})
    except Exception as e:
        logger.error(f"Error in submit: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/delete/<int:semester>', methods=['DELETE'])
def delete_semester(semester):
    try:
        logger.debug(f"Deleting semester: {semester}")
        success = predictor.delete_semester(semester)
        if success:
            return jsonify({"status": "success", "message": "Semester deleted successfully!"})
        else:
            return jsonify({"status": "error", "message": "Semester not found"}), 404
    except Exception as e:
        logger.error(f"Error in delete: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/predict')
def predict():
    try:
        logger.debug("Training model...")
        predictor.train_model()
        
        logger.debug("Getting prediction...")
        predicted_marks = predictor.predict_next_semester()
        
        logger.debug("Getting stats...")
        stats = predictor.get_semester_stats()
        
        if predicted_marks is None:
            logger.warning("No data available for prediction")
            return jsonify({"status": "error", "message": "No data available for prediction"})
        
        logger.debug(f"Prediction: {predicted_marks}, Stats: {stats}")
        return jsonify({
            "status": "success",
            "predicted_marks": predicted_marks,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/semesters')
def get_semesters():
    try:
        semesters = predictor.get_all_semesters()
        logger.debug(f"Retrieved {len(semesters)} semesters")
        return jsonify({"status": "success", "data": semesters})
    except Exception as e:
        logger.error(f"Error in get_semesters: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)