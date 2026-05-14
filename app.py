
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the combined model and vectorizer
combined_filename = 'combined_sentiment_model_new.joblib'
loaded_combined_objects = joblib.load(combined_filename)
loaded_model = loaded_combined_objects['model']
loaded_vectorizer = loaded_combined_objects['vectorizer']

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        review = data.get('review')
    else:
        review = request.form.get('review')

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    # Convert the review into vector format
    sample_vector = loaded_vectorizer.transform([review])

    # Predict sentiment
    prediction = loaded_model.predict(sample_vector)

    # Convert numeric prediction into text label
    if prediction[0] == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    return jsonify({'review': review, 'predicted_sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
