```python
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and vectorizer
combined_filename = 'combined_sentiment_model_new.joblib'

loaded_combined_objects = joblib.load(combined_filename)

loaded_model = loaded_combined_objects['model']
loaded_vectorizer = loaded_combined_objects['vectorizer']


# Home Page
@app.route('/')
def home():

    return '''
    <html>

        <head>
            <title>Sentiment Analysis</title>
        </head>

        <body style="font-family: Arial; padding: 40px;">

            <h1>Sentiment Analysis App</h1>

            <form action="/predict" method="post">

                <input
                    type="text"
                    name="review"
                    placeholder="Enter your review"
                    style="width:300px; padding:10px;"
                >

                <button type="submit" style="padding:10px;">
                    Predict
                </button>

            </form>

        </body>

    </html>
    '''


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # JSON request (Postman/API)
    if request.is_json:

        data = request.get_json()

        review = data.get('review')

    # Form request (browser form)
    else:

        review = request.form.get('review')

    # Validation
    if not review:

        return jsonify({
            'error': 'No review provided'
        }), 400

    # Convert text into vector
    sample_vector = loaded_vectorizer.transform([review])

    # Predict sentiment
    prediction = loaded_model.predict(sample_vector)

    # Convert output label
    if prediction[0] == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    # If request came from browser form
    if not request.is_json:

        return f'''

        <h1>Prediction Result</h1>

        <p><strong>Review:</strong> {review}</p>

        <p><strong>Sentiment:</strong> {sentiment}</p>

        <a href="/">Try Again</a>

        '''

    # If request came from API/Postman
    return jsonify({

        'review': review,
        'predicted_sentiment': sentiment

    })


# Run Flask App
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
```
