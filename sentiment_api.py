from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data['text']
        
        result = classifier(text)
        sentiment = result[0]['label']
        
        response = {'sentiment': sentiment}
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
