from flask import Flask, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS, cross_origin
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

vader = SentimentIntensityAnalyzer()
cred = credentials.Certificate("cubstart-sentiment-analysis-firebase-adminsdk-uaisd-9dbc608b30.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


@app.route("/")
@cross_origin()
def home():
    return "Welcome to Cubstart! This our backend that we'll be using to analyze some text."


# localhost:5000/api/analyze_sentiment/test-id?sentence=i am feeling great today
@app.route("/api/analyze_sentiment/<document_id>", methods=["GET","POST"])
@cross_origin()
def analyze_text(document_id):
    text = request.args.get("sentence")
    sentiment = vader.polarity_scores(text)['compound']
    result = {"sentiment": sentiment}

    doc = db.collection('sentiment-data').document(document_id)
    doc.set(result, merge=True)

    return {"success": 'true', "sentiment": sentiment}


if __name__ == "__main__":
    app.run(debug=True, port=3000)
