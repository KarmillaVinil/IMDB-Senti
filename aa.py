import pickle
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
import webbrowser  


nltk.download('punkt')

data = pd.read_csv(r'C:\Users\keert\Desktop\final\imdb\imdb\dataset\Train.csv')



data['text'] = data['text'].apply(word_tokenize)


tfidf_vectorizer = TfidfVectorizer(max_features=5000)  
X_train_tfidf = tfidf_vectorizer.fit_transform(data['text'].apply(' '.join))  
y_train = data['label']

nb_classifier = MultinomialNB()


nb_classifier.fit(X_train_tfidf, y_train)


with open('new_picklee.pkl', 'wb') as file:
    pickle.dump((nb_classifier, tfidf_vectorizer), file)

app = Flask(__name__, template_folder=r'C:\Users\keert\Desktop\final\imdb\imdb\templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        review = request.form['review']
        review_tokens = word_tokenize(review)
        review_tfidf = tfidf_vectorizer.transform([' '.join(review_tokens)])
        sentiment = nb_classifier.predict(review_tfidf)

        if sentiment[0] == 1:
            result = "Positive"
        else:
            result = "Negative"

        return result
    except Exception as e:
        return str(e), 500  


@app.errorhandler(500)
def internal_server_error(e):
    return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
    webbrowser.open('http://127.0.0.1:5000/') 
