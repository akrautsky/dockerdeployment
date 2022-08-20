import json
import pickle
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text
import nltk
nltk.download('stopwords')
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from nltk.corpus import stopwords


from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

print(tf.__version__)
app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regression_model.pkl','rb'))
scalar=pickle.load(open('scaler.pkl','rb'))





@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]

    ## CDS
    
    youtube_link = 'https://youtu.be/FRptcX6iDVc' #input("Please enter the video link for presentation..")
    if youtube_link != '':
        v_transcript = transcript(extract_video_id(youtube_link))


    if v_transcript != 'Nosubtitles':
        preds = my_saved_model.predict((np.array([v_transcript]).reshape(1,-1)))[0][0]

    return render_template("home.html",prediction_text=f"The House price prediction is {output} and submission {preds}")


def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
    return "video not on Youtube or Does not exist"

def transcript(video_id):
    try :
        m = YouTubeTranscriptApi.get_transcript(video_id)
        line = []
        new_line=[]
        stopwords_line=""
        for word in m:
#             line.append(word['text'])
            for w in word['text'].split(" "):
                line.append(w)
#             line = line + (word['text'])
        stopwords_line=remove_stopwords(line)
        return stopwords_line
    except :
        return 'Nosubtitles'

def remove_stopwords(word_list):
        processed_word_list = []
        line = ""
        for word in word_list:
            word = word.lower() # in case they arenet all lower cased
            if word not in stopwords.words("english") and len(word) >=3 :
                processed_word_list.append(word)
        for word in processed_word_list:
            line = line + word + " "
        return line   


if __name__=="__main__":
    app.run(debug=True)
   
     