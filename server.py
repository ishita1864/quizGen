from flask import Flask, render_template
from pdfToTextConv import pdfconv
from questiongenerator import genques
from flask_cors import CORS, cross_origin

from questiongenerator import remove_stopwords
from questiongenerator import get_nouns_multipartite
from questiongenerator import tokenize_sentences
from questiongenerator import get_sentences_for_keyword
from questiongenerator import get_distractors_wordnet
from questiongenerator import get_wordsense
from questiongenerator import get_distractors_conceptnet



app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
# if __name__ == '__main__':
#     app.run()




@app.route('/convpdf')
@cross_origin()
def convpdf():
    print("working")
    pdfconv()
    return 'text from pdf comes off'



# @app.route('/quesgen')
# @cross_origin()
# def quesgen():
#     print("quesgen working")
#     txt= pdfconv()
#     return 'pdf comes off'


@app.route('/genques')
@cross_origin()
def questions():
    print("working")
    q=genques("new.txt")
    # print (q)
    f = open('generatedquestions.txt', 'w')
    for t in q:
        f.write(' '.join(str(s) for s in t) + '\n')

    return 'ques gen comes off'



@app.route('/quiz')
def quiz():
    print("dis works")
    return 'quiz tyme bois'
