import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
from flask import current_app
from flask_server.university.models import Course
from flask_server import db

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

import spacy 
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

btech_pattern = [
    [{"LOWER": "b."}, {"LOWER": "tech"}],
    [{"LOWER": "b"}, {"LOWER": "tech"}],
    [{"LOWER": "btech"}]
]

matcher.add("mtech", [[{"LOWER": "mtech"}], [{"LOWER": "m", "OP": "+"}, {"LOWER": "tech"}]])
matcher.add("btech", btech_pattern)

def course_matcher(sentence):
    with current_app.app_context():
        courses = db.session.query(Course).all()
        
    for course in courses:
        pattern = [{"LOWER": course.name.lower()}]
        matcher.add(course.name, [pattern])

    doc = nlp(sentence)
    matches = matcher(doc)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # 'mtech or btech'
        return string_id
