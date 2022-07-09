import nltk 
import numpy as np
import string 
from statistics import mean

def complexity(texts): 
    return np.array([[mean(len(w) for w in wt), len(set(wt)) / len(wt), st, len([w for w in wt if len(w) >= 6]) * 1.0 / len(wt)] for st, wt in 
            zip(
                (mean(len([w.lower() for w in nltk.word_tokenize(s) if w not in string.punctuation]) for s in nltk.sent_tokenize(text)) for text in texts), 
                ([w.lower() for w in nltk.word_tokenize(text) if w not in string.punctuation] for text in texts))
           ]), ["average number of characters per word", "#unique words / #total words", "average number of words per sentence", 'count of "long" words - words with >= 6 letters']
