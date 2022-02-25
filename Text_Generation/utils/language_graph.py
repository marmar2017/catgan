import pandas as pd
import csv
from textblob import TextBlob
from polyglot.detect import Detector
from os import path
import random
import seaborn as sns
import matplotlib.pyplot as plt

def is_english_sentence(sentence):
    try:
        language = Detector(sentence, quiet=True).languages[0]
        return language.code
    except:
        return ''
dataset_path = '/home/user1/Ru_experiement/TextGAN-PyTorch/dataset'
df = pd.read_csv(path.join(dataset_path, 'reviews.csv'))
res = {'language code':[]}
for i in range(len(df)):
    review, label = df['Review'][i], df['Label'][i]
    # using lower and upper method to judge whether it is alphabet
    temp = is_english_sentence(review)
    res['language code'].append(temp)
r = pd.DataFrame(res)
data = r.groupby('language code')['language code'].count()
data.to_csv('lang_dis.csv')