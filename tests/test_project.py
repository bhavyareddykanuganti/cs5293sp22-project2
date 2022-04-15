import pytest
import project2
import json
import argparse
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
file = open('C:/Users/Bhavya/Downloads/yummly.json')
data = json.load(file)
file.close()
cuisine_p = pd.read_json("C:/Users/Bhavya/Downloads/yummly.json")
print(cuisine_p)
df = pd.DataFrame(cuisine_p)
a = df.loc[:,"ingredients"]
for i in a:
    z=''.join(i)
df['ingredientsss'] = [','.join(map(str, l)) for l in df['ingredients']]
df.ingredients = df.ingredientsss
df = df.drop(columns="ingredientsss")

ing_string = 'paprika'
def test_LogReg():
    a,b,c,d  = project2.LogReg(df, ing_string)
    assert a is not None
    assert b is not None
    assert c is not None
    assert d is not None
def test_similarity():
    a, b, c, d = project2.LogReg(df, ing_string)
    x,y = project2.similarity(a,b,c, d)
    assert x is not None
    assert y is not None
