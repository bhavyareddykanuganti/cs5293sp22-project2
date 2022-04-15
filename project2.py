##Install Packages##
import json
import argparse
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

###ArgumentParser####
def main(args):

    if args.ingredient:
        ing_list = []
        for i in args.ingredient:
            ing_list.append(i)
    ing_string=' '.join(ing_list)

    project2(ing_string, args.N)

#####Loading the file######
def project2(ing_string, N):
    file = open('C:/Users/Bhavya/Downloads/yummly.json')
    data = json.load(file)
    file.close()
    cuisine_p = pd.read_json("C:/Users/Bhavya/Downloads/yummly.json")

#####Convert to dataframe###
    df = pd.DataFrame(cuisine_p)
    a = df.loc[:,"ingredients"]
    for i in a:
        z=''.join(i)
    df['ingredientsss'] = [','.join(map(str, l)) for l in df['ingredients']]
    df.ingredients = df.ingredientsss
    df = df.drop(columns="ingredientsss")
    predicted_cusine, test, ing, df = LogReg(df, ing_string)
    df, sort_sim_score = similarity(predicted_cusine,test, ing, df)
    df = df.iloc[1:, :]
    ####Cuisine, ID and Similarity score###
    x = []
    for i in range(N):
        dict = {
            "id": int(df['id'].iloc[i]),
            "score": df['cosine_values'].iloc[i]
        }
        x.append(dict)
    ####OUTPUT#####
    output = {
        "cuisine": predicted_cusine[0],
        "score": sort_sim_score[0],
        "closest": x
    }
    print(json.dumps(output, indent=1))


###label encoding, Vectorizing and TfidfVectorizer#####
def LogReg(df, ing_string):
    labelencoder = LabelEncoder()
    Vectorizer1 = TfidfVectorizer()
    df['values'] = labelencoder.fit_transform(df['cuisine'])
    ingts = df.loc[:,"ingredients"]
    ing = Vectorizer1.fit_transform(ingts).toarray()
    test = Vectorizer1.transform([ing_string]).toarray()
    model = LogisticRegression(max_iter=100000)
    model.fit(ing,df['values'])
    x = model.predict(Vectorizer1.transform([ing_string]).toarray())
    y = labelencoder.inverse_transform(x)
    return y, test, ing, df

def similarity(x,test, ing, df):
    similarity_score_list = cosine_similarity(test[0:1], ing)
    df['cosine_values'] = similarity_score_list[0]
    df = df.sort_values(by=['cosine_values'],ascending=False)
    sort_sim_score=sorted(similarity_score_list[0],reverse=True)
    return df, sort_sim_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient", type=str, required=True, action='append', help="predict")
    parser.add_argument("--N", type=int, required=True, help="n_neighbours")
    args = parser.parse_args()
    main(args)