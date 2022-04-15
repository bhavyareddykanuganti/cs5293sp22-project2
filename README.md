# Cuisine Predictor

## Author

### Bhavya Reddy Kanuganti

Email: bhavya.reddy.kanuganti-1@ou.edu

## Project Description

The aim of the project is to take a list of ingredients from command line
and predict the type of cuisine and N similar meals. 

## Dataset

The data sets for this project is provided by Yummly.com. It contains sets of id, 
ingredient and cuisine, and the dile is in json format.

## Packages Installed

- json
- argparse
- pandas
- numpy
- sklearn
  - sklearn.feature_extraction.text import TfidfVectorizer
  - sklearn.preprocessing import LabelEncoder
  - sklearn.linear_model import LogisticRegression
  - sklearn.neighbors import KNeighborsClassifier
  - sklearn.metrics.pairwise import cosine_similarity
# Files Description
## project2.py
The json dataset has to be loaded and read. This file is then
converted into a dataframe with "id", "cuisine", "ingredients"
columns. The ingredients in the dataframe are of type list, this
column is converted to string format and is replaced in the dataframe.

Label encoder has been used to convert the cuisine to numeric 
format so that it can be machine-readable. The column of label encoded 
cusine values are added to the dataframe.

Logistic regression is a classification approach in machine learning
which is used to train and predict the categorical dependent variable. 
The LogisticRegression.fit() function in this project is being used to 
model according to the ingredients and the label encoded cuisine values.
The user enters the ingredients through the commad line. Term frequency-inverse 
document frequency(TfidfVectorizer)has been used 
to transform the user input ingredients into feature vector that can be
fed into a predictor. The cuisine is predicted and using cosine_similarity
method the similarity score is found.
The similarity score for each meal has been added to the dataframe and
have been sorted using "df.sort_values(by=['cosine_values'],ascending=False)".
Now using a for loop the top-N similar cuisne ids can be generated.
json.dumps() has been used to print the output in the required format.

## test_project2.py
## Execution
The following command has to be used to run the project2.py file:
pipenv run python project2.py --N 7 --ingredient "chili powder" --ingredient "crushed red pepper flakes" --ingredient paprika --ingredient "ground black pepper" --ingredient "dried oregano"

### Output
This output shows the closest cuisine and second the top-N cuisine dishes.
{
 "cuisine": "cajun_creole",
 "score": 0.8524293248457364,
 "closest": [
  {
   "id": 10276,
   "score": 0.8300989368535436
  },
  {
   "id": 37038,
   "score": 0.8274282066307155
  },
  {
   "id": 13296,
   "score": 0.8103487221368857
  },
  {
   "id": 2298,
   "score": 0.7716439771568867
  },
  {
   "id": 38677,
   "score": 0.7455082566480353
  },
  {
   "id": 16366,
   "score": 0.7358876970558029
  },
  {
   "id": 19327,
   "score": 0.7246516366749831
  }
 ]
}

## External links used
https://realpython.com/logistic-regression-python/#when-do-you-need-classification
https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd