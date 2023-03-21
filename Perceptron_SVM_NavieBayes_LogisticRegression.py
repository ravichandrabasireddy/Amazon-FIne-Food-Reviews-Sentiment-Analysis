#Python Version : 3.9
import re
import string
import pandas as pd
import nltk
nltk.download('wordnet',quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import contractions
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.metrics import classification_report

#!pip3 install bs4 
#!pip3 install contractions

def remove_punctuation(review):
    return ''.join([words for words in review if words not in string.punctuation ])

def clean_review(review):    
    review = re.sub(r"http\S+", "", review)
    review = re.sub('<.*?>+', '', review)
    review = re.sub('[^A-Za-z]+', ' ', review)
    review = BeautifulSoup(review, "html.parser").get_text()
    review = contractions.fix(review)
    review= remove_punctuation(review)
    review = re.sub("\S*\d\S*", "", review).strip()
    review = review.lower()
    return review

def average_count(sampled_reviews):
    number_of_sentences=len(sampled_reviews)
    return sum(map(len,sampled_reviews))/number_of_sentences

def metrics(prediction, actual): 
    classificationReport=classification_report(actual, prediction,output_dict=True)
    for i in classificationReport:
        if(i in ['1','2','3','4','5','weighted avg']):
            print(f"{classificationReport[i]['precision']},{classificationReport[i]['recall']},{classificationReport[i]['f1-score']}")


amazon_reviews=pd.read_csv('data.tsv', sep='\t',usecols=['star_rating','review_body'],low_memory=False)
amazon_reviews=amazon_reviews.dropna()
amazon_reviews=amazon_reviews.drop_duplicates()
star_one=amazon_reviews[amazon_reviews.star_rating=='1']
star_one=star_one.sample(n=20000)
star_two=amazon_reviews[amazon_reviews.star_rating=='2']
star_two=star_two.sample(n=20000)
star_three=amazon_reviews[amazon_reviews.star_rating=='3']
star_three=star_three.sample(n=20000)
star_four=amazon_reviews[amazon_reviews.star_rating=='4']
star_four=star_four.sample(n=20000)
star_five=amazon_reviews[amazon_reviews.star_rating=='5']
star_five=star_five.sample(n=20000)
sampled_reviews=pd.concat([star_one,star_two,star_three,star_four,star_five],ignore_index=True)
sampled_reviews.shape

beforeCleaning=average_count(sampled_reviews['review_body'])
sampled_reviews['review_body']=sampled_reviews['review_body'].apply(lambda review:clean_review(review))
afterCleaning=average_count(sampled_reviews['review_body'])
print(f"{beforeCleaning},{afterCleaning}")

stop_words = set(stopwords.words('english'))
beforePreprocessing=average_count(sampled_reviews['review_body'])
sampled_reviews['review_body']= sampled_reviews['review_body'].apply(lambda review: " ".join([word for word in review.split() if word not in stop_words]))

lemmatizer = WordNetLemmatizer()
sampled_reviews['review_body']= sampled_reviews['review_body'].apply(lambda review: ' '.join(lemmatizer.lemmatize(e) for e in nltk.word_tokenize(review)))
afterPreprocessing=average_count(sampled_reviews['review_body'])
print(f"{beforePreprocessing},{afterPreprocessing}")

tf_idf_vect=TfidfVectorizer(ngram_range=(1,3))
final_tf_idf=tf_idf_vect.fit_transform(sampled_reviews['review_body'].values)

xtrain, xtest, ytrain, ytest = train_test_split(final_tf_idf, sampled_reviews['star_rating'], test_size = 0.2)

perceptronModel = Perceptron(random_state=0,n_jobs=-1)
perceptronModel.fit(xtrain, ytrain)
predictions=perceptronModel.predict(xtest)
metrics(predictions, ytest)

SVM = LinearSVC(C=0.1)
SVM.fit(xtrain, ytrain)
predictions=SVM.predict(xtest)
metrics(predictions, ytest)

logisticModel=LogisticRegression(solver='saga',random_state=0,n_jobs=-1,max_iter=200)
logisticModel.fit(xtrain, ytrain)
predictions=logisticModel.predict(xtest)
metrics(predictions, ytest)

naiveBayesModel = naive_bayes.MultinomialNB()
naiveBayesModel.fit(xtrain, ytrain)
predictions=naiveBayesModel.predict(xtest)
metrics(predictions, ytest)
