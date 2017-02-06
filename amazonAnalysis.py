import csv
import re
import string
import pandas as pd
from nltk.probability import FreqDist as nF
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from nltk.tokenize import word_tokenize
#List for stopwords
stopWords=[]
dataSet=[]
decisionAttributes=[]
#regular expressions
tokens_re=""

def initializeSystem():
    print "Preparing stop words."
    stop = stopwords.words('english') + punctuation + ['rt', 'via','i\'m','us','it']
    for x in stop:
    	stopWords.append(stemmer.stem(lemmatiser.lemmatize(x, pos="v")))

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else stemmer.stem(lemmatiser.lemmatize(token.lower(), pos="v")) for token in tokens]
    return tokens

def processString(string):
    terms_all = [term for term in preprocess(string)]
    terms_stop = [term for term in preprocess(string) if term not in stopWords and len(str(term)) >1 and not term.isnumeric()]
    return terms_stop

#Function to load datasets
def loadFile(filePath):
    print "Loading the dataset..."
    fileRead = open(filePath,"r")
    reader=csv.reader(fileRead,dialect='excel')
    for row in reader:
        temp=(row[1],row[-1])
        dataSet.append(temp)
		
		
#Function to prepare sparse matrix
def prepareSparseMatrix(convertedReviews):
    sparseMatrix=[]
    for cr in convertedReviews:
        newCr=[0]*len(decisionAttributes)
        for word in cr:
            if word in decisionAttributes:
                index=decisionAttributes.index(word)
                newCr[index]+=1
            else:
                pass
        sparseMatrix.append(newCr)
    return sparseMatrix

def convertReviews(reviews):
    convertedReviews=[]
    for a in reviews:
        convertedReviews.append(processString(str(a).lower()))
    return convertedReviews

#Function to get decision attrubites
def getDecisionAttributes(convertedReviews) :
    toCount=[]
    for a in convertedReviews:
        toCount.append(" ".join(a))
    str1=""
    for a in toCount:
        str1+="".join(a)
    x=Counter(str1.split(" "))
    for (k,v) in x.most_common(min(500,len(x))):
        decisionAttributes.append(k)
#    return decisionAttributes


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    r'<[^>]+>', # HTML tags
    r"(?:[a-z][a-z\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
#stemmer from nltk library called
stemmer = PorterStemmer()
#lemmatizer from mltk library called
lemmatiser = WordNetLemmatizer()
#initializing the system
initializeSystem()

print "Please wait! Analysing the dataset.."

loadFile("./DataSet/AmazonDataSet/amazon_baby_train.csv")
trainDataFeaturesReviews = pd.DataFrame(dataSet,columns=["review","rating"])
targetRating = (trainDataFeaturesReviews['rating']).reshape(-1,1)
targetReview = trainDataFeaturesReviews['review']

loadFile("./DataSet/AmazonDataSet/amazon_baby_test.csv")
testDataFeaturesReviews = pd.DataFrame(dataSet,columns=["review","rating"])
testReview = testDataFeaturesReviews['review']
testRating = testDataFeaturesReviews['rating']

print "Preprocessing the data set..."
trainReviews = convertReviews(targetReview)
getDecisionAttributes(trainReviews)


#Creating sparse matrix for training data and test data
trainSparseMatrix = prepareSparseMatrix(trainReviews)
testSparseMatrix=prepareSparseMatrix(convertReviews(testReview))


dataFeatures = pd.DataFrame(trainSparseMatrix,columns=decisionAttributes)
testDataFeatures = pd.DataFrame(testSparseMatrix,columns=decisionAttributes)
print dataFeatures
#Calling decision tree function to classify the data.
dtree = DecisionTreeClassifier(min_samples_split=5, random_state=9)
dtree.fit(dataFeatures,targetRating)
print "Decision tree is ready"

import subprocess
with open("AmazonDT.dot", 'w') as f:
	export_graphviz(dtree, out_file=f,feature_names=trainSparseMatrix)
command = ["dot", "-Tpng", "dtree.dot", "-o", "dt.png"]

print "Predicting"

s=f=0
predictedRating = list(dtree.predict(testDataFeatures))
for i in range(len(predictedRating)):
	if predictedRating[i] == testRating[i]:
		s+=1
	else :
		f+=1
print "Prediction rate is for the query is: "+str(float(s)/len(predictedRating)*100.0)