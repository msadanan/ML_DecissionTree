import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

trainData = pd.read_csv(open('./DataSet/DigitRecognition/optdigits_raining.csv'))
testdata = pd.read_csv(open('./DataSet/DigitRecognition/optdigits_test.csv'))
#trainData = pd.read_csv(open('D:\\Coursework\\ML\\ITCS6156_SLProject\\DigitRecognition\\optdigits_raining.csv'))
#testdata = pd.read_csv(open('D:\\Coursework\\ML\\ITCS6156_SLProject\\DigitRecognition\\optdigits_test.csv'))
features = list(trainData.columns[:-1])
x=trainData['result']
y=trainData[features]
xx=x.reshape(-1,1)
dt = DecisionTreeClassifier(min_samples_split=5, random_state=9)
dt.fit(y,xx)
#print(len(features))
import subprocess
with open("dt.dot", 'w') as f:
	export_graphviz(dt, out_file=f,feature_names=features)

testResult=testdata['result']
command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
testfeatures = list(testdata.columns[:-1])
testFeature=testdata[testfeatures]
actualResult=list(testResult)
predictedData=list(dt.predict(testFeature))
s=f=0
for i in range(len(actualResult)-2):
	if actualResult[i] == predictedData[i]:
		s=s+1
	else:
		f=f+1
print "Prediction rate is "+str(float(s)/len(actualResult)*100.0)
