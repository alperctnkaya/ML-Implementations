import sys
import pandas
import math
import time
import argparse

#DATASET: heart.csv = https://www.kaggle.com/zhaoyingzhu/heartcsv
sys.argv = ["../heart.csv"]


def classifyNaiveBayes(dataset, features, target, samples):

    predictions=[]
    
    #since it is a binary classification problem, we only have 2 classes
    posExamples=dataset.loc[dataset[target] == 1]
    negExamples=dataset.loc[dataset[target] == 0]
    
    for row in samples.itertuples():
        probPos=1
        probNeg=1
        for feature, featureValue in zip(features,row[2:]):
            if(featureValue in posExamples[feature].values):
                probPos*=posExamples[feature].value_counts().to_frame()[feature][featureValue]/len(posExamples)
            if(featureValue in negExamples[feature].values):
                probNeg*=negExamples[feature].value_counts().to_frame()[feature][featureValue]/len(negExamples)
            
        probPos*=len(posExamples)/len(dataset)
        probNeg*=len(negExamples)/len(dataset)
        
        if probPos>probNeg:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def kFoldValidation(dataset, k , features, target, maxDepth):
    foldSize = math.floor(len(dataset)/k)
    dtLogErr=0
    nbErr=0
    dtPredictions=0
    for i in range(k):
        
        if i == k-1:
            #last fold
            testData=dataset[foldSize*i:]
        else:
            testData=dataset[foldSize*i:foldSize*(i+1)]
            
        trainingData=dataset.drop(testData.index)

        nbPredictions = classifyNaiveBayes(trainingData, features, target, testData)

        nbErr+=sum(abs(testData[target].values-nbPredictions))/len(testData)
        
    return nbErr/k 

start=time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data = pandas.read_csv(sys.argv[0])
    #for another dataset format which first row is not index
    if data.columns[0] != "Unnamed: 0":
        data.insert(0,"Unnamed: 0", range(1,len(data)+1))

    features=data.columns[1:len(data.columns)-1]
    target=data.columns[len(data.columns)-1]

    
    nbError = kFoldValidation(data , 5, features, target, maxDepth)
    print("NB:" + str(nbError))
    
    
print("Time Elapsed: ")
print(time.time()-start)

