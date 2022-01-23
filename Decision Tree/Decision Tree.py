import sys
import pandas
import math
import time
import argparse
from sklearn.linear_model import LogisticRegression


#DATASET: heart.csv = https://www.kaggle.com/zhaoyingzhu/heartcsv
sys.argv = ["heart.csv", "-d" , "3"]

        
class treeNode:
    def __init__(self):
        self.feature=""
        self.isLeaf=False
        self.predictor=None # if leafNode , predictor is set
        self.childs={} 
        self.level=None #indicates which level node belongs to 

class logisticRegressorClassifier:
    def __init__(self, trainingData, target):
        self.oneTarget = target
        if target  == None:
            self.X=trainingData.iloc[:, :-1].values
            self.y=trainingData.iloc[:, -1].values
            self.clf=LogisticRegression(random_state=0, max_iter=10000).fit(self.X,self.y)
        
    def predict(self, sample):
        if self.oneTarget != None:
            prediction = self.oneTarget
        else:
            prediction = self.clf.predict(sample[2:-1])
        return prediction

def infoGain(dataset, feature, target):
    ent = entropy(dataset, target)
    sum=0
    for i in dataset[feature].unique():
        sv_over_s = dataset[feature].value_counts().to_frame()[feature][i]/len(dataset)
        sum+=sv_over_s*entropy(dataset.loc[dataset[feature] == i], target)
    return ent - sum        

def entropy(dataset, target):
    entropy = 0

    for i in dataset[target].unique():
        pi=dataset[target].value_counts().to_frame()[target][i]/len(dataset)
        entropy += -pi*math.log(pi,2)        

    return entropy

def selectBestFeature(dataset, features, target):
    highestInfoGain = -1
    feature=""
    for i in features:
        informationGain = infoGain(dataset,i,target)
        if informationGain > highestInfoGain:
            highestInfoGain = informationGain
            feature = i
    return feature

#value is the parent nodes feature value
def buildDecisionTree(trainingData, features, target, node, value, maxDepth): 
    #root node selection
    if node == None:
        root = treeNode()
        root.feature=selectBestFeature(trainingData, features, target)
        root.level = 0
        if(len(features)==1 or maxDepth == 0):
            for i in trainingData[root.feature].unique():
                newLeafNode = treeNode()
                newLeafNode.isLeaf = True
                trData=trainingData.loc[trainingData[root.feature] == i]
                if len(trData[target].unique()) == 1:
                    #logicticRegression Classifier can not trained with a dataset with only one target value
                    #in this cases, predict target value instead of training a classifier for prediction
                    newLeafNode.predictor = logisticRegressorClassifier(None,trData[target].unique()[0])                
                else:
                    newLeafNode.predictor = logisticRegressorClassifier(trData, None)
    
                root.childs[i] = newLeafNode

            return root
        
        else:
            features=features.drop(root.feature)
            for i in trainingData[root.feature].unique():
                newTrainingData=trainingData.loc[trainingData[root.feature]==i]
                buildDecisionTree(newTrainingData, features, target, root, i, maxDepth)
    else:
                
        if (node.level == maxDepth or len(features)==0):
            for i in trainingData[node.feature].unique():
                newLeafNode = treeNode()
                newLeafNode.isLeaf = True
                trData=trainingData.loc[trainingData[node.feature] == i]
                if len(trData[target].unique()) == 1:
                    newLeafNode.predictor = logisticRegressorClassifier(None,trData[target].unique()[0])
                    
                else:
                    newLeafNode.predictor = logisticRegressorClassifier(trData, None)
                    
                node.childs[i]= newLeafNode
                
            return
        
        else: 
            for i in trainingData[node.feature].unique():
                newTrData=trainingData.loc[trainingData[node.feature]==i]
                if len(newTrData[target].unique()) == 1:
                    newLeafNode=treeNode()
                    newLeafNode.isLeaf=True
                    newLeafNode.predictor = logisticRegressorClassifier(None,newTrData[target].unique()[0])
                    node.childs[i] = newLeafNode
                else:
                    newNode=treeNode()
                    newNode.level=node.level+1
                    newNode.feature = selectBestFeature(newTrData,features,target)
                    node.childs[i] = newNode
                    #grow tree recursively with corresponding subset of dataset and remaining features
                    buildDecisionTree(newTrData, features.drop(newNode.feature), target, newNode , i ,maxDepth)
            return
    return root


def classifySamplesDT(rootNode, samples):        
    predictions=[]
    for row in samples.itertuples():
        node = rootNode    
        while not node.isLeaf:
            if getattr(row,node.feature) in node.childs:
                node = node.childs[getattr(row,node.feature)]

            #sample feature value not present in the branches(feature Values) of node, then go for nearest featureValue
            else:
                min = max(node.childs)+getattr(row,node.feature)
                for featureValue in node.childs:
                    nearest = abs(featureValue - getattr(row,node.feature))
                    if nearest < min:
                        min = nearest
                        feature=featureValue                
                node = node.childs[feature]
                                      
        predictions.append(node.predictor.predict(row))
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

        root = buildDecisionTree(trainingData, features, target, None, None, maxDepth)
        dtPredictions = classifySamplesDT(root , testData)

        

        dtLogErr+=sum(abs(testData[target].values-dtPredictions))/len(testData)
       
        
    return dtLogErr/k


start=time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d" , action="store")
    maxDepth = parser.parse_args().d

    data = pandas.read_csv(sys.argv[0])
    #for another dataset format which first row is not index
    if data.columns[0] != "Unnamed: 0":
        data.insert(0,"Unnamed: 0", range(1,len(data)+1))

    features=data.columns[1:len(data.columns)-1]
    target=data.columns[len(data.columns)-1]

    
    dtError = kFoldValidation(data , 5, features, target, maxDepth)
    print("DTLog:" + str(dtError))
    
    
print("Time Elapsed: ")
print(time.time()-start)

