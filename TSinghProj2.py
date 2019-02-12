#!/usr/bin/env python
# coding: utf-8

# In[189]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random


# In[190]:


trainingPercent=80
validationPercent=10
testPercent=10
numberOfClusters=15
C_Lambda = 0.03
lr=0.01


# In[191]:


def generateTargetArray(filepath): ##function to create target array from given filepath
    humanObservedTarget=[]
    with open(filepath,'rU') as f:
        reader=csv.reader(f)
        first_row=next(reader)
        count=1
        for row in reader:
            if count==791:
                break
            count+=1
            humanObservedTarget.append(int(row[2]))
           
            
    #print(humanObservedTarget)
    return np.array(humanObservedTarget)



# In[192]:


def shuffdiff(filepath): # function to shuffle the different pair csv file and store the result in a new output file 
    temp=[]             #so that when every time a random combination of different pairs are selected for human observed csv
    temp1=[]
    with open(filepath, 'rU') as f:
        treader=csv.reader(f)
        firstt_row=next(treader)
        temp.append(firstt_row)
        for row in treader:
            temp1.append(row)
            
    random.shuffle(temp1)
    temp.extend(temp1)
    with open("/home/tanmay/Documents/machine learning/proj 2/HumanObserved-Dataset /HumanObserved-Dataset/HumanObserved-Features-Data/output.csv", 'w') as f2:
        writer = csv.writer(f2)
        writer.writerows(temp)
        
    return
        
    
def shuffdiff1(filepath):# function to shuffle the different pair csv file and store the result in a new output file             
                        #so that when every time a random combination of different pairs are selected for GSC csv
    temp=[]
    temp1=[]
    with open(filepath, 'rU') as f:
        treader=csv.reader(f)
        firstt_row=next(treader)
        temp.append(firstt_row)
        for row in treader:
            temp1.append(row)
            
    random.shuffle(temp1)
    temp.extend(temp1)
    with open("/home/tanmay/Documents/machine learning/proj 2/GSC-Dataset/GSC-Dataset/GSC-Features-Data/output.csv", 'w') as f2:
        writer = csv.writer(f2)
        writer.writerows(temp)
        
    return
        


# In[193]:


def flatten(l):   # function to create a list from a list of list matrix
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList 
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    
    
    return flatList


# In[194]:


def generateFeatureArray(sameFilePath,diffFilePath,FeaturesData):##function to create concatanated and subtracted dataset
    humanObservedFeaturesSameConcat=[] #for human observed csv files
    humanObservedFeaturesDiffConcat=[]
    humanObservedFeaturesSameSub=[]
    humanObservedFeaturesDiffSub=[]
    featureDictionary={}
    with open(FeaturesData,'rU') as f:###create a dictionary to store image id as key and feature list as values so 
        reader=csv.reader(f)###that it can be easily searched for the required key values
        first_row = next(reader)
        temp=[]
        for rows in reader:
            for i in range(2,11):
                temp.append(int(rows[i]))
            featureDictionary.update({rows[1]:temp})
            temp=[] 
    
    with open(sameFilePath,'rU') as f1:#reading the same pair file and searching the features in the dictionary and appending
        count=1
        reader1=csv.reader(f1)
        first_row=next(reader1)
        for row in reader1:
            if count==791:
                break
            count+=1
            humanObservedFeaturesSameConcat.append(featureDictionary.get(row[0])+featureDictionary.get(row[1]))
                    
    
    with open(diffFilePath,'rU') as f2:#reading the different pair file and searching the features in the dictionary and appending
        reader2=csv.reader(f2)
        first_row=next(reader2)
        count = 1
        for row in reader2:
            if count==791:
                break
            count+=1
            humanObservedFeaturesDiffConcat.append(featureDictionary.get(row[0])+featureDictionary.get(row[1]))
            
    humanObservedFeaturesSameConcat.extend(humanObservedFeaturesDiffConcat)
    
    with open(sameFilePath,'rU') as f1:#reading the same pair file and searching the features in the dictionary 
        reader1=csv.reader(f1)          #performing subtraction of the features and appending 
        first_row=next(reader1)
        count=1
        for row in reader1:
            if count==791:
                break
            count+=1
            temp3=[0 for i in range(9)]
            temp1=featureDictionary.get(row[0])
            temp2=featureDictionary.get(row[1])
            for i in range(len(temp1)):
                temp3[i]=abs(temp1[i]-temp2[i])
            humanObservedFeaturesSameSub.append(temp3)       
    
    
    with open(diffFilePath,'rU') as f1:   #reading the different pair file and searching the features in the dictionary 
        reader1=csv.reader(f1)            #performing subtraction of the features and appending 
        first_row=next(reader1)
        count1=1
        for row in reader1:
            if count1==791:
                break
            count1+=1    
            temp3=[0 for i in range(9)]
            #print(featureDictionary.get(row[0]))
            temp1=featureDictionary.get(row[0])
            temp2=featureDictionary.get(row[1])
            for i in range(len(temp1)):
                temp3[i]=abs(temp1[i]-temp2[i])
            humanObservedFeaturesDiffSub.append(temp3)
               
    humanObservedFeaturesSameSub.extend(humanObservedFeaturesDiffSub)
    
    humanObservedFeaturesSameConcat = np.asarray(humanObservedFeaturesSameConcat)#concatanating same and different pair 
    humanObservedFeaturesSameSub=np.asarray(humanObservedFeaturesSameSub)
    
    
    humanObservedFeaturesSameConcat = np.transpose(humanObservedFeaturesSameConcat)#performing transpose of concatanated and subtracted matrix
    humanObservedFeaturesSameSub=np.transpose(humanObservedFeaturesSameSub)
    
    return np.asarray(humanObservedFeaturesSameConcat),np.asarray(humanObservedFeaturesSameSub)   
            


# In[195]:


def generateGSCFeatureArray(sameFilePath,diffFilePath,FeaturesData):##function to create concatanated and subtracted dataset
    GSCFeaturesSameConcat=[]                                         #for GSC dataset
    GSCFeaturesDiffConcat=[]
    GSCFeaturesSameSub=[]
    GSCFeaturesDiffSub=[]
    featureDictionary={}
    with open(FeaturesData,'rU') as f:#####create a dictionary to store image id as key and feature list as values so 
        reader=csv.reader(f)#####that it can be easily searched for the required key values
        first_row = next(reader)
        temp=[]
        for rows in reader:
            for i in range(1,512):
                temp.append(int(rows[i]))
            featureDictionary.update({rows[0]:temp})
            temp=[] 
            
            
    with open(sameFilePath,'rU') as f1:###reading the same pair file and searching the features in the dictionary and appending
        count=1
        reader1=csv.reader(f1)
        first_row=next(reader1)
        for row in reader1:
            if count==791:
                break
            count+=1
            GSCFeaturesSameConcat.append(featureDictionary.get(row[0])+featureDictionary.get(row[1]))
                    
    
    with open(diffFilePath,'rU') as f2:##reading the different pair file and searching the features in the dictionary and appending
        reader2=csv.reader(f2)
        first_row=next(reader2)
        count = 1
        for row in reader2:
            if count==791:
                break
            count+=1
            GSCFeaturesDiffConcat.append(featureDictionary.get(row[0])+featureDictionary.get(row[1]))
            
    GSCFeaturesSameConcat.extend(GSCFeaturesDiffConcat)#concatanating same and different pair 
    
    with open(sameFilePath,'rU') as f1:#reading the same pair file and searching the features in the dictionary
        reader1=csv.reader(f1)#performing subtraction of the features and appending
        first_row=next(reader1)
        count=1
        for row in reader1:
            if count==791:
                break
            count+=1
            temp3=[0 for i in range(512)]
            #print(featureDictionary.get(row[0]))
            temp1=featureDictionary.get(row[0])
            temp2=featureDictionary.get(row[1])
            for i in range(len(temp1)):
                temp3[i]=abs(temp1[i]-temp2[i])
            GSCFeaturesSameSub.append(temp3)       
    
    
    with open(diffFilePath,'rU') as f1:#reading the different pair file and searching the features in the dictionary
        reader1=csv.reader(f1)#performing subtraction of the features and appending
        first_row=next(reader1)
        count1=1
        for row in reader1:
            if count1==791:
                break
            count1+=1    
            temp3=[0 for i in range(512)]
            #print(featureDictionary.get(row[0]))
            temp1=featureDictionary.get(row[0])
            temp2=featureDictionary.get(row[1])
            for i in range(len(temp1)):
                temp3[i]=abs(temp1[i]-temp2[i])
            GSCFeaturesDiffSub.append(temp3)
               
                
   
    GSCFeaturesSameSub.extend(GSCFeaturesDiffSub)#concatanating same and different pair 
    
    GSCFeaturesSameConcat = np.asarray(GSCFeaturesSameConcat)
    GSCFeaturesSameConcat=GSCFeaturesSameConcat+0.001##adding a very small noise 0.001 to make all the values of matrix greater than 0
    GSCFeaturesSameSub=np.asarray(GSCFeaturesSameSub)#to avoid singular matrix error
    GSCFeaturesSameSub=GSCFeaturesSameSub+0.001
    
    GSCFeaturesSameConcat = np.transpose(GSCFeaturesSameConcat)#performing transpose of concatanated and subtracted matrix
    GSCFeaturesSameSub=np.transpose(GSCFeaturesSameSub)
   
    return np.asarray(GSCFeaturesSameConcat),np.asarray(GSCFeaturesSameSub)   
            


# In[196]:


def generateTraining(rawData,TrainingPercent):#spliting the data as per the training percent
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def generateValid(rawData, ValPercent, TrainingCount): #spliting the data as per the training count and valid percent
    valLen = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valLen
    dataMatrix = rawData[:,TrainingCount+1:V_End] 
    return dataMatrix


# In[197]:


def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t


def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,trainingPercent):##calculating big sigma having dimension as per the number of features
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(trainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]

    BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, trainingPercent):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(trainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, trainingPercent):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(trainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.pinv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):#calcuating the error 
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# In[198]:


def linearRegression(inputFeaturesData,trainingData,validTesting,testingData,name):
    
    n=len(inputFeaturesData)
    m=len(inputFeaturesData[0])
    trainingData=np.transpose(trainingData)

    ErmsArr = []
    AccuracyArr = []
    kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(trainingData[:,:len(trainingData[0])-1])#using Kmeans function to create M clusters for our model
    Mu = kmeans.cluster_centers_
    
    BigSigma     = GenerateBigSigma(inputFeaturesData[:n-1,:], Mu, trainingPercent)
    TRAINING_PHI = GetPhiMatrix(inputFeaturesData[:n-1,:], Mu, BigSigma, trainingPercent)
    TrainingTarget=trainingData[:,len(trainingData[0])-1:].tolist()
    TrainingTarget=flatten(TrainingTarget)
    
    rawTarget=inputFeaturesData[n-1:,:].tolist()
    rawTarget=flatten(rawTarget)
   
    
    W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) #initializing the weight matrix
    TEST_PHI     = GetPhiMatrix(testingData[:len(testingData)-1,:], Mu, BigSigma, 100) 
    VAL_PHI      = GetPhiMatrix(validTesting[:len(validTesting)-1,:], Mu, BigSigma, 100)
    
    ValDataAct = np.array(GenerateValTargetVector(rawTarget,validationPercent, (len(TrainingTarget))))
    TestDataAct = np.array(GenerateValTargetVector(rawTarget,testPercent, (len(TrainingTarget)+len(ValDataAct))))
    
    W_Now        = np.dot(220, W)
    La           = 2##Regularization for stochastic gradient descent
    learningRate = 0.02 #steps size of gradient descent
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    for i in range(0,1000):
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)#derivative of the error function to perform gradient descent
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W#updated weight after gradient descent
        W_Now         = W_T_Next
        
    #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)#calculating error of training
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)#calculating error of validation
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestDataAct)#calculating error of testing
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
    print ('----------'+(name)+'--------------------')
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    
    E_rms_Training   = str(np.around(min(L_Erms_TR),5))
    E_rms_Validation   = str(np.around(min(L_Erms_Val),5))
    E_rms_Testing    = str(np.around(min(L_Erms_Test),5))
    
    return E_rms_Training,E_rms_Validation,E_rms_Testing


# In[199]:


humanObservedTargetData=[]
humanObservedConcat=np.array
humanObservedSub=np.array
humanObservedSameFilepath="/home/tanmay/Documents/machine learning/proj 2/HumanObserved-Dataset /HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv"
humanObservedTargetData.extend(generateTargetArray(humanObservedSameFilepath))
humanObservedDiffFilepath="/home/tanmay/Documents/machine learning/proj 2/HumanObserved-Dataset /HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv"
shuffdiff(humanObservedDiffFilepath)
humanObservedDiffFilepath="/home/tanmay/Documents/machine learning/proj 2/HumanObserved-Dataset /HumanObserved-Dataset/HumanObserved-Features-Data/output.csv"
humanObservedTargetData.extend(generateTargetArray(humanObservedDiffFilepath))
humanObservedFeaturesData="/home/tanmay/Documents/machine learning/proj 2/HumanObserved-Dataset /HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv"

humanObservedConcat,humanObservedSub=generateFeatureArray(humanObservedSameFilepath,humanObservedDiffFilepath,humanObservedFeaturesData)
humanObservedTargetData=np.asarray(humanObservedTargetData)
humanObservedConcat=np.transpose(humanObservedConcat)
humanObservedSub=np.transpose(humanObservedSub)
humanObservedTargetData=np.transpose(humanObservedTargetData)
humanObservedConcat= np.c_[humanObservedConcat.reshape(len(humanObservedConcat), -1), humanObservedTargetData.reshape(len(humanObservedTargetData), -1)]
humanObservedSub= np.c_[humanObservedSub.reshape(len(humanObservedSub), -1), humanObservedTargetData.reshape(len(humanObservedTargetData), -1)]

np.random.shuffle(humanObservedConcat)
np.random.shuffle(humanObservedSub)

humanObservedConcat=np.transpose(humanObservedConcat)
humanObservedSub=np.transpose(humanObservedSub)#human observed concatanated and subtracted dataset created


GSCTargetData=[]
GSCConcat=np.array
GSCSub=np.array
GSCSameFilepath="/home/tanmay/Documents/machine learning/proj 2/GSC-Dataset/GSC-Dataset/GSC-Features-Data/same_pairs.csv"
GSCTargetData.extend(generateTargetArray(GSCSameFilepath))
GSCDiffFilepath="/home/tanmay/Documents/machine learning/proj 2/GSC-Dataset/GSC-Dataset/GSC-Features-Data/diffn_pairs.csv"
shuffdiff1(GSCDiffFilepath)
GSCDiffFilepath="/home/tanmay/Documents/machine learning/proj 2/GSC-Dataset/GSC-Dataset/GSC-Features-Data/output.csv"
GSCTargetData.extend(generateTargetArray(GSCDiffFilepath))
GSCFeaturesData="/home/tanmay/Documents/machine learning/proj 2/GSC-Dataset/GSC-Dataset/GSC-Features-Data/GSC-Features.csv"

GSCConcat,GSCSub=generateGSCFeatureArray(GSCSameFilepath,GSCDiffFilepath,GSCFeaturesData)
GSCTargetData=np.asarray(GSCTargetData)
GSCConcat=np.transpose(GSCConcat)
GSCSub=np.transpose(GSCSub)
GSCTargetData=np.transpose(GSCTargetData)
GSCConcat= np.c_[GSCConcat.reshape(len(GSCConcat), -1), GSCTargetData.reshape(len(GSCTargetData), -1)]
GSCSub= np.c_[GSCSub.reshape(len(GSCSub), -1), GSCTargetData.reshape(len(GSCTargetData), -1)]

np.random.shuffle(GSCConcat)
np.random.shuffle(GSCSub)

GSCConcat=np.transpose(GSCConcat)
GSCSub=np.transpose(GSCSub)#GSC concatanated and subtracted dataset created


# In[200]:


##creating the training testing and validation dataset for all 4 datasets
humanObservedConcatTraining=np.array
humanObservedConcatTraining=generateTraining(humanObservedConcat,trainingPercent)
print(humanObservedConcatTraining.shape)
humanObservedConcatValid=np.array
humanObservedConcatValid=generateValid(humanObservedConcat,10,len(humanObservedConcatTraining))
print(humanObservedConcatValid.shape)


humanObservedConcatTest=np.array
humanObservedConcatTest=generateValid(humanObservedConcat,10,len(humanObservedConcatTraining)+len(humanObservedConcatValid))
print(humanObservedConcatTest.shape)

humanObservedSubTraining=np.array
humanObservedSubTraining=generateTraining(humanObservedSub,trainingPercent)


humanObservedSubValid=np.array
humanObservedSubValid=generateValid(humanObservedSub,10,len(humanObservedSubTraining))


humanObservedSubTest=np.array
humanObservedSubTest=generateValid(humanObservedSub,10,len(humanObservedSubTraining)+len(humanObservedSubValid))

GSCConcatTraining=np.array
GSCConcatTraining=generateTraining(GSCConcat,trainingPercent)


GSCConcatValid=np.array
GSCConcatValid=generateValid(GSCConcat,10,len(GSCConcatTraining[0]))


GSCConcatTest=np.array
GSCConcatTest=generateValid(GSCConcat,10,len(GSCConcatTraining[0])+len(GSCConcatValid[0]))
print(GSCConcatTest.shape)

GSCSubTraining=np.array
GSCSubTraining=generateTraining(GSCSub,trainingPercent)


GSCSubValid=np.array
GSCSubValid=generateValid(GSCSub,10,len(GSCSubTraining[0]))


GSCSubTest=np.array
GSCSubTest=generateValid(GSCSub,10,len(GSCSubTraining[0])+len(GSCSubValid[0]))


# In[202]:


def sigmoid(z):#makes the values between o and 1
    return 1 / (1 + np.exp(-z))

def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)

def loss_function(predictions,labels):#used to calculate the error of the model 
                                    #The cost should continuously decrease for the model to work correctly 
    observations = len(labels)

    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum()/observations

    return cost

def update_weights(features, labels, weights, lr):
    N = len(features)
    
    #1 - Get Predictions
    predictions = predict(features, weights)

    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T,  predictions - labels)
    print(gradient.shape)
    #3 Take the average cost derivative for each feature
    gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= lr

    #5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights

def decision_boundary(prob):
    return 1 if prob >= .5 else 0

def accuracy(predicted_labels, actual_labels):
    
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


# In[203]:


def logisticRegression(trainingData,validationData,testData,name):
    trainingData=np.transpose(trainingData)
    testData=np.transpose(testData)
    validationData=np.transpose(validationData)
    weights=np.random.randn(len(trainingData[0])-1,1)#randomly initializing the weights having dimension equal to (num of features,target vector)
    trainingFeatures=trainingData[:,:len(trainingData[0])-1]
    trainingLabels=trainingData[:,len(trainingData[0])-1:]
    testingFeatures=testData[:,:len(testData[0])-1]
    testingLabels=testData[:,len(testData[0])-1:]
    validationFeatures=validationData[:,:len(validationData[0])-1]
    validationLabels=validationData[:,len(validationData[0])-1:]
    trainingProb=[]
    validationProb=[]
    testingProb=[]
    
    for i in range(5000):
        trainingProb=[]
        validationProb=[]
        testingProb=[]
        N = len(trainingFeatures)
        predictionsTraining = predict(trainingFeatures, weights)
        for j in range(len(predictionsTraining)):#generating prediction for each input data
            trainingProb.append(decision_boundary(predictionsTraining[j][0]))
        loss=loss_function(predictionsTraining,trainingLabels)#calculating the error in prediction
        gradient = np.dot(trainingFeatures.T,  predictionsTraining - trainingLabels)##performing gradient descent to generate the next weights
        gradient /= N
        gradient *= lr
        weights -= gradient
        trainingAccuracy=accuracy(trainingProb,trainingLabels.flatten())
        
        predictionsValidation=predict(validationFeatures,weights)#using the generated weights to test on validation and testing datasets
        for j in range(len(predictionsValidation)):
            validationProb.append(decision_boundary(predictionsValidation[j][0]))
        validationAccuracy=accuracy(validationProb,validationLabels.flatten())
        
        predictionsTesting=predict(testingFeatures,weights)
        for j in range(len(predictionsTesting)):
            testingProb.append(decision_boundary(predictionsTesting[j][0]))
        testingAccuracy=accuracy(testingProb,testingLabels.flatten())
    print("---------------------------------"+(name)+"----------------------------------------------------------")
    print ("Training Accuracy  = " + str(trainingAccuracy))
    print ("Validation Accuracy  = " + str(validationAccuracy))
    print ("Testing Accuracy   = " + str(testingAccuracy))
    
    Training_Accuracy  = str(trainingAccuracy)
    Validation_Accuracy  = str(validationAccuracy)
    Testing_Accuracy   = str(testingAccuracy)
    
    return Training_Accuracy,Validation_Accuracy,Testing_Accuracy
    
        
    
    
    


# In[204]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers

import numpy as np

drop_out = 10
first_dense_layer_nodes  = 256
#third_dense_layer_nodes = 100
second_dense_layer_nodes = 1

def get_model(input_size):
    
    # Why do we need a model?
    #we need a model so that we can define the input output tensors, the network of nodes and various other parameters 
    #so that regression and classification operation can be perfromed
    
    # Why use Dense layer and then activation?
    # dense layer is nothing but a regular layer with neurons. we use dense layer to ensure that all the nodes
    #are connected to all other nodes in the next layer.
    
    
    # Why use sequential model with layers?
    #we have two types of model sequential and functional
    # In sequential model all the layers are connected in a sequence with the forward and the previous layers.
    #whereas in functional model the layers can be connected with any other layers 
    #since we do not require very complex computation sequential model is sufficient.
    
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Why dropout?
    # dropout is a regularization function.dropout is used to prevent overfitting of model by temporarily disabling 
    #some of the nodes in the layer while training
    model.add(Dropout(drop_out))
    
    #model.add(Dense(third_dense_layer_nodes))
    #model.add(Activation('relu'))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    #we are using sigmoid to limit the values between 0 and 1
    model.summary()
    
    # Why use categorical_crossentropy?
    #we use binary cross entropy so that the outputs can be classified in n two classes.
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


# In[205]:


def neuralNetworkModel(rawData,testData):
    model = get_model(len(rawData)-1)
   
    validation_data_split = 0.1
    num_epochs = 10000
    model_batch_size = 12
    tb_batch_size = 8
    early_patience = 100
    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')#making provision for early stopping of model if gaph converge
    rawData=np.transpose(rawData)
    testData=np.transpose(testData)
    processedData=rawData[:,:len(rawData[0])-1]
    processedLabel=rawData[:,len(rawData[0])-1:]
    history = model.fit(processedData###creating a model with the following parameters
                        , processedLabel
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                    
                       )
    
    wrong   = 0.0
    right   = 0.0
    processedTestData=testData[:,:len(testData[0])-1]
    processedTestLabel=testData[:,len(testData[0])-1:]
    print(processedTestData.shape,processedTestLabel.shape)
    predictedTestLabel = []
    
    for i,j in zip(processedTestData,processedTestLabel):
        y = model.predict(np.array(i).reshape(-1,len(processedTestData[0])))
        predictedTestLabel.append(decodeLabel(y))
        if j == decision_boundary(y):
            right = right + 1
        else:
            wrong = wrong + 1
    
    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("Testing Accuracy: " + str(right/(right+wrong)*100))
    df = pd.DataFrame(history.history)
    return str(right/(right+wrong)*100),df
    


# In[206]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Different Writer"
    elif encodedLabel == 1:
        return "Same writer"
  


# In[207]:


E_rmstrain1,E_rmsvalid1,E_rmstest1=linearRegression(humanObservedConcat,humanObservedConcatTraining,humanObservedConcatValid,humanObservedConcatTest,"human-observed-concat-linear-regression")
E_rmstrain2,E_rmsvalid2,E_rmstest2=linearRegression(humanObservedSub,humanObservedSubTraining,humanObservedSubValid,humanObservedSubTest,"human-observed-sub-linear-regression")
E_rmstrain3,E_rmsvalid3,E_rmstest3=linearRegression(GSCConcat,GSCConcatTraining,GSCConcatValid,GSCConcatTest,"GSC-concat-linear-regression")
E_rmstrain4,E_rmsvalid4,E_rmstest4=linearRegression(GSCSub,GSCSubTraining,GSCSubValid,GSCSubTest,"GSC-sub-linear-regression")


# In[208]:


train1,valid1,test1=logisticRegression(humanObservedConcatTraining,humanObservedConcatValid,humanObservedConcatTest,"logistic-regression-on-human-observed-concatanation")
train2,valid2,test2=logisticRegression(humanObservedSubTraining,humanObservedSubValid,humanObservedSubTest,"logistic-regression-on-human-observed-subtraction")
train3,valid3,test3=logisticRegression(GSCConcatTraining,GSCConcatValid,GSCConcatTest,"logistic-regression-on-GSC-concatanation")
train4,valid4,test4=logisticRegression(GSCSubTraining,GSCSubValid,GSCSubTest,"logistic-regression-on-GSC-subtraction")


# In[209]:


acc1,df1=neuralNetworkModel(humanObservedConcatTraining,humanObservedConcatTest)
acc2,df2=neuralNetworkModel(humanObservedSubTraining,humanObservedSubTest)
acc3,df3=neuralNetworkModel(GSCConcatTraining,GSCConcatTest)
acc4,df4=neuralNetworkModel(GSCSubTraining,GSCSubTest)


# In[210]:


print("Testing Accuracy NN human observed concat: " + acc1)
print("Testing Accuracy NN human observed Sub: " + acc2)
print("Testing Accuracy NN GSC concat: " + acc3)
print("Testing Accuracy NN GSC sub: " + acc4)


# In[211]:



df1.plot(subplots=True, grid=True, figsize=(5,20))
df2.plot(subplots=True, grid=True, figsize=(5,20))
df3.plot(subplots=True, grid=True, figsize=(5,20))
df4.plot(subplots=True, grid=True, figsize=(5,20))


# In[ ]:




