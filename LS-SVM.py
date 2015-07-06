# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:22:18 2015

@author: menchangqian
"""
from numpy import *
from numpy import linalg as LA
import cPickle
import gzip
import theano
#

class parameter:
    kernel='linear'
    pa=2**(-20)
    C=2
    def __init__(self,kernel,pa,C):
        self.C=C
        self.kernel=kernel
        self.pa=pa

class model:
    alpha=0;
    b=0
    def __init__(self,alpha,b):
        self.alpha=alpha
        self.b=b

def LSTest(validselect,validselectl,testselect,testselectl):
    p=parameter('rbf',2**(-30),2**(1))
    alpha,b= LS_SVM(validselect,validselectl,p.C,p.pa,1,7)
    m=model(alpha,b)
    r,rate=output(validselect,validselectl,testselect,testselectl,m.alpha,m.b,1,7,2**(-20))
    return r,rate,m

def loaddata(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def kernelfunction(u,v,ktype,para):
    if ktype=='rbf':
        return e**(-1*para*(LA.norm(u-v)**2))
    else:
        return dot(u,v)

def kernerlmatrix(dataset,pa):
    m,n=shape(dataset)
    kernel=[kernelfunction(u,v,'rbf',pa) for u in dataset for v in dataset]
   
    return array(kernel).reshape(m,m)

def LS_SVM(dataset,datasetL,C,pa,i,j):
    k=kernerlmatrix(dataset,pa)
    datasetL=array(datasetL)
    datasetL[datasetL==i]=1
    datasetL[datasetL==j]=-1
    datasetL=matrix(datasetL).T
    m,n=shape(k)
    re=(1.0/C)*eye(m)
    M=diag(datasetL)*LA.pinv(k+re)*diag(datasetL)
    b=(datasetL.T*M*matrix(ones(m)).T)/(datasetL.T*M*datasetL)
    alpha=M*(matrix(ones(m)).T-datasetL*b)
    return alpha,b

def predict(dataset,datasetL,x,alpha,b,pa):
    kv=[kernelfunction(u,x,'rbf',pa)  for u in dataset]
    fval=alpha.T*diag(datasetL)*matrix(kv).T+b
    return float(sign(fval))

def output(trainset,trainsetL,testset,testsetL,alpha,b,i,j,pa):
    trainset=array(trainset)
    trainsetL=array(trainsetL)
    testset=array(testset)
    testsetL=array(testsetL)    
    testsetL[testsetL==i]=1
    testsetL[testsetL==j]=-1
    trainsetL[trainsetL==i]=1
    trainsetL[trainsetL==j]=-1
    m,n=shape(testset)
    result=[]
    error=0
    for i in range(m):
        f=predict(trainset,trainsetL,testset[i,:],alpha,b,pa)
        result.append(f)
        if f!=testsetL[i]:
            error+=1
    return result,error/float(m)

def dataselect(dataset,i,j):
    train_set, valid_set, test_set=loaddata(dataset)
    train=train_set[0]
    trainl=train_set[1]
    valid=valid_set[0]
    validl=valid_set[1]   
    test=test_set[0]
    testl=test_set[1]
    m,n=shape(train)
    trainselect=[train[k] for k in range(m) if trainl[k]==i or trainl[k]==j]
    trainselectl=[trainl[k] for k in range(m) if trainl[k]==i or trainl[k]==j]
#    trainselectl[trainselectl==i]=1
#    trainselectl[trainselectl==j]=-1
    m,n=shape(valid)
    validselect=[valid[k] for k in range(m) if validl[k]==i or validl[k]==j]
    validselectl=[validl[k] for k in range(m) if validl[k]==i or validl[k]==j]
    m,n=shape(test)
    testselect=[test[k] for k in range(m) if testl[k]==i or testl[k]==j]
    testselectl=[testl[k] for k in range(m) if testl[k]==i or testl[k]==j]
    return trainselect,trainselectl,validselect,validselectl,testselect,testselectl


#train_set, valid_set, test_set=loaddata("mnist.pkl.gz")
#k=kernerlmatrix(dataset)
trainselect,trainselectl,validselect,validselectl,testselect,testselectl=dataselect("mnist.pkl.gz",1,7)
#alpha,b= LS_SVM(validselect,validselectl,2,2**(-20),6,8)
#
#r,rate=output(validselect,validselectl,testselect,testselectl,m.alpha,m.b,1,7)
r,rate,m= LSTest(validselect,validselectl,testselect,testselectl)