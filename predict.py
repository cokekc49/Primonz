from botnoi import scrape as sc
from botnoi import cv
import os
import glob


import glob
import pandas as pd
import pickle

dataset = []
def createdataset():
  imgfolder = glob.glob('images/*')
  for cls in imgfolder:
    clsset = pd.DataFrame()
    pList = glob.glob(cls+'/*')
    featvec = []
    for p in pList:
      dat = pickle.load(open(p,'rb'))
      featvec.append(dat.resnet50)
    clsset['feature'] = featvec
    cls = cls.split('/')[-1]
    clsset['label'] = cls
    dataset.append(clsset)
  return pd.concat(dataset,axis=0)

dataset = createdataset()


imgfolder = glob.glob('images/*')
for cls in imgfolder:
  imgList = glob.glob(cls+'/*')

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC
def trainmodel(dataset,modfile=''):
  trainfeat, testfeat, trainlabel, testlabel = train_test_split(dataset['feature'], dataset['label'], test_size=0.33, random_state=42)
  clf = LinearSVC()
  mod = clf.fit(np.vstack(trainfeat.values),trainlabel.values)
  res = mod.predict(np.vstack(testfeat.values))
  if modfile!='':
    pickle.dump(mod,open(modfile,'wb'))
  acc = sum(res == testlabel)/len(res)
  return mod,acc

mod,acc = trainmodel(dataset,'mymod.mod')

# output function
modFile = 'mymod.mod'
mod = pickle.load(open(modFile,'rb'))

def predicting(imgurl):
  a = cv.image(imgurl)
  feat = a.getresnet50()
  res = mod.predict([feat])
  return res


def answer(name):
  a=predicting(name)
  b = str(a).split(' ')
  c = b[0].split('\'')
  return str(c[1])
