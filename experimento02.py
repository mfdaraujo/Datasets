# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:01:24 2021

@author: mfdar
"""


import numpy as np
import pandas as pd
import scipy as sp

# Visualização
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

def scoreStatistical(gallery, sample, loc, scale):
    n = len(gallery.columns)
    rows = len(gallery)
    exp = -np.abs(loc-sample)/scale
    score = 1 - np.exp(exp.astype(float)).sum()/n
    return score

def metricsSection(fmrSection, fnmrSection):
    fmr = np.sum(fmrSection)/len(fmrSection)
    fnmr = np.sum(fnmrSection)/len(fnmrSection)
    hter = (fmr+fnmr)/2
    
    return fmr, fnmr, hter

class statClassifier:
  
  def __init__(self, sample, columns):
        self.gallery = pd.DataFrame(columns=columns)
        self.gallery = self.gallery.append(sample)
        self.loc  =  self.gallery.mean()
        self.scale = self.gallery.std()
        self.threshold = -np.inf
  
  def verification(self, sample):
        threshold = self.threshold
        gallery = self.gallery
        loc = self.loc
        scale = self.scale
        score = scoreStatistical(gallery, sample, loc, scale)
        
        if score < threshold:
            status = True
        else:
            status = False
        
        return status, score

  def setThreshold(self, threshold):
        self.threshold = np.float64(threshold)
        return None
  
  def updateGallery(self, sample, gallerySize=10):
        rows = len(self.gallery)

        if rows < gallerySize:
            self.gallery = self.gallery.append(sample)
        else:
            self.gallery = self.gallery.append(sample)
            self.gallery = self.gallery.iloc[1:]
        self.loc  =  self.gallery.mean()
        self.scale = self.gallery.std()
        return None
    
df = pd.read_csv(r'DSL-StrongPasswordData-Modificado.csv', sep = ',', error_bad_lines=False)  
df = df.drop(["rep","sessionIndex","Unnamed: 0"],axis=1)
userMax = 51
users = [0 for _ in range(userMax)]
dfThresholdEvo = pd.DataFrame(columns=['section','bestsThreshold'])
for j in range(userMax):
    dfGen = df.iloc[400*j:400*(j+1)]
    dfImp = df.drop(dfGen.index)
    enrollmentSamples = dfGen.iloc[:10,1:]
    dfSec = dfGen.iloc[:10,:]
    dfSec = dfSec.append(dfImp.sample(n=5))
    enrollmentY = dfGen.iloc[0,0]
    dfGen = dfGen.drop(enrollmentSamples.index)
    users[j] = statClassifier(enrollmentSamples,enrollmentSamples.columns)



    #Etapa de verificação

    sectionMax = 39
    sectionSize = 15
    fnmrSubject = []
    fmrSubject = []
    hterSubject = []
    
    # 400 amostras (10 no cadastro; 390 verificação; 39 seções)
    for sec in range(sectionMax):
            # Encontrar o threshold

        num = 101
        sectionSize = 15
        hterMin = np.inf
        dfThreshold = pd.DataFrame(columns=['hter','threshold'])
        dfSec = dfGen.iloc[:10,:]
        dfGen = dfGen.drop(dfSec.index)
        dfSec = dfSec.append(dfImp.sample(n=5))
        for threshold in np.linspace(0,1,num=num):
            users[j].setThreshold(threshold)
            fnmrSec = np.array([])
            fmrSec = np.array([])
          
            for i in range(sectionSize):
                sample = dfSec.iloc[i,1:]
                sampleY = dfSec.iloc[i,0]
                status, score = users[j].verification(sample)
                if (sampleY == enrollmentY) and status:
                    fnmrSec= np.append(fnmrSec, 0)
                elif (sampleY != enrollmentY) and not status:
                    fmrSec= np.append(fmrSec, 0)
                elif (sampleY == enrollmentY) and not status:
                    fnmrSec= np.append(fnmrSec, 1)
                elif (sampleY != enrollmentY) and status:
                    fmrSec= np.append(fmrSec, 1)
            fmr, fnmr, hter = metricsSection(fmrSec,fnmrSec)
            aux = np.array([hter,threshold])
            aux = pd.Series(data=aux, index=['hter','threshold'])
            dfThreshold = dfThreshold.append(aux, ignore_index=True)
            if hter < hterMin:
                hterMin = hter
        # Definindo threshold
        mask = dfThreshold['hter']==hterMin
        thAtual = dfThreshold['threshold'][mask]
        threshold = thAtual.sample()
        # secAtual = [sec for _ in thAtual]
        # secAtual = np.array(secAtual)
        # userAtual = [j for _ in thAtual]
        # userAtual = np.array(userAtual)
        users[j].setThreshold(threshold)
        if len(thAtual) == 1:
            aux = np.array([j, sec, thAtual.iloc[0]])
            aux = pd.DataFrame(data=[aux], columns=['user','section','bestsThreshold'])
        else:
            aux = np.array([j, sec, thAtual.mean()])
            aux = pd.DataFrame(data=[aux], columns=['user','section','bestsThreshold'])
        if len(dfThresholdEvo) == 0:
            dfThresholdEvo = pd.DataFrame(data = aux,columns=['user','section','bestsThreshold'])
        else:
            dfThresholdEvo = dfThresholdEvo.append(aux, ignore_index=True)
        # Utiliza as 10 primeiras amostras
        # print(f'Th: {threshold}')
        fnmrSec = np.array([])
        fmrSec = np.array([])
        for i in range(sectionSize):
            sample = dfSec.iloc[i,1:]   
            sampleY = dfSec.iloc[i,0]
            status, score = users[j].verification(sample)
            if (sampleY == enrollmentY) and status:
                    fnmrSec= np.append(fnmrSec, 0)
            elif (sampleY != enrollmentY) and not status:
                    fmrSec= np.append(fmrSec, 0)
            elif (sampleY == enrollmentY) and not status:
                    fnmrSec= np.append(fnmrSec, 1)
            elif (sampleY != enrollmentY) and status:
                    fmrSec= np.append(fmrSec, 1)
            #print(f'Score = {score}; Aprovado? {status}')
        fmr, fnmr, hter = metricsSection(fmrSec,fnmrSec)
        fmrSubject.append(fmr)
        fnmrSubject.append(fnmr)
        hterSubject.append(hter)
    accSubject = 1 - np.array(hterSubject)
    
    
    if j == 0:
        section = np.arange(39)
        user = [0 for _ in section]
        user = np.array(user)
        metrics = pd.DataFrame(data=[user,section,fmrSubject,fnmrSubject,hterSubject, accSubject],
                           index=['user','section','fmr','fnmr','hter','acc'])
        metrics = metrics.T
    else:
        user = user + 1
        aux = pd.DataFrame(data=[user,section,fmrSubject,fnmrSubject,hterSubject, accSubject],
                           index=['user','section','fmr','fnmr','hter','acc'])
        metrics = metrics.append(aux.T)

sns.set_style("darkgrid")
f, axs = plt.subplots(figsize=(6, 4))
# fig1 = sns.relplot(x="section", y="hter", data=metrics)
sns.lineplot(x="section", y="acc", data=metrics)
metrics.to_csv('metrics_verification2.csv', index=False)
dfThresholdEvo.to_csv('ThresholdEvoNoUpdate.csv', index=False)
