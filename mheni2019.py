"""

Autor: Mateus Figueira de Araújo.
Reprodução do Sistema Biométrico Mheni Et al. 2019
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import time




class doubleSerialKeystroke:
    ''' Double serial keystroke system. 
    1. Pre-Tratamento
    2. Cadastro - ok
    3. Verificação
    4. Adaptação'''

    def __init__(self, sample, columns):
        self.gallery = pd.DataFrame(columns=columns)
        self.gallery = self.gallery.append(sample)
        self.thresholdVerification = np.float(0)
        self.thresholdUpdate = np.float(0)
        self.weights = np.array([0,0,0,0],dtype=float)
        return None
    
    def updateGallery(self, sample, score):
        rows = len(self.gallery)
        threshold = self.thresholdUpdate
        
        if score < threshold:
            if rows < 10:
                self.gallery = self.gallery.append(sample)
            else:
                self.gallery = self.gallery.append(sample)
                self.gallery = self.gallery.iloc[1:]
        return None
    
    def verification(self, sample):
        threshold = self.thresholdVerification
        weights = self.weights
        gallery = self.gallery
        score = sampleScore(gallery, sample, weights)
        
        if score < threshold:
            status = True
        else:
            status = False
        
        return status, score
    
    def setThreshold(self, thresholdVerification, thresholdAdaptation):
        self.thresholdVerification = np.float64(thresholdVerification)
        self.thresholdUpdate = np.float64(thresholdAdaptation)
        return None
    
    def setWeights(self, weights):
        self.weights = np.array(weights, dtype=float)
        return None
def knnEuclidian(gallery, sample,k=1):
    '''1-NN
    Calcula a distância euclidiana da galeria para a amostra
    e retorna a distância mais próxima.
    '''
    dist = gallery - sample
    dist = np.linalg.norm(dist, axis=1)
    minimunDistance = dist.min()
    return minimunDistance

def knnManhattan(gallery, sample,k=1):
    '''1-NN
    Calcula a distância de Manhattan da galeria para a amostra
    e retorna a distância mais próxima.
    '''
    dist = distance.cdist(gallery,np.array([sample]),'cityblock')
    minimunDistance = dist.min()
    return minimunDistance

def knnHamming(gallery, sample,k=1):
    """1-NN Calcula a distância de Hammingg da galeria x 10 para a amostra x 10
    e retorna a distância mais próxima."""
    gallery = (gallery*10).astype(int)
    sample = (sample*10).astype(int)
    dist = distance.cdist(gallery,np.array([sample]),'hamming')
    minimunDistance = dist.min()
    return minimunDistance

def knnStatistical(gallery, sample,k=1):
    """1-NN Calcula a distância de Estatistica da galeria para a amostra."""
    n = len(gallery.columns)
    rows = len(gallery)
    if rows == 1:
        loc = gallery
        scale = gallery - gallery
        minimunDistance = 1
    else:
        loc = gallery.mean()
        scale = gallery.std()
        exp = -np.abs(loc-sample)/scale
        minimunDistance = 1 - np.exp(exp.astype(float)).sum()/n
    return minimunDistance

def sampleScore(gallery, sample, weights):
    a, b, c, d = weights
    dEuclidian = knnEuclidian(gallery, sample)
    dHamming = knnHamming(gallery, sample)
    dManhattan = knnManhattan(gallery, sample)
    dStat = knnStatistical(gallery, sample)
    score = a*dEuclidian+b*dHamming+c*dManhattan+d*dStat
    return score

def metricsSection(fmrSection, fnmrSection):
    fmr = np.sum(fmrSection)/len(fmrSection)
    fnmr = np.sum(fnmrSection)/len(fnmrSection)
    hter = (fmr+fnmr)/2
    
    return fmr, fnmr, hter

def arithRecombination(p1,p2):
    alpha = np.random.rand()
    c1 = alpha*p1+(1-alpha)*p2
    c2 = alpha*p2+(1-alpha)*p1
    
    return c1,c2

def mutationGauss(gene, scale, size, prob):
    r = np.random.rand()
    if r <= prob:
        newGene = gene + np.random.normal(loc=0, scale=scale,size=size)
        return newGene
    else:
        return gene

def chanceAcc(vetor):
    n = len(vetor)
    chance = (1-vetor)/((n-1)*np.sum(vetor))
    acc= np.zeros(n)
    for idx in range(1,n):
        acc[idx] = chance[idx-1]+chance[idx]
    return acc

def sus(fitGen, n, pop_size):
    choices = []
    pointers = []
    r = 2
    step = 1/n
    while r >= 1/n:
        r = np.random.rand()
    chance = chanceAcc(fitGen)
    for i in range(n):
        pointer = r + i*step
        pointers.append(pointer)
        for j in range(pop_size):
            if pointer < chance[j]:
                choices.append(j)
                break
    return choices

"""
Preparando Dados (ISSO É FIXO)
"""
start_time = time.time()
df = pd.read_csv('CMU_Mheni2019.csv')
df = df.iloc[:,1:]
dfImp = df.iloc[400:,:]
dfGen = df.iloc[:400,]
dfSec = dfGen.iloc[1:6,:]
dfSec=dfSec.append(dfImp.sample(n=3, random_state=1))
popSize = 50
generations = 10
variables = 4
enroll = dfGen.iloc[0,3:]
enrollY = dfGen.iloc[0,0]
columns = enroll.index
"""
AQUI MUDA PARA CADA INDIVÍDUO DA POPULAÇÃO
"""
s002 = []
fnmrGen = np.array([])
fmrGen = np.array([])
fitnessPai = np.array([])

sectionSize = 8

for j in np.arange(popSize):
    weights = np.random.normal(0,3,size=4)
    s002.append(doubleSerialKeystroke(enroll, columns))
    s002[j].setThreshold(0.8,0.8)
    s002[j].setWeights(weights)
for j in np.arange(popSize):
    fnmrSec = []
    fmrSec = []
    for i in np.arange(sectionSize):
        sample = dfSec.iloc[i,3:]
        sampleY = dfSec.iloc[i,0]
        status, score = s002[j].verification(sample)

        """
        Verificando se a amostra era genuina ou impostora
        """
        if (sampleY == enrollY) and status:
            fnmrSec.append(0)
        elif (sampleY != enrollY) and not status:
            fmrSec.append(0)
        elif (sampleY == enrollY) and not status:
            fnmrSec.append(1)
        elif (sampleY != enrollY) and status:
            fmrSec.append(1)
    
        """
        Hora de adaptar o template
        """
        if status is True:
            s002[j].updateGallery(sample, score)
    fmr, fnmr, hter = metricsSection(fmrSec,fnmrSec)
    fitnessPai = np.append(fitnessPai,hter)



for generation in np.arange(generations):
    for j in np.arange(0,popSize,2):
        p1 = s002[j].weights
        p2 = s002[j+1].weights
        c1, c2 = arithRecombination(p1,p2)
        c1 = mutationGauss(c1, scale=1, size = variables, prob=0.1)
        c2 = mutationGauss(c2, scale=1, size = variables, prob=0.1)
        s002[j] = doubleSerialKeystroke(enroll, columns)        
        s002[j].setWeights(c1)
        s002[j].setThreshold(0.8,0.8)
        s002[j+1] = doubleSerialKeystroke(enroll, columns)
        s002[j+1].setWeights(c2)
        s002[j+1].setThreshold(0.8,0.8)
    fitnessPai = np.array([])
    
    for j in np.arange(popSize):
        fnmrSec = np.array([])
        fmrSec = np.array([])
        for i in np.arange(sectionSize):
            
            sample = dfSec.iloc[i,3:]
            sampleY = dfSec.iloc[i,0]
            status, score = s002[j].verification(sample)

            """
            Verificando se a amostra era genuina ou impostora
            """
            if (sampleY == enrollY) and status:
                fnmrSec= np.append(fnmrSec, 0)
            elif (sampleY != enrollY) and not status:
                fmrSec= np.append(fmrSec, 0)
            elif (sampleY == enrollY) and not status:
                fnmrSec= np.append(fnmrSec, 1)
            elif (sampleY != enrollY) and status:
                fmrSec= np.append(fmrSec, 1)
        
            """
            Hora de adaptar o template
            """
            if status is True:
                s002[j].updateGallery(sample, score)
        fmr, fnmr, hter = metricsSection(fmrSec,fnmrSec)
        fitnessPai = np.append(fitnessPai,hter)
    
    

    print(generation)

print("--- %s seconds ---" % (time.time() - start_time))
# columnsMetrics = ["subject","section","FMR","FNMR","HTER"]
# metrics = pd.DataFrame(columns=columnsMetrics)
# fmr, fnmr, hter = metricsSection(r1, r2)
# sectionMetrics = pd.DataFrame([["s002",0,fmr,fnmr,hter]], columns=columnsMetrics)
# metrics = metrics.append(sectionMetrics, ignore_index = True)



