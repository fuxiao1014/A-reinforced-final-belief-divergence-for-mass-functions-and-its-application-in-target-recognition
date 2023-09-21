from pyds import MassFunction
from itertools import product
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter
import pandas as pd
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split

import csv

#=============================1.Divergence=================================================

#===============================Proposed method============================================
# (1)Calculate the similarity factor G and the quantity factor F
      
def ffgg(ss): #M is a list. Calculate two factors.
      global fr
      
      #Add to list arr1 and arr2
      arr1 = []
      arr2 = []
      for i in ss:
            arr1.append(i)
      for j in ss:
            arr2.append(j)


      #get four matrices, Sam,Dif,Fm, and Fi
      Sam = [[] for j in range(0,fr)] #similarity
      Dif = [[] for j in range(0,fr)] #difference
      Fm = [[] for j in range(0,fr)] #max
      Fi = [[] for j in range(0,fr)] #min
      
      #two  correlation coefficients
      F1 = [[] for j in range(0,fr)]

      F2 = [[] for j in range(0,fr)]
           
      for mm1 in range(0,fr):
            for mm2 in range(0,fr):
                  sam = 0
                  dif = 0
            
                  #Calculate sam and dif
                  for i in arr1[mm1]:
                        for j in arr2[mm2]:
                              if(i==j):
                                    sam += 1
                  dif = len(arr1[mm1]) + len(arr2[mm2]) - sam
                  #Add Amm1∩Bmm2 to list Sam. Add Amm1∪Bmm2 to list Dif.
                  Sam[mm1].append(sam) 
                  Dif[mm1].append(dif) 

                  #max min
                  Fm[mm1].append(max(len(arr1[mm1]),len(arr2[mm2])))
                  Fi[mm1].append(min(len(arr1[mm1]),len(arr2[mm2])))

                  if (len(arr1[mm1]) >= len(arr2[mm2])):
                        F1[mm1].append(Sam[mm1][mm2] / Fi[mm1][mm2])
                        F2[mm1].append(Sam[mm1][mm2] / Fm[mm1][mm2])
                  else:
                        F1[mm1].append(Sam[mm1][mm2] / Fm[mm1][mm2])
                        F2[mm1].append(Sam[mm1][mm2] / Fi[mm1][mm2])

      
      
      #get G and F
      G = [[] for j in range(0,fr)]
      F = [[] for j in range(0,fr)]
      
      for i in range(0,fr):
            for j in range(0,fr):
                  g = (2**Sam[i][j] - 1)/(2**Dif[i][j] - 1)
                  G[i].append(g)

                  f = 1/(2**Fm[i][j] - 1)
                  F[i].append(f)
                  
      #print('G:\n',np.matrix(G))
      #print('F:\n',np.matrix(F))
      fandg = []
      fandg.append(G)
      fandg.append(F)
      fandg.append(F1)
      fandg.append(F2)
      return fandg

#==========================================================================================
# (2)Calculate the similarity-quantity coefficient α(Ai,Aj)
def aaa(m1,m2,ss): #Calculate αij
      global fr
      global GG
      global FF
      
      G_ = ffgg(ss)[0] #use function ffgg()
      F_ = ffgg(ss)[1]

      F1_ = ffgg(ss)[2]
      F2_ = ffgg(ss)[3]
      
      #Calculate the mean of F and G respectively
      Fav = 0
      Gav = 0
      sFav = 0
      sGav = 0
      avee = []
      k1 = len(m1)
      k2 = len(m2)
      
      for i in range(0,fr):
            for j in range(0,fr):
                  ff = F_[i][j]
                  gg = G_[i][j]
                  if (frozenset(ss[i]) in m1 and frozenset(ss[j]) in m2):
                        sFav += ff
                        sGav += gg
                        
                        
      Fav = sFav / (k1 * k2)
      Gav = sGav / (k1 * k2)
      #print("the mean of F:",Fav)
      #print("the mean of G:",Gav)
      
      #get αij
      alpha = [[[] for i in range(0,fr)] for j in range(0,2)]
      alp = 0
      alp_ = 0

      #G-influence degree coefficient Ga and F-influence degree coefficient Fa
      Ga = math.exp((Fav-1)*(Gav**2))

      Fa = math.exp(-Gav*((Fav-1)**2))

      #the normalized influence degree coefficients Gu and Fu
      Gu = Ga / (Ga + Fa) 
      Fu = Fa / (Ga + Fa)
      
      for n in range(0,2):           
            for i in range(0,fr):
                   for j in range(0,fr):
                        #Use the geometric mean to get the similarity-quantity coefficient α(Ai,Aj)
                        if (n == 0):
                              alp = math.sqrt(Gu * G_[i][j] * Fu * F_[i][j])
                              if (alp == 0):
                                    alphaij = 0
                              else:
                                    alphaij = F1_[i][j] * math.exp(1 - alp) / (math.exp(1/2))
                        else:
                              alp_ = math.sqrt(Gu * G_[i][j] * Fu * F_[i][j])
                              if (alp_ == 0):
                                    alphaij = 0
                              else:
                                    alphaij = F2_[i][j] * math.exp(1 - alp_) / (math.exp(1/2))
                        alpha[n][i].append(alphaij)
      #print('α(Ai,Aj):\n',np.matrix(alpha))
      return alpha

#==========================================================================================
# (3)Reinforced final belief divergence measure (RFBD)
def Div(m1,m2):  #Calculate FBD(m1,m2)
      div = 0
      global sub_
      global fr
      alpha_ = aaa(m1,m2,sub_) #use function aaa()
      arr1_ = value(m1,sub_) #m1
      arr2_ = value(m2,sub_) #m2
      for i in range(0,fr):
            for j in range(0,fr):
                  if (arr1_[i] != 0):
                        x = 2 * arr1_[i] / (arr1_[i] + arr2_[j])
                        x_ = arr1_[i] * math.log(x,2)
                  else:
                        x_ = 0
                  if (arr2_[j] != 0):
                        y = 2 * arr2_[j] / (arr1_[i] + arr2_[j])
                        y_ = arr2_[j] * math.log(y,2)
                  else:
                        y_ = 0
                  div += alpha_[0][i][j] * x_ + alpha_[1][i][j] * y_
      
      #print('FBD:',div)
      return div

def value(m,ss_): #The values of the mass function
      arr = []
      global fr
      ling = 0.0
      t = 0
      for i in ss_:
            for j in m.items(): 
                  if (i == j[0]):  
                        arr.append(j[1])
                        break
                  else:
                        t += 1
                  if (t == len(m.items())): 
                        arr.append(ling)
                        t = 0
      return arr


def D(m1,m2):  #Calculate RFBD(m1,m2)
      d = (Div(m1,m1)+Div(m2,m2)-2*Div(m1,m2))/2
      d_ = pow(abs(d),1/2)
      print('RFBD:',d_)
      return d_


def MTD(M,k):  #Calculate the matrix of RFBD(m1,m2)
      D_ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D_[p].append(D(M[p],M[q]))
      print('the matrix of RFBD:\n',np.matrix(D_))
      return D_


#==========================================================================================

#=================================Method of Fuyuan Xiao====================================

def Div_xiao(m1,m2):
      div_ = 0
      global sub_
      global fr
      
      F1__ = ffgg(sub_)[2]
      F2__ = ffgg(sub_)[3]
      arr1__ = value(m1,sub_) 
      arr2__ = value(m2,sub_) 

         
      for i in range(0,fr):
            for j in range(0,fr):
                  if (arr1__[i] != 0):
                        x = 2 * arr1__[i] / (arr1__[i] + arr2__[j])
                        x_ = arr1__[i] * math.log(x,2)
                  else:
                        x_ = 0
                  if (arr2__[j] != 0):
                        y = 2 * arr2__[j] / (arr1__[i] + arr2__[j])
                        y_ = arr2__[j] * math.log(y,2)
                  else:
                        y_ = 0
                  div_ += F1__[i][j] * x_ + F2__[i][j] * y_
      
      #print('Xiao's B:',div)
      return div_


def D_xiao(m1,m2):
      d_ = (Div_xiao(m1,m1)+Div_xiao(m2,m2)-2*Div_xiao(m1,m2))/2
      d__ = pow(abs(d_),1/2)
      #print('Xiao's RB:',d__)
      return d__


def MTD_xiao(M,k):
      D__ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D__[p].append(D_xiao(M[p],M[q]))
      #print('the matrix of RB:\n',np.matrix(D__))
      return D__



#==========================================================================================

#==================================Method of Zichong Chen==================================

def Div_chen(m1,m2):
      div_ = 0
      global sub_
      global fr
      mm1 = []
      for m in m1.items():
            sorted(m[0])
            mm1.append(m)          
      mm2 = []      
      for m in m2.items():
            sorted(m[0])
            mm2.append(m)
      F0 = 0
      for m_1 in range(0,len(mm1)):
            for m_2 in range(0,len(mm2)):                                    
                  Fm = max(len(mm1[m_1][0]),len(mm2[m_2][0])) #max
                  if(Fm > F0): 
                        F0 = Fm

      sum_th = 0
      div_ = 0
      f_flag = 1
      for i in range(0,len(mm1)):
            for j in range(0,len(mm2)):
                  if(mm1[i][0] == mm2[j][0]):
                        if (mm1[i][1] == 0 or mm2[j][1] == 0):
                              th = 0
                        elif(F0 == 1): #all singleton sets
                              div_ += mm1[i][1] * math.log(mm1[i][1]/mm2[j][1],2) #degenerate into KL divergence                    
                              f_flag = 0
                        else:
                              th = round(pow(mm1[i][1],F0) * pow(mm2[j][1],1-F0),2)
                              sum_th += th

      if(f_flag == 1):
            div_ = (1/(F0-1)) * math.log(sum_th,2)
      
      #print('Chen's RBD:',div_)
      return div_


def D_chen(m1,m2):
      
      d__ = (Div_chen(m1,m2)+Div_chen(m2,m1))/2

      #print('Chen's MRBD:',d__)
      return d__


def MTD_chen(M,k):
      D__ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D__[p].append(D_chen(M[p],M[q]))
      #print('the matrix of MRBD:\n',np.matrix(D__))
      return D__


#=============================================================================================================

#==================================Method of Lipeng Pan=======================================================
def Div_pan(m1,m2):
      div_ = 0
      div_x = 0
      div_y = 0
      global sub_
      global fr

      F__ = {}

      for i in sub_:
            SAM = {}
            for j in sub_:
                  sam = 0
                  for i_ in i:
                        for j_ in j:
                              if(i_ == j_):
                                    sam += 1
                  cad_i = len(i)
                  cad_j = len(j)
                  sam_x = sam/cad_i
                  sam_y = sam/cad_j
                  sam_ = sam_x * sam_y
                  SAM[frozenset(j)] = sam_
            F__[frozenset(i)] = SAM  #Similarity Measure of Elements
            
         
      for i in sub_:
            for j in sub_:
                  if (frozenset(i) in m1 and m1[frozenset(i)] != 0 and F__[frozenset(i)][frozenset(j)] != 0):
                        if(frozenset(i) not in m2):
                              m2[frozenset(i)] = 0
                        x = 2 * m1[frozenset(i)] / (m1[frozenset(i)] + m2[frozenset(i)])
                        x_ = m1[frozenset(i)] * math.log(F__[frozenset(i)][frozenset(j)] * x,2)
                        div_x += x_
                        
                  else:
                        x_ = 0

                  if (frozenset(j) in m2 and m2[frozenset(j)] != 0 and F__[frozenset(j)][frozenset(i)] != 0):
                        if(frozenset(j) not in m1):
                              m1[frozenset(j)] = 0
                        y = 2 * m2[frozenset(j)] / (m1[frozenset(j)] + m2[frozenset(j)])
                        y_ = m2[frozenset(j)] * math.log(F__[frozenset(j)][frozenset(i)] * y,2)
                        div_y += y_

                  else:
                        y_ = 0

      div_ = (div_x + div_y)/2
                  
      #print('Pan's MJSD:',div_)
      return div_

def D_pan(m1,m2):

      d_ = (1/2)*(Div_pan(m1,m1)+Div_pan(m2,m2)-2*Div_pan(m1,m2))/(Div_pan(m1,m1)+Div_pan(m2,m2))
      d__ = pow(abs(d_),1/2)

      #print('Pan's EMJSD:',d__)
      return d__


def MTD_pan(M,k):
      D__ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D__[p].append(D_pan(M[p],M[q]))
      #print('the matrix of EMJSD:\n',np.matrix(D__))
      return D__


#==========================================================================================

#=================================Xiao's BJS divergence====================================

def Div_bjs(m1,m2):
      div_ = 0
      global sub_
      global fr

      for i in range(0,fr):
            if(frozenset(sub_[i]) in m1 and m1[frozenset(sub_[i])] != 0):
                  if(frozenset(sub_[i]) not in m2):
                        m2[frozenset(sub_[i])] = 0
                  x = 2 * m1[frozenset(sub_[i])] / (m1[frozenset(sub_[i])] + m2[frozenset(sub_[i])])
                  x_ = m1[frozenset(sub_[i])] * math.log(x,2)
            else:
                  x_ = 0
            if(frozenset(sub_[i]) in m2 and m2[frozenset(sub_[i])] != 0):
                  if(frozenset(sub_[i]) not in m1):
                        m1[frozenset(sub_[i])] = 0
                  y = 2 * m2[frozenset(sub_[i])] / (m1[frozenset(sub_[i])] + m2[frozenset(sub_[i])])
                  y_ = m2[frozenset(sub_[i])] * math.log(y,2)
            else:
                  y_ = 0
            div_ += (x_ + y_)/2
      #print('Xiao's BJS:',div_)
      return div_


def MTD_bjs(M,k):
      D__ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D__[p].append(Div_bjs(M[p],M[q]))
      #print('the matrix of BJS:\n',np.matrix(D__))
      return D__



#==========================================================================================

#frame of discernment
def FRAME(M,k):
      s = 0
      Fra = []
      for fra in M[0]:
            Fra.append(fra)
      if (k > 1):
            for i in range(1,k):
                  for fra1 in M[i]: 
                        for fra2 in Fra: 
                              if (fra1 != fra2):
                                    s += 1
                              if (s == len(Fra)):
                                    Fra.append(fra1)
                        s = 0
            return sorted(Fra)
      else:
            return sorted(Fra)

#all of subsets
def Subset(f): #f:frame of discernment
      n = len(f)
      subset = []
      for i in range(0,2 ** n):
            sub = [] 
            for j in range(0,n):
                if(i>>j)%2:
                    sub.append(f[j])
            subset.append(sub)
      del(subset[0])
      subset_ = []
      for i in subset:
            subset_.append(set(i))
      return subset_

#=================================2.Fusion=========================================================

#==================================RFBD========================================================
#External weight of RFBD
def Weightex(M,k): #M:list
      Dp = []
      Sump = []
      Sumdp = 0
      for p in range(0,k): #p mass functions
            NDp = []  
            NNDp = [] 
            SDp = 0
            SSqu = 0
            
            for q in range(0,k): 
                  SDp += M[p][q]
            Sump.append(SDp)
            for q in range(0,k):
                  if (q != p):                        
                        ndp = M[p][q] / SDp
                        NDp.append(ndp) #Normalize each RFBD
                        nndp = (1 - ndp) / (k-2)
                        NNDp.append(nndp) #the negation of NRFBD

            for q in range(0,k-1):
                  SSqu += pow((NNDp[q] - NDp[q]),2)
            dp = math.sqrt(SSqu) #distance between NRFBD and NNRFBD
            Dp.append(dp)
            Sumdp += dp
            #print("-",p,"-Normalization of each RFBD:",NDp)
            #print("-",p,"-the negation of NRFBD:",NNDp)
      #print("distance:",Dp)
      Disp = [] 
      for p in range(0,k):
            Ndp = Dp[p] / Sumdp
            Disp.append(Ndp)
      #print("Normalize each distance：",Disp)


      SUPP = [] 
      CREDP = [] 
      Sumpp = 0
      for p in range(0,k):
            Avep = Sump[p] / (k - 1)
            Supp = 1/Avep #support degree
            SUPP.append(Supp)
            Sumpp += Supp
      for p in range(0,k):
            Credp = SUPP[p] / Sumpp
            CREDP.append(Credp)
      #print("credibility：",CREDP)


      WEXP = [] 
      sumwexp = 0
      for p in range(0,k):
            sumwexp += CREDP[p] * Disp[p]
      for p in range(0,k):
            Wexp = CREDP[p] * Disp[p] / sumwexp
            WEXP.append(Wexp)
      #print("External weight:",WEXP)
      return WEXP


#Internal weight
def Weightin(M,k): 
      global frame
      global sub_
      lengx = pow(2,len(frame)) - 1
      
      ep = 0 
      IVP = [] 
      IVp = 0
      Winp = 0
      WINP = []
      for p in range(0,k): #p
            mm = M[p]
            for ma in mm:
                  #Cui's improved belief entropy
                  e1 = 0
                  sam = {}
                  lenga = pow(2,len(ma)) - 1 #cardinality |.|
                  for i in range(0,len(sub_)):
                        k1 = 0
                        if (sub_[i] == ma or frozenset(sub_[i]) not in mm):
                              continue
                        for t1 in list(sub_[i]):
                              for t2 in list(ma):
                                    if(t1 == t2):
                                          k1 += 1
                        sam[frozenset(sub_[i])] = k1 
                  x1 = 0
                  for a in sam.values():
                        x1 += a / lengx
                  if(mm[ma] != 0):
                        x2 = mm[ma] / lenga * math.exp(x1)
                        e1 = mm[ma] * math.log(x2,2)
                  else:
                        e1 = 0
                  ep = ep - e1
                  
                  #print("Entropy:",ep)

            IVp = math.exp(ep)
            IVP.append(IVp) #Information volume
            ep = 0
      #print("Information volume:",IVP)

      sumIVP = 0
      for p in range(0,k): # Normalize the information volume
            sumIVP += IVP[p]

      for p in range(0,k):
            Winp = IVP[p] / sumIVP 
            WINP.append(Winp)
      #print("Internal weight:",WINP)
      return WINP


#Final weight
def fweight(M,k): 
      sumWeight = 0
      FWEIGHT = []
      EX = Weightex(MTD(M,k),k)
      IN = Weightin(M,k)
      for p in range(0,k):
            sumWeight += EX[p] * IN[p]
      for p in range(0,k):
            fweightp = EX[p] * IN[p] / sumWeight
            FWEIGHT.append(fweightp)
      #print("Final weight:",FWEIGHT)
      return FWEIGHT


#The weighted mass function
def remass(M,k): 
      ww = fweight(M,k)
      rema = {} 
      for p in range(0,k):
            mm = []
            for m in M[p].items():
                  m1 = list(m)
                  sorted(m1[0])
                  m1[1] *= ww[p]
                  mm.append(m1)

            for i in range(0,len(mm)):
                  if mm[i][0] in rema: 
                        rema[mm[i][0]] += mm[i][1]
                  else:
                        rema[mm[i][0]] = mm[i][1]
      
      #print('The weighted mass function:',rema)
      return rema


#Dempster's combination rule
def dsmass(M,k):  
      mm1 = remass(M,k) 
      DSM = [] 
      DSM.append(mm1)
      
      for p in range(0,k-1): #k-1 times
            dsm = {} 
            mm = DSM[p] 
            kk = 0
            flag = 0
            for i in mm:
                  for j in mm1:
                        sam = [] 
                        for x1 in i:
                              for x2 in j:
                                    if(x1 == x2):
                                          sam.append(x1) 
                                          flag += 1 
                                          sam1 = frozenset(sam)
                        #Conflict
                        if(flag == 0):
                              kk += mm[i] * mm1[j]
                        else: 
                              if sam1 in dsm:
                                    dsm[sam1] += mm[i] * mm1[j]
                              else:
                                    dsm[sam1] = mm[i] * mm1[j]
                        flag = 0
            k1 =1/(1-kk)
            for maa in dsm:
                  dsm[maa] *= k1 
            DSM.append(dsm)
             
      #Finally
      #print('Fusion result:',DSM[k-1])
      return DSM[k-1]

#======================================RB====================================================

#External weight
def Weightex_xiao(M,k): 
      
      Sump = [] 
      for p in range(0,k):             
            SDp = 0
            for q in range(0,k):
                  SDp += M[p][q]
            Sump.append(SDp)
            
      #credibility
      SUPP = [] 
      CREDP = [] 
      Sumpp = 0
      for p in range(0,k):
            Avep = Sump[p] / (k - 1)
            Supp = 1/Avep 
            SUPP.append(Supp)
            Sumpp += Supp
      for p in range(0,k):
            Credp = SUPP[p] / Sumpp
            CREDP.append(Credp)

      return CREDP

#The weighted mass function
def remass_xiao(M,k): 
      ww = Weightex_xiao(MTD_xiao(M,k),k)
      rema = {} 
      for p in range(0,k):
            mm = []
            for m in M[p].items():
                  m1 = list(m)
                  sorted(m1[0])
                  m1[1] *= ww[p]
                  mm.append(m1)

            for i in range(0,len(mm)):
                  if mm[i][0] in rema: 
                        rema[mm[i][0]] += mm[i][1]
                  else:
                        rema[mm[i][0]] = mm[i][1]
      
      #print('The weighted mass function:',rema)
      return rema


def dsmass_xiao(M,k):  
      mm1 = remass_xiao(M,k) 
      DSM = [] 
      DSM.append(mm1)
      
      for p in range(0,k-1): 
            dsm = {} 
            mm = DSM[p] 
            kk = 0
            flag = 0
            for i in mm:
                  for j in mm1:
                        sam = [] 
                        for x1 in i:
                              for x2 in j:
                                    if(x1 == x2):
                                          sam.append(x1) 
                                          flag += 1 
                                          sam1 = frozenset(sam)
                        if(flag == 0):
                              kk += mm[i] * mm1[j] 
                        else: 
                              if sam1 in dsm:
                                    dsm[sam1] += mm[i] * mm1[j]
                              else:
                                    dsm[sam1] = mm[i] * mm1[j]
                        flag = 0
            k1 =1/(1-kk)
            for maa in dsm:
                  dsm[maa] *= k1 
            DSM.append(dsm)

      #print('Fusion result:',DSM[k-1])
      return DSM[k-1]                 
#==========================================================================================

#============================================MRBD==============================================

#Internal weight
def Weightin_chen(M,k):
      global frame
      ep = 0 
      iqp = 0
      EP = []      
      IQP = []
      NewIQP = []
      
      for p in range(0,k): 
            mm = []
            for m in M[p].items():
                  sorted(m[0])
                  mm.append(m)
            for ma in mm:
                  e1 = 0
                  lenga = pow(2,len(ma[0])) - 1
                  if(ma[1] != 0):
                        x2 = ma[1] / lenga
                        e1 = ma[1] * math.log(x2,2)
                  else:
                        e1 = 0
                  ep = ep - e1
                  iqp += pow(ma[1],2)

            EP.append(ep)
            ep = 0
            #算IQ
            IQP.append(iqp)  #IQ
            iqp = 0

      sumEP = 0
      for p in range(0,k):
            sumEP += EP[p]
      for p in range(0,k):
            uncerp = EP[p] / sumEP 
            infp = 1-uncerp
            newIQ = infp * IQP[p]
            NewIQP.append(newIQ)
      
      #print("Internal weight:",NewIQP)
      return NewIQP


#Final weight
def fweight_chen(M,k):
      sumWeight = 0
      FWEIGHT = []
      EX = Weightex_xiao(MTD_chen(M,k),k) 
      IN = Weightin_chen(M,k)
      for p in range(0,k):
            sumWeight += EX[p] * IN[p]
      for p in range(0,k):
            fweightp = EX[p] * IN[p] / sumWeight
            FWEIGHT.append(fweightp)
      #print("Final weight:",FWEIGHT)
      return FWEIGHT


def remass_chen(M,k): 
      ww = fweight_chen(M,k)
      rema = {} 
      for p in range(0,k):
            mm = []
            for m in M[p].items():
                  m1 = list(m)
                  sorted(m1[0])
                  m1[1] *= ww[p]
                  mm.append(m1)
            for i in range(0,len(mm)):
                  if mm[i][0] in rema: 
                        rema[mm[i][0]] += mm[i][1]
                  else:
                        rema[mm[i][0]] = mm[i][1]

      return rema


def dsmass_chen(M,k):  
      mm1 = remass_chen(M,k) 
      DSM = [] 
      DSM.append(mm1)
      
      for p in range(0,k-1): 
            dsm = {} 
            mm = DSM[p] 
            kk = 0
            flag = 0
            for i in mm:
                  for j in mm1:
                        sam = [] 
                        for x1 in i:
                              for x2 in j:
                                    if(x1 == x2):
                                          sam.append(x1) 
                                          flag += 1 
                                          sam1 = frozenset(sam)

                        if(flag == 0):
                              kk += mm[i] * mm1[j] 
                        else: 
                              if sam1 in dsm:
                                    dsm[sam1] += mm[i] * mm1[j]
                              else:
                                    dsm[sam1] = mm[i] * mm1[j]
                        flag = 0
            k1 =1/(1-kk)
            for maa in dsm:
                  dsm[maa] *= k1 
            DSM.append(dsm)

      #print('Fusion result:',DSM[k-1])
      return DSM[k-1]                  
#==========================================================================================

#============================================BJS============================================

#Internal weight
def Weightin_bjs(M,k):      
      ep = 0 
      IVP = [] 
      IVp = 0
      Winp = 0
      WINP = []
      for p in range(0,k): 
            mm = []
            for m in M[p].items():
                  sorted(m[0])
                  mm.append(m) 
            for ma in mm:
                  e1 = 0                  
                  lenga = pow(2,len(ma[0])) - 1 
                  
                  if(ma[1] != 0):
                        x2 = ma[1] / lenga
                        e1 = ma[1] * math.log(x2,2)
                  else:
                        e1 = 0
                  ep = ep - e1
                  
            IVp = math.exp(ep)
            IVP.append(IVp) 
            ep = 0

      sumIVP = 0
      for p in range(0,k):
            sumIVP += IVP[p]

      for p in range(0,k):
            Winp = IVP[p] / sumIVP 
            WINP.append(Winp)
      #print("Internal weight:",WINP)
      return WINP


#Final weight
def fweight_bjs(M,k):
      sumWeight = 0
      FWEIGHT = []
      EX = Weightex_xiao(MTD_bjs(M,k),k) 
      IN = Weightin_bjs(M,k) 
      for p in range(0,k):
            sumWeight += EX[p] * IN[p]
      for p in range(0,k):
            fweightp = EX[p] * IN[p] / sumWeight
            FWEIGHT.append(fweightp)
      #print("Final:",FWEIGHT)
      return FWEIGHT



def remass_bjs(M,k):
      ww = fweight_bjs(M,k)
      rema = {} 
      for p in range(0,k):
            mm = []
            for m in M[p].items():
                  m1 = list(m)
                  sorted(m1[0])
                  m1[1] *= ww[p]
                  mm.append(m1)

            for i in range(0,len(mm)):
                  if mm[i][0] in rema: 
                        rema[mm[i][0]] += mm[i][1]
                  else:
                        rema[mm[i][0]] = mm[i][1]
      
      #print('Weighted mass function:',rema)
      return rema


def dsmass_bjs(M,k):  
      mm1 = remass_bjs(M,k) 
      DSM = [] 
      DSM.append(mm1)
      
      for p in range(0,k-1): 
            dsm = {} 
            mm = DSM[p] 
            kk = 0
            flag = 0
            for i in mm:
                  for j in mm1:
                        sam = []
                        for x1 in i:
                              for x2 in j:
                                    if(x1 == x2):
                                          sam.append(x1) 
                                          flag += 1
                                          sam1 = frozenset(sam)

                        if(flag == 0):
                              kk += mm[i] * mm1[j] 
                        else: 
                              if sam1 in dsm:
                                    dsm[sam1] += mm[i] * mm1[j]
                              else:
                                    dsm[sam1] = mm[i] * mm1[j]
                        flag = 0
            k1 =1/(1-kk)
            for maa in dsm:
                  dsm[maa] *= k1 
            DSM.append(dsm)

      #print('Fusion result:',DSM[k-1])
      return DSM[k-1]

#========================================================================================
#============================================EMJSD============================================

#Internal weight
def fweight_pan(M,k): 

      Sump = [] 
      for p in range(0,k):             
            SDp = 0
            for q in range(0,k): 
                  SDp += M[p][q]
            Sump.append(SDp)
      SSum = 0 
      for i in Sump:
            SSum += i
      
      
      #D
      AVE = []
      DAVE = [] 
      
      for p in range(0,k):
            Avep = Sump[p] / SSum #SMi
            AVE.append(Avep)
      SAve = 0
      for p in range(0,k):
            SAve += 1 - AVE[p]
            
      for p in range(0,k):
            DAvep = (1 - AVE[p])/SAve
            DAVE.append(DAvep)

      S = 1/k #S
      SC = 0 
      CP = [] 
      for p in range(0,k):
            SC += DAVE[p]*S
      for p in range(0,k):
            Cp = DAVE[p]*S/SC
            CP.append(Cp)
      

      return CP



def remass_pan(M,k): 
      ww = fweight_pan(MTD_pan(M,k),k)
      rema = {} 
      for p in range(0,k):
            mm = []
            for m in M[p].items():
                  m1 = list(m)
                  sorted(m1[0])
                  m1[1] *= ww[p]
                  mm.append(m1)

            for i in range(0,len(mm)):
                  if mm[i][0] in rema:
                        rema[mm[i][0]] += mm[i][1]
                  else:
                        rema[mm[i][0]] = mm[i][1]

      return rema


def dsmass_pan(M,k):  
      mm1 = remass_pan(M,k) 
      DSM = [] 
      DSM.append(mm1)
      
      for p in range(0,k-1): 
            dsm = {} 
            mm = DSM[p] 
            kk = 0
            flag = 0
            for i in mm:
                  for j in mm1:
                        sam = [] 
                        for x1 in i:
                              for x2 in j:
                                    if(x1 == x2):
                                          sam.append(x1) 
                                          flag += 1 
                                          sam1 = frozenset(sam)

                        if(flag == 0):
                              kk += mm[i] * mm1[j] 
                        else: 
                              if sam1 in dsm:
                                    dsm[sam1] += mm[i] * mm1[j]
                              else:
                                    dsm[sam1] = mm[i] * mm1[j]
                        flag = 0
            k1 =1/(1-kk)
            for maa in dsm:
                  dsm[maa] *= k1 
            DSM.append(dsm)

      #print('Fusion result:',DSM[k-1])
      return DSM[k-1]
#==========================================================================================

#Iris dataset
def yuanweihua_in(X,y,train_): #Get BPA
      test_per = round(1-train_,2)
      X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_per,
                                                 stratify=y,  
                                                 shuffle=True, 
                                                 random_state=1)   


      train_qua = int(50*train_)
      test_qua = int(50*test_per)


      #---training set---#
      w1 = X_train[df['class'] == 0]
      w2 = X_train[df['class'] == 1]
      w3 = X_train[df['class'] == 2]

      #class=0
      x1 = w1[['SepalLen']]
      x11=x1.values.tolist()

      x2 = w1[['SepalWid']]
      x21=x2.values.tolist()

      x3 = w1[['PetalLen']]
      x31=x3.values.tolist()

      x4 = w1[['PetalWid']]
      x41=x4.values.tolist()


      #class=1
      y1 = w2[['SepalLen']]
      y11=y1.values.tolist()

      y2 = w2[['SepalWid']]
      y21=y2.values.tolist()

      y3 = w2[['PetalLen']]
      y31=y3.values.tolist()

      y4 = w2[['PetalWid']]
      y41=y4.values.tolist()

      #class=2
      z1 = w3[['SepalLen']]
      z11=z1.values.tolist()

      z2 = w3[['SepalWid']]
      z21=z2.values.tolist()

      z3 = w3[['PetalLen']]
      z31=z3.values.tolist()

      z4 = w3[['PetalWid']]
      z41=z4.values.tolist()

      #min-max
      A3 = [[] for i in range(0,4)] #max-list
      A1 = [[] for i in range(0,4)] #min-list


      #dada=[SepalLen，SepalWid，PetalLen，PetalWid]
      dada = [[x11,y11,z11],[x21,y21,z21],[x31,y31,z31],[x41,y41,z41]]

      for i in range(0,4):
            for j in range(0,3):
                  maxx = max(dada[i][j])[0]
                  A3[i].append(maxx)
                  minx = min(dada[i][j])[0]
                  A1[i].append(minx)

      FUZ = [] #Triangular fuzzy number
      for i in range(0,4):
            sa = [A1[i][0],A3[i][0]]
            sb = [A1[i][1],A3[i][1]]
            sc = [A1[i][2],A3[i][2]]

            sab = []
            sac = []
            sbc = []
            sabc = []

            #---sab---#
            if(A1[i][1]<A1[i][0]):
                  if(A3[i][1]>A3[i][0]):
                        sab = [A1[i][0],A3[i][0]]
                  else:
                        if(A3[i][1]<A1[i][0]):
                              sab = []
                        else:
                              sab = [A1[i][0],A3[i][1]]
            else:
                  if(A3[i][1]>A3[i][0]):
                        if(A1[i][1]>A3[i][0]):
                              sab = []
                        else:
                              sab = [A1[i][1],A3[i][0]]
                  else:
                        sab = [A1[i][1],A3[i][1]]
            #print("sab:",sab)
      
            #---sac---#     
            if(A1[i][2]<A1[i][0]):
                  if(A3[i][2]>A3[i][0]):
                        sac = [A1[i][0],A3[i][0]]
                  else:
                        if(A3[i][2]<A1[i][0]):
                              sac = []
                        else:
                              sac = [A1[i][0],A3[i][2]]
            else:
                  if(A3[i][2]>A3[i][0]):
                        if(A1[i][2]>A3[i][0]):
                              sac = []
                        else:
                              sac = [A1[i][2],A3[i][0]]
                  else:
                        sac = [A1[i][2],A3[i][2]]

            #---sbc---#
            if(A1[i][2]<A1[i][1]):
                  if(A3[i][2]>A3[i][1]):
                        sbc = [A1[i][1],A3[i][1]]
                  else:
                        if(A3[i][2]<A1[i][1]):
                              sbc = []
                        else:
                              sbc = [A1[i][1],A3[i][2]]
            else:
                  if(A3[i][2]>A3[i][1]):
                        if(A1[i][2]>A3[i][1]):
                              sbc = []
                        else:
                              sbc = [A1[i][2],A3[i][1]]
                  else:
                        sbc = [A1[i][2],A3[i][2]]

            #---sabc---#
            if(len(sab)==0 or len(sac)==0 or len(sbc)==0):
                  sabc=[]
            elif(min(sab[1]-sab[0],sac[1]-sac[0],sbc[1]-sbc[0])==sab[1]-sab[0]):
                  sabc = sab
            elif(min(sab[1]-sab[0],sac[1]-sac[0],sbc[1]-sbc[0])==sac[1]-sac[0]):
                  sabc = sac
            elif(min(sab[1]-sab[0],sac[1]-sac[0],sbc[1]-sbc[0])==sbc[1]-sbc[0]):
                  sabc = sbc
            else:
                  print("error")

            fuz = [sa,sb,sc,sab,sac,sbc,sabc]
            FUZ.append(fuz)


      #--- test set ---#
      Test = [] 
      for i in range(0,test_qua):
            test_a = X_test.iloc[i:i+1].values.tolist()
            test_t = y_test.iloc[i:i+1].values.tolist()
            test_a.append(test_t)
            Test.append(test_a)

      P = []
      Q = []
      R = []
      T = []
      MB = [] #target

      for t in Test:
            P.append(t[0][0])
            Q.append(t[0][1])
            R.append(t[0][2])
            T.append(t[0][3])
            MB.append(t[1][0])
      SJ = [P,Q,R,T] 
      BPAm = [[{} for i in range(0,4)] for j in range(0,test_qua)]
      

      for c in range(0,test_qua):
            #construct BPA for each data of test set
            target = 0
            alpha = 5
            for i in range(0,4): 

                  RX = []
                  RY = []
                  SS = [] 
                  NSS = [] 
                  zero = 0
                  for f in FUZ[i]:
                        if(len(f)==0):
                              SS.append(zero)
                              continue

                        D2 = pow((f[0]+f[1])/2 -(SJ[i][c]+SJ[i][c])/2,2) + pow((f[1]-f[0])+(SJ[i][c]-SJ[i][c]),2)/12
                        D = math.sqrt(D2)                        
                        S = 1 / (1+alpha*D)
                        SS.append(S)
                        
                  sums = 0
                  for jj in SS:
                        sums += jj
                  for jjj in range(0,len(SS)):
                        Si = SS[jjj]/sums
                        NSS.append(Si)

                  BPAm[c][i][frozenset({'a'})] = NSS[0]
                  BPAm[c][i][frozenset({'b'})] = NSS[1]
                  BPAm[c][i][frozenset({'c'})] = NSS[2]
                  if(NSS[3]!=0):
                        BPAm[c][i][frozenset({'a','b'})] = NSS[3]
                  if(NSS[4]!=0):
                        BPAm[c][i][frozenset({'a','c'})] = NSS[4]
                  if(NSS[5]!=0):
                        BPAm[c][i][frozenset({'b','c'})] = NSS[5]
                  if(NSS[6]!=0):
                        BPAm[c][i][frozenset({'a','b','c'})] = NSS[6]
                      
      return BPAm,MB

def methodff(train_,BPAm,MB,method): #Calculate the accuracy of the Iris dataset
      global sub_
      global fr
      global GG
      global FF
      global mubiao
      #use different methods
      test_per = round(1-train_,2)
      test_qua = int(50*test_per)
      test_se = 0
      test_ve = 0
      test_vi = 0
      for i in MB:
            if(i == 0):
                  test_se += 1
            elif(i == 1):
                  test_ve += 1
            else:
                  test_vi += 1
      cor = 0 #Correct
      cor_se = 0 #Setosa
      cor_ve = 0 #Versicolor
      cor_vi = 0 #Virginica
      
      cor_per = 0 #Accuracy
      cor_se_per = 0
      cor_ve_per = 0
      cor_vi_per = 0
      COR = []
      
      for c in range(0,test_qua):
            if(method == "proposed"):
                  dict_fe = dsmass([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
            elif(method == "xiao"):
                  dict_fe = dsmass_xiao([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
            elif(method == "pan"):
                  dict_fe = dsmass_pan([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
            elif(method == "chen"):
                  dict_fe = dsmass_chen([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
            elif(method == "bjs"):
                  dict_fe = dsmass_bjs([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
            else:
                  print("error")

            max_fe = max(zip(dict_fe.values(),dict_fe.keys()))
      
      
            if(max_fe[1] == frozenset({'a'})):
                  target = 0
            elif(max_fe[1] == frozenset({'b'})):
                  target = 1
            elif(max_fe[1] == frozenset({'c'})):
                  target = 2
            else:
                  target = 3
            if(target == mubiao[c]):
                  cor += 1
                  if(target == 0):
                        cor_se += 1
                  elif(target == 1):
                        cor_ve += 1
                  else:
                        cor_vi += 1


            if(c == test_qua-1):
                  cor_per = cor / test_qua
                  cor_se_per = cor_se / test_se
                  cor_ve_per = cor_ve / test_ve
                  cor_vi_per = cor_vi / test_vi
                  COR.append(cor_per)
                  COR.append(cor_se_per)
                  COR.append(cor_ve_per)
                  COR.append(cor_vi_per)

      return COR
      
#mean average
def tomean_(corm):
      sumcor = 0
      for i in range(0,len(corm)):
            sumcor += corm[i]
      avecor = sumcor / len(corm)
      return avecor



#=========================================================================================================================================================

if __name__ == "__main__":

      #Please follow the instructions to add or remove comments for the following test content

#=================================Experiment of Iris dataset========================================
      cols_name = ['SepalLen','SepalWid','PetalLen','PetalWid','class']
      df = pd.read_csv('iris.csv',names=cols_name)
      df['class'] = pd.Categorical(df['class'])
      df['class'] = df['class'].cat.codes 

      
      frame = ['a', 'b', 'c']
      sub_ = Subset(frame)
      fr = len(sub_)
      GG = ffgg(sub_)[0]
      FF = ffgg(sub_)[1]
      

      #training set and test set
      X = df[['SepalLen','SepalWid','PetalLen','PetalWid']]
      y = df['class']
      #percentage
      train__ = [0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20]

      
      COR_proposed = [] 
      COR_proposed_se = []
      COR_proposed_ve = []
      COR_proposed_vi = []
      COR_xiao = [] 
      COR_xiao_se = []
      COR_xiao_ve = []
      COR_xiao_vi = []
      COR_pan = [] 
      COR_pan_se = []
      COR_pan_ve = []
      COR_pan_vi = []
      COR_chen = [] 
      COR_chen_se = []
      COR_chen_ve = []
      COR_chen_vi = []
      COR_bjs = [] 
      COR_bjs_se = []
      COR_bjs_ve = []
      COR_bjs_vi = []

      
      for i in range(0,len(train__)):
            
            lili = yuanweihua_in(X,y,train__[i]) 
            BPAm_ = lili[0] 
            mubiao = lili[1]

            jingdu_p = methodff(train__[i],BPAm_,mubiao,"proposed")
            COR_proposed.append(jingdu_p[0])
            COR_proposed_se.append(jingdu_p[1])
            COR_proposed_ve.append(jingdu_p[2])
            COR_proposed_vi.append(jingdu_p[3])
            #print("RB:")
            jingdu_x = methodff(train__[i],BPAm_,mubiao,"xiao")
            COR_xiao.append(jingdu_x[0])
            COR_xiao_se.append(jingdu_x[1])
            COR_xiao_ve.append(jingdu_x[2])
            COR_xiao_vi.append(jingdu_x[3])
            #print("EMJSD:")
            jingdu_s = methodff(train__[i],BPAm_,mubiao,"pan")
            COR_pan.append(jingdu_s[0])
            COR_pan_se.append(jingdu_s[1])
            COR_pan_ve.append(jingdu_s[2])
            COR_pan_vi.append(jingdu_s[3])
            #print("MRBD:")
            jingdu_c = methodff(train__[i],BPAm_,mubiao,"chen")
            COR_chen.append(jingdu_c[0])
            COR_chen_se.append(jingdu_c[1])
            COR_chen_ve.append(jingdu_c[2])
            COR_chen_vi.append(jingdu_c[3])
            #print("BJS:")
            jingdu_ds = methodff(train__[i],BPAm_,mubiao,"bjs")
            COR_bjs.append(jingdu_ds[0])
            COR_bjs_se.append(jingdu_ds[1])
            COR_bjs_ve.append(jingdu_ds[2])
            COR_bjs_vi.append(jingdu_ds[3])
      
      print("proposed-se:",tomean_(COR_proposed_se))
      print("proposed-ve:",tomean_(COR_proposed_ve))
      print("proposed-vi:",tomean_(COR_proposed_vi))
      print("xiao-se:",tomean_(COR_xiao_se))
      print("xiao-ve:",tomean_(COR_xiao_ve))
      print("xiao-vi:",tomean_(COR_xiao_vi))
      print("pan-se:",tomean_(COR_pan_se))
      print("pan-ve:",tomean_(COR_pan_ve))
      print("pan-vi:",tomean_(COR_pan_vi))
      print("chen-se:",tomean_(COR_chen_se))
      print("chen-ve:",tomean_(COR_chen_ve))
      print("chen-vi:",tomean_(COR_chen_vi))
      print("bjs-se:",tomean_(COR_bjs_se))
      print("bjs-ve:",tomean_(COR_bjs_ve))
      print("bjs-vi:",tomean_(COR_bjs_vi))
      #---------------------------------------
      #print("proposed:",COR_proposed)
      #print("xiao:",COR_xiao)
      #print("pan:",COR_pan)
      #print("chen:",COR_chen)
      #print("bjs:",COR_bjs)
      #---------------------------------------

      
      
      #Fig6

      #Results:
      COR_proposed = [0.9, 0.9166666666666666, 0.9333333333333333, 0.9411764705882353, 1.0, 0.9090909090909091, 0.96, 0.9629629629629629, 0.9666666666666667, 0.9375, 0.9714285714285714, 0.8918918918918919, 0.975]
      COR_xiao = [1.0, 0.9166666666666666, 0.9333333333333333, 0.9411764705882353, 1.0, 0.9090909090909091, 0.96, 0.9259259259259259, 0.9333333333333333, 0.875, 0.8857142857142857, 0.8648648648648649, 0.925]
      COR_pan = [1.0, 0.9166666666666666, 0.9333333333333333, 0.9411764705882353, 1.0, 0.9090909090909091, 0.92, 0.9259259259259259, 0.9333333333333333, 0.875, 0.8857142857142857, 0.8648648648648649, 0.925]
      COR_chen = [1.0, 0.9166666666666666, 0.9333333333333333, 0.9411764705882353, 1.0, 0.8636363636363636, 0.92, 0.9259259259259259, 0.9, 0.875, 0.8571428571428571, 0.8918918918918919, 0.9]
      COR_bjs = [0.9, 0.9166666666666666, 0.9333333333333333, 0.8235294117647058, 0.85, 0.9090909090909091, 0.88, 0.8888888888888888, 0.9, 0.8125, 0.8285714285714286, 0.8648648648648649, 0.925]

      x = np.linspace(0.2,0.8,13)

      y1=COR_proposed #RFBD
      y2=COR_xiao #RB
      y3=COR_pan #EMJSD
      y4=COR_chen #MRBD
      y5=COR_bjs #BJS

      x_major_locator=MultipleLocator(0.1)
      y_major_locator=MultipleLocator(0.1)
      ax=plt.gca()
      ax.xaxis.set_major_locator(x_major_locator)
      ax.yaxis.set_major_locator(y_major_locator)
      
      plt.plot(x,y4,marker = 'o',color='green',label='MRBD')
      plt.plot(x,y3,marker = '+',color='red',label='EMJSD')
      plt.plot(x,y2,marker = 'x',color='blue',label='RB',linestyle = '--')
      plt.plot(x,y5,marker = '*',color='purple',label='BJS')
      plt.plot(x,y1,marker = '>',color='orange',label='RFBD',linestyle = '--')
      plt.xlabel('Percentage of testing data set')
      plt.ylabel('Recognition accuracy')
      plt.xlim(0.17,0.83)
      plt.ylim(0.35,1.05)
      ticks = ax.set_xticks(x)
      labels = ax.set_xticklabels(['20%','25%','30%','35%','40%','45%','50%','55%','60%','65%','70%','75%','80%'],rotation = 0,fontsize = 'small') 
      plt.legend()
      plt.show()

#===========================================Numerical examples======================================================================

#===============================Example 1========================================

      M1 = MassFunction([({'a'}, 0.3), ({'b'}, 0.2), ({'c'}, 0.2), ({'d'}, 0.3)]) 
      M1_ = {frozenset({'a'}):0.3, frozenset({'b'}):0.2, frozenset({'c'}):0.2, frozenset({'d'}):0.3}

      M2 = MassFunction([({'a'}, 0.3), ({'b'}, 0.2), ({'c'}, 0.2), ({'d'}, 0.3)])
      M2_ = {frozenset({'a'}):0.3, frozenset({'b'}):0.2, frozenset({'c'}):0.2, frozenset({'d'}):0.3}
      
#===============================Example 2========================================
      
      M3 = MassFunction([({'a'}, 0.3), ({'b'}, 0.2), ({'c'}, 0.2), ({'a' ,'b', 'c'}, 0.3)])
      M3_ = {frozenset({'a'}):0.3, frozenset({'b'}):0.2, frozenset({'c'}):0.2, frozenset({'a', 'b', 'c'}):0.3}

      M4 = MassFunction([({'a'}, 0.3), ({'b'}, 0.2), ({'c'}, 0.2), ({'a' ,'b', 'c'}, 0.3)])
      M4_ = {frozenset({'a'}):0.3, frozenset({'b'}):0.2, frozenset({'c'}):0.2, frozenset({'a', 'b', 'c'}):0.3}

#===============================Example 3========================================

      M5 = MassFunction([({'a','c'}, 0.90), ({'b','d'}, 0.10)])
      M5_ = {frozenset({'a','c'}):0.90, frozenset({'b','d'}):0.10}

      M6 = MassFunction([({'a','c'}, 0.10), ({'b','d'}, 0.90)])
      M6_ = {frozenset({'a','c'}):0.10, frozenset({'b','d'}):0.90}

#===============================Example 4========================================
      
      M7 = MassFunction([({'b'}, 0.90), ({'a','b','c','d'}, 0.10)]) 
      M7_ = {frozenset({'b'}):0.90, frozenset({'a','b','c','d'}):0.10}

      M8 = MassFunction([({'b'}, 0.10), ({'a','b','c','d'}, 0.90)])
      M8_ = {frozenset({'b'}):0.10, frozenset({'a','b','c','d'}):0.90}
      
#===============================Example 5========================================
      
      M9 = MassFunction([({'b','d'}, 0.90), ({'a','b','c','d','e','f','g','h'}, 0.10)]) 
      M9_ = {frozenset({'b','d'}):0.90, frozenset({'a','b','c','d','e','f','g','h'}):0.10}

      M10 = MassFunction([({'b','d'}, 0.10), ({'a','b','c','d','e','f','g','h'}, 0.90)]) 
      M10_ = {frozenset({'b','d'}):0.10, frozenset({'a','b','c','d','e','f','g','h'}):0.90}


#================================================================================
      frame = FRAME([M1.frame(),M2.frame()],2)
      #frame = FRAME([M3.frame(),M4.frame()],2)
      #frame = FRAME([M5.frame(),M6.frame()],2)
      #frame = FRAME([M7.frame(),M8.frame()],2)
      #frame = FRAME([M9.frame(),M10.frame()],2)
      sub_ = Subset(frame)
      fr = len(sub_)

      print(MTD([M1_,M2_],2)) #RFBD
      print(MTD_chen([M1_,M2_],2)) #MRBD
      print(MTD_xiao([M1_,M2_],2)) #RB
      print(MTD_bjs([M1_,M2_],2)) #BJS
      print(MTD_pan([M1_,M2_],2)) #EMJSD



#===============================Example 6========================================
      
      MZ = [{'a'},
            {'a', 'b'},
            {'a', 'b', 'c'},
            {'a', 'b', 'c','d'},
            {'a', 'b', 'c','d','e'},
            {'a', 'b', 'c','d','e','f'},
            {'a', 'b', 'c','d','e','f','g'},
            {'a', 'b', 'c','d','e','f','g','h'},
            {'a', 'b', 'c','d','e','f','g','h','i'},
            {'a', 'b', 'c','d','e','f','g','h','i','j'}]


      MM = []
      MM_chen = []
      MM_song = []
      MM_bjs = []
      MM_pan = []
      for i in range(0,len(MZ)):
            MD = sorted(MZ[i])
            M11 = MassFunction([({'b'}, 0.05), (MD, 0.95)]) 
            M11_ = {frozenset({'b'}):0.05, frozenset(MD):0.95}

            M12 = MassFunction([({'b'}, 0.95), (MD, 0.05)]) 
            M12_ = {frozenset({'b'}):0.95, frozenset(MD):0.05}
            frame = FRAME([M11.frame(),M12.frame()],2)
            sub_ = Subset(frame)
            fr = len(sub_)

            MM.append(MTD([M9_,M10_],2)[0][1])
            MM_xiao.append(MTD_xiao([M9_,M10_],2)[0][1])
            MM_chen.append(MTD_chen([M9_,M10_],2)[0][1])
            MM_bjs.append(MTD_bjs([M9_,M10_],2)[0][1])
            MM_pan.append(MTD_pan([M9_,M10_],2)[0][1])
      print("MM_chen:",MM_chen)
      print("MM_bjs:",MM_bjs)
      print("MM_xiao:",MM_xiao)
      print("pan:",MM_pan)
      print("propose:",MM)


      #Results
      MM = [1.194657308925069, 0.31842805019433507, 0.47645371049772217, 0.5704423173752836, 0.6325876675933586, 0.6759608783965123, 0.7072184161225014, 0.7302552901596331, 0.7475480990340648, 0.7607495055522091]
      MM_chen = [3.8231347620992264, 4.174137252729227, 4.210927514045822, 4.22326065350032, 4.229427368083758, 4.233127397154833, 4.235594083202956, 4.2373560018087595, 4.238677440763114, 4.239705226616499]
      MM_bjs = [0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436]
      MM_xiao = [1.1946573089250687, 0.5973286544625345, 0.6897357188972442, 0.7315752060882262, 0.7555676239140826, 0.7711479769819305, 0.7820867925264978, 0.7901915353410044, 0.7964382059415458, 0.8014004857610418]
      MM_pan = [0.8447502843349883, 0.7711479769819601, 0.6235369239568782, 0.5682401147799955, 0.5393871421030927, 0.5214046091279334, 0.5087325386030257, 0.4989769331665077, 0.49099074599206266, 0.48418000425441227]

      #Fig1a & Fig1b
      X = np.linspace(1,len(MZ),len(MZ))
      Y1 = MM
      Y2 = MM_xiao
      Y3 = MM_chen
      Y4 = MM_pan
      Y5 = MM_bjs

      plt.plot(X,Y1,marker = '>',markersize=4,color='orange',label='RFBD')
      plt.plot(X,Y2,marker = '*',markersize=4,color='blue',linestyle='--',label='RB')
      plt.plot(X,Y3,marker = 'o',markersize=4,color='green',label='MRBD')
      plt.plot(X,Y4,marker = '+',markersize=4,color='red',linestyle='--',label='EMJSD')
      plt.plot(X,Y5,marker = 'x',markersize=4,color='purple',label='BJS')
      
      plt.xlabel('t')
      plt.ylabel('The divergence measure')
      
      x_major_locator=MultipleLocator(1)
      y_major_locator=MultipleLocator(0.1)
      ax=plt.gca()
      ax.xaxis.set_major_locator(x_major_locator)
      ax.yaxis.set_major_locator(y_major_locator)
      plt.xlim(0.5,10.5)
      plt.ylim(0.08,1.4)
      
      plt.legend()
      plt.show()


#===============================Example 7========================================
      
      
      MZ_ = [{'a'},
             {'a', 'b'},
             {'a', 'b', 'c'},
             {'a', 'b', 'c','d'},
             {'a', 'b', 'c','d','e'},
             {'a', 'b', 'c','d','e','f'},
             {'a', 'b', 'c','d','e','f','g'},
             {'b', 'c','d','e','f','g'},
             {'c', 'd','e','f','g'},
             {'d', 'e','f','g'},
             {'e','f','g'},
             {'f','g'},
             {'g'}]
      
      for i in range(0,len(MZ_)):
            MD = sorted(MZ_[i])
            M13 = MassFunction([({'a'}, 0.05), (MD, 0.95)]) 
            M13_ = {frozenset({'a'}):0.05, frozenset(MD):0.95}

            M14 = MassFunction([({'g'}, 1.00)]) 
            M14_ = {frozenset({'g'}):1.00}
            frame = FRAME([M9.frame(),M10.frame()],2)
            sub_ = Subset(frame)
            fr = len(sub_)

            MM.append(MTD([M9_,M10_],2)[0][1])
            MM_xiao.append(MTD_xiao([M9_,M10_],2)[0][1])

      print("proposed:",MM)
      print("xiao:",MM_xiao)


      X = np.linspace(1,len(MZ_),len(MZ_))

      #Results
      Y1_ = [1.194657308925069, 0.31842805019433507, 0.47645371049772217, 0.5704423173752836, 0.6325876675933586, 0.6759608783965123, 0.7072184161225014, 0.7302552901596331, 0.7475480990340648, 0.7607495055522091]
      Y2_ = [1.1946573089250687, 0.5973286544625345, 0.6897357188972442, 0.7315752060882262, 0.7555676239140826, 0.7711479769819305, 0.7820867925264978, 0.7901915353410044, 0.7964382059415458, 0.8014004857610418]

      
      Y1_ = MM
      Y2_ = MM_xiao
      plt.plot(X,Y1_,marker = '>',markersize=4,color='orange',label='RFBD')
      plt.plot(X,Y2_,marker = '*',markersize=4,color='blue',linestyle='--',label='RB')
      
      plt.xlabel('p')
      plt.ylabel('The divergence measure')
      
      x_major_locator=MultipleLocator(1)
      y_major_locator=MultipleLocator(0.1)
      ax=plt.gca()
      ax.xaxis.set_major_locator(x_major_locator)
      ax.yaxis.set_major_locator(y_major_locator)
      plt.xlim(0.5,13.5)
      plt.ylim(-0.01,1.71)
      
      plt.legend()
      plt.show()

#===============================Example 8========================================
      qua_ = 99*len(MZ)
      final_ = []
      for i in range(9,10):
            x1 = 0.01
            MD = sorted(MZ[i])
            y = len(MD)
            for j in range(0,101):                  
                  M15 = MassFunction([({'b'}, x1), (MD, 1-x1)])
                  M15_ = {frozenset({'b'}):x1, frozenset(MD):1-x1}

                  M16 = MassFunction([({'b'}, 0.99), (MD, 0.01)])
                  M16_ = {frozenset({'b'}):0.99, frozenset(MD):0.01}

                  frame = FRAME([M15.frame(),M16.frame()],2)
                  sub_ = Subset(frame)
                  fr = len(sub_) 
                  GG = ffgg(sub_)[0]
                  FF = ffgg(sub_)[1]

                  MM1 = MTD([M9_,M10_],2)[0][1] #RFBD
                  MM1_xiao = MTD_xiao([M9_,M10_],2)[0][1] #RB
                  shuju = [x1,y,MM1,MM1_xiao]
                  final_.append(shuju)

                  if(x1 + 0.01 >= 1):
                        break
                  x1 += 0.01
      #save the data
      filename = 'data_huitu_1_10_.csv'
      with open(filename,'w',newline='') as file:
            writer = csv.writer(file)
            writer.writerows(final_)
      
#===============================Target recognition with multi-element sets========================================
      
      M17 = MassFunction([({'a','d'}, 0.70), ({'b','c'}, 0.10), ({'a','b','c','d'}, 0.20)]) 
      M17_ = {frozenset({'a','d'}):0.70, frozenset({'b','c'}):0.10, frozenset({'a','b','c','d'}):0.20}
      

      M18 = MassFunction([({'a','d'}, 0.15), ({'b','c'}, 0.65), ({'a','b','c','d'}, 0.20)]) 
      M18_ = {frozenset({'a','d'}):0.15, frozenset({'b','c'}):0.65, frozenset({'a','b','c','d'}):0.20}
      

      M19 = MassFunction([({'a','b'}, 0.70), ({'a','d'}, 0.10), ({'a','b','c','d'}, 0.20)])
      M19_ = {frozenset({'a','b'}):0.70, frozenset({'a','d'}):0.10, frozenset({'a','b','c','d'}):0.20}


      M20 = MassFunction([({'a','b'}, 0.25), ({'b','c'}, 0.25), ({'a','d'}, 0.25), ({'c','d'}, 0.25)])
      M20_ = {frozenset({'a','b'}):0.25, frozenset({'b','c'}):0.25, frozenset({'a','d'}):0.25, frozenset({'c','d'}):0.25}
      
      #======= Mass function 17 18 19 20 ===========  
      frame = FRAME([M17.frame(),M18.frame(),M19.frame(),M20.frame()],4)
      sub_ = Subset(frame)
      fr = len(sub_) 
      print('RFBD:',dsmass([M17_,M18_,M19_,M20_],4))
      print('RB:',dsmass_xiao([M17_,M18_,M19_,M20_],4))
      print('MRBD:',dsmass_chen([M17_,M18_,M19_,M20_],4))
      print('BJS:',dsmass_bjs([M17_,M18_,M19_,M20_],4))
      print('EMJSD:',dsmass_pan([M17_,M18_,M19_,M20_],4))

      
      #======= Mass function 17 18 19 =========== 
      #frame = FRAME([M17.frame(),M18.frame(),M19.frame()],3)
      #sub_ = Subset(frame)
      #fr = len(sub_) 
      #print('RFBD:',dsmass([M17_,M18_,M19_],3))
      #print('RB:',dsmass_xiao([M17_,M18_,M19_],3))
      #print('MRBD:',dsmass_chen([M17_,M18_,M19_],3))
      #print('BJS:',dsmass_bjs([M17_,M18_,M19_],3))
      #print('EMJSD:',dsmass_pan([M17_,M18_,M19_],3))
      #==========================================
      
      
#=======================================================================
      #Fig5a & Fig5b
      labels = np.array(["M(A)", "M(B)", "M(AB)", "M(AD)", "M(BC)"])
      dataLenth  = 5
      
      data_x_1 = np.array([0.3399, 0.2774, 0.0371, 0.0963, 0.0704])#1 BJS
      data_p_1 = np.array([0.2787, 0.2154, 0.0776, 0.2455, 0.1696])#2 BJS
      
      data_x_2 = np.array([0.3854, 0.2820, 0.0511, 0.1011, 0.0625])#1 EMJSD
      data_p_2 = np.array([0.3131, 0.2279, 0.1022, 0.2111, 0.1334])#2 EMJSD
      
      data_x_3 = np.array([0.4019, 0.3486, 0.1387, 0.0454, 0.0364])#1 MRBD
      data_p_3 = np.array([0.3197, 0.2118, 0.3807, 0.0503, 0.0285])#2 MRBD
      
      data_x_4 = np.array([0.4013, 0.2767, 0.0550, 0.1030, 0.0582])#1 RB
      data_p_4 = np.array([0.3293, 0.2274, 0.1140, 0.2000, 0.1172])#2 RB
      
      data_x_5 = np.array([0.4801, 0.2410, 0.0754, 0.1314, 0.0469])#1 RFBD
      data_p_5 = np.array([0.3716, 0.1352, 0.0850, 0.3207, 0.0756])#2 RFBD
      
      
      angles = np.linspace(0,2*np.pi,dataLenth,endpoint=False)
 

      data_x_1 = np.concatenate((data_x_1,[data_x_1[0]]))
      data_p_1 = np.concatenate((data_p_1,[data_p_1[0]]))
      data_x_2 = np.concatenate((data_x_2,[data_x_2[0]]))
      data_p_2 = np.concatenate((data_p_2,[data_p_2[0]]))
      data_x_3 = np.concatenate((data_x_3,[data_x_3[0]]))
      data_p_3 = np.concatenate((data_p_3,[data_p_3[0]]))
      data_x_4 = np.concatenate((data_x_4,[data_x_4[0]]))
      data_p_4 = np.concatenate((data_p_4,[data_p_4[0]]))
      data_x_5 = np.concatenate((data_x_5,[data_x_5[0]]))
      data_p_5 = np.concatenate((data_p_5,[data_p_5[0]]))
      
      
      angles = np.concatenate((angles,[angles[0]]))
      labels=np.concatenate((labels,[labels[0]])) 

      
      #--Fig5a--#
      fig = plt.figure(figsize=(7,7),facecolor="white")
      
      ax1 = plt.subplot(111,polar=True)     

      ax1.plot(angles,data_x_1,marker = 'x',markersize=4,color ='purple',label='BJS')
      ax1.fill(angles,data_x_1,facecolor='purple',alpha=0.05)  
      ax1.plot(angles,data_x_2,marker = '+',markersize=4,color ='red',linestyle = '--',label='EMJSD')
      ax1.fill(angles,data_x_2,facecolor='red',alpha=0.05)
      ax1.plot(angles,data_x_3,marker = 'o',markersize=4,color ='green',label='MRBD')
      ax1.fill(angles,data_x_3,facecolor='green',alpha=0.05)    
      ax1.plot(angles,data_x_4,marker = '*',markersize=4,color ='blue',linestyle = '--',label='RB')
      ax1.fill(angles,data_x_4,facecolor='blue',alpha=0.05)    
      ax1.plot(angles,data_x_5,marker = '>',markersize=4,color ='orange',label='RFBD')
      ax1.fill(angles,data_x_5,facecolor='orange',alpha=0.05)
      
      
      ax1.set_theta_zero_location('N')
      ax1.set_rlim(0, 0.5)
      ax1.set_rticks(np.arange(0.05, 0.50, 0.05))
      ax1.set_rlabel_position(300)
      plt.thetagrids(angles*180/np.pi,labels)          
      plt.legend(loc='best')  

      plt.show()
      
      
      #--Fig5b--#
      ax2 = plt.subplot(111,polar=True)     
      
      ax2.plot(angles,data_p_1,marker = 'x',markersize=4,color ='purple',label='BJS')
      ax2.fill(angles,data_p_1,facecolor='purple',alpha=0.05)   
      ax2.plot(angles,data_p_2,marker = '+',markersize=4,color ='red',linestyle = '--',label='EMJSD')
      ax2.fill(angles,data_p_2,facecolor='red',alpha=0.05)
      ax2.plot(angles,data_p_3,marker = 'o',markersize=4,color ='green',label='MRBD')
      ax2.fill(angles,data_p_3,facecolor='green',alpha=0.05)    

      ax2.plot(angles,data_p_4,marker = '*',markersize=4,color ='blue',linestyle = '--',label='RB')
      ax2.fill(angles,data_p_4,facecolor='blue',alpha=0.05) 
      ax2.plot(angles,data_p_5,marker = '>',markersize=4,color ='orange',label='RFBD')
      ax2.fill(angles,data_p_5,facecolor='orange',alpha=0.05)
      

      ax2.set_theta_zero_location('N')

      ax2.set_rlim(0, 0.5)
      ax2.set_rticks(np.arange(0.05, 0.50, 0.05))

      ax2.set_rlabel_position(300)
      plt.thetagrids(angles*180/np.pi,labels)          

      plt.legend(loc='best')  
      plt.show()
      
