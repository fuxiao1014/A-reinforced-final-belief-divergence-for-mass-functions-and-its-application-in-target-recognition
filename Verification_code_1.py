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
      #print('RFBD:',d_)
      return d_


def MTD(M,k):  #Calculate the matrix of RFBD(m1,m2)
      D_ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D_[p].append(D(M[p],M[q]))
      #print('the matrix of RFBD:\n',np.matrix(D_))
      return D_


#==========================================================================================

#=================================Method of Fuyuan Xiao(2020)====================================

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

#=================================Method of Fuyuan Xiao(2019)-BJS divergence====================================

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

#=============================Method of Gao and Xiao====================================

def Div_gao(m1,m2):
      IBX = 0
      div_ = 0
      global sub_
      global fr

      for i in range(0,fr):
            len_sub = pow(2,len(frozenset(sub_[i])))-1
            if(frozenset(sub_[i]) in m1 and frozenset(sub_[i]) in m2):
                  if(m1[frozenset(sub_[i])] != 0 or m2[frozenset(sub_[i])] != 0):
                        H1 = m1[frozenset(sub_[i])] / len_sub
                        H2 = m2[frozenset(sub_[i])] / len_sub
                        IBX += pow(H1-H2,2)/(H1+H2)
            else:
                  IBX += 0

      div_ = IBX/2
      #print('Gao and Xiao:',div_)
      return div_


def MTD_gao(M,k):
      D__ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D__[p].append(Div_gao(M[p],M[q]))
      #print('the matrix of Gao and Xiao:\n',np.matrix(D__))
      return D__

#==========================================================================================
#=============================Method of Zeng(fractal)====================================

def Div_fractal(m1,m2):
      H = 0
      div_ = 0
      global sub_
      global fr

      m1_H = {}
      m2_H = {}
      for i in range(0,fr):
            shuzu = []
            len_sub = pow(2,len(frozenset(sub_[i])))-1
            for j in frozenset(sub_[i]):
                  shuzu.append(j)
            sub_in = Subset(shuzu)
            for j in range(0,len(sub_in)):
                  if(frozenset(sub_[i]) in m1):
                        if(frozenset(sub_in[j]) not in m1_H):
                              m1_H[frozenset(sub_in[j])] = m1[frozenset(sub_[i])]/len_sub
                              
                        else:
                              m1_H[frozenset(sub_in[j])] += m1[frozenset(sub_[i])]/len_sub
                  
                  
                  if(frozenset(sub_[i]) in m2):
                        if(frozenset(sub_in[j]) not in m2_H):
                              m2_H[frozenset(sub_in[j])] = m2[frozenset(sub_[i])]/len_sub
                        else:
                              m2_H[frozenset(sub_in[j])] += m2[frozenset(sub_[i])]/len_sub

      for i in range(0,fr):
            if(frozenset(sub_[i]) not in m1_H or m1_H[frozenset(sub_[i])] == 0):
                  m1_H[frozenset(sub_[i])] = pow(10,-12)
            if(frozenset(sub_[i]) not in m2_H or m2_H[frozenset(sub_[i])] == 0):
                  m2_H[frozenset(sub_[i])] = pow(10,-12)

            h1 = m1_H[frozenset(sub_[i])] / m2_H[frozenset(sub_[i])]
            H1 = m1_H[frozenset(sub_[i])] * math.log(pow(h1,1/2),2)
            h2 = m2_H[frozenset(sub_[i])]/m1_H[frozenset(sub_[i])]
            H2 = m2_H[frozenset(sub_[i])] * math.log(pow(h2,1/2),2)
            H += H1+H2

      div_ = H/2
      #print('Zeng:',div_)
      return div_


def MTD_fractal(M,k):
      D__ = [[] for j in range(0,k)] #k×k
      for p in range(0,k):
            for q in range(0,k):
                  D__[p].append(Div_fractal(M[p],M[q]))
      #print('the matrix of Zeng:\n',np.matrix(D__))
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
            #IQ
            IQP.append(iqp)
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
#======================================Gao and Xiao===(the same as BJS)=========================================

#Final weight
def fweight_gao(M,k):
      sumWeight = 0
      FWEIGHT = []
      EX = Weightex_xiao(MTD_gao(M,k),k) 
      IN = Weightin_bjs(M,k) 
      for p in range(0,k):
            sumWeight += EX[p] * IN[p]
      for p in range(0,k):
            fweightp = EX[p] * IN[p] / sumWeight
            FWEIGHT.append(fweightp)
      #print("Final:",FWEIGHT)
      return FWEIGHT



def remass_gao(M,k):
      ww = fweight_gao(M,k)
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


def dsmass_gao(M,k):  
      mm1 = remass_gao(M,k) 
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
#==================================Zeng's fractal-based method=============================

#External weight
def Weightex_zeng(M,k):
      max_ = M[0][0]
      min_ = M[0][1]
      for i in range(0,k):
            for j in range(0,k):
                  if(M[i][j]>max_):
                        max_ = M[i][j]
                  if(M[i][j]<min_ and M[i][j] != 0):
                        min_ = M[i][j]

      S = [[] for i in range(0,k)]
      for i in range(0,k):
            for j in range(0,k):
                  if(i==j):
                        S[i].append(1)
                  else:
                        Sij = (max_ - M[i][j])/(max_ - min_)
                        S[i].append(Sij)

      Sump = [] 
      for p in range(0,k):             
            SDp = 0
            for q in range(0,k):
                  SDp += math.exp(S[p][q])
            Sump.append(SDp)

      CREDP = [] 
      Sumpp = 0
      for p in range(0,k):
            Sumpp += Sump[p]
      for p in range(0,k):
            Credp = Sump[p] / Sumpp
            CREDP.append(Credp)

      return CREDP

#The weighted mass function
def remass_zeng(M,k): 
      ww = Weightex_zeng(MTD_fractal(M,k),k)
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


def dsmass_zeng(M,k):  
      mm1 = remass_zeng(M,k) 
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
      test_per = round(1-train_,3)
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
      test_per = round(1-train_,3)
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
      
      #F1 Score
      FN_se = 0 
      FN_ve = 0 
      FN_vi = 0
      FP_se = 0 
      FP_ve = 0 
      FP_vi = 0
      
      cor_per = 0 #Accuracy
      cor_se_per = 0
      cor_ve_per = 0
      cor_vi_per = 0
      COR = []
      FS = []
      
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
            elif(method == "Gao"):
                  dict_fe = dsmass_gao([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
            elif(method == "Zeng"):
                  dict_fe = dsmass_zeng([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3]],4)
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
                  
            #Calculation of Accuracy
            '''
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
            '''
            #Calculation of F1 Score
            if(target == 0):
                  if(target == mubiao[c]): 
                        cor += 1
                        cor_se += 1 #0-0
                  else:
                        FP_se += 1 #1-0 & 2-0
            elif(target == 1):
                  if(target == mubiao[c]):
                        cor += 1
                        cor_ve += 1 #1-1
                  else:
                        FP_ve += 1 #0-1 & 2-1
            elif(target == 2):
                  if(target == mubiao[c]):
                        cor += 1
                        cor_vi += 1 #2-2
                  else:
                        FP_vi += 1 #0-2 & 1-2
            else:
                  print('error')

            if(mubiao[c] == 0):
                  if(target != 0): 
                        FN_se += 1 #0-1 & 0-2
            elif(mubiao[c] == 1):
                  if(target != 1):
                        FN_ve += 1 #1-0 & 1-2
            elif(mubiao[c] == 2):
                  if(target != 2):
                        FN_vi += 1 #2-0 & 2-1
            
            if(c == test_qua-1):
                  P_0 = cor_se / (cor_se + FP_se)
                  R_0 = cor_se / (cor_se + FN_se)
                  FS_0 = 2 * P_0 * R_0 / (P_0 + R_0)
                  FS.append(FS_0)
                  
                  P_1 = cor_ve / (cor_ve + FP_ve)
                  R_1 = cor_ve / (cor_ve + FN_ve)
                  FS_1 = 2 * P_1 * R_1 / (P_1 + R_1)
                  FS.append(FS_1)
                  
                  P_2 = cor_vi / (cor_vi + FP_vi)
                  R_2 = cor_vi / (cor_vi + FN_vi)
                  FS_2 = 2 * P_2 * R_2 / (P_2 + R_2)
                  FS.append(FS_2)
            
      #return COR
      return FS
      
#mean average
def tomean_(corm):
      sumcor = 0
      for i in range(0,len(corm)):
            sumcor += corm[i]
      avecor = sumcor / len(corm)
      return avecor


#Wheat seeds dataset
def xiaomai_in(X,y,train_): #Wheat Seeds Dataset(get the new mass function)
      #7 attributes
      test_per = round(1-train_,3)
      X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_per,
                                                 stratify=y,  
                                                 shuffle=True, 
                                                 random_state=1)   


      test_qua = int(210*test_per)


      #---training set---#
      w1 = X_train[df['class'] == 0]

      w2 = X_train[df['class'] == 1]
      w3 = X_train[df['class'] == 2]


      #-----------class=0-----------------
      x1 = w1[['1']]
      x11=x1.values.tolist()

      x2 = w1[['2']]
      x21=x2.values.tolist()

      x3 = w1[['3']]
      x31=x3.values.tolist()

      x4 = w1[['4']]
      x41=x4.values.tolist()

      x5 = w1[['5']]
      x51=x5.values.tolist()

      x6 = w1[['6']]
      x61=x6.values.tolist()

      x7 = w1[['7']]
      x71=x7.values.tolist()
      
      

      #-----------class=1------------------
      y1 = w2[['1']]
      y11=y1.values.tolist()

      y2 = w2[['2']]
      y21=y2.values.tolist()

      y3 = w2[['3']]
      y31=y3.values.tolist()

      y4 = w2[['4']]
      y41=y4.values.tolist()

      y5 = w2[['5']]
      y51=y5.values.tolist()

      y6 = w2[['6']]
      y61=y6.values.tolist()

      y7 = w2[['7']]
      y71=y7.values.tolist()
      

      #-----------class=2----------------
      z1 = w3[['1']]
      z11=z1.values.tolist()

      z2 = w3[['2']]
      z21=z2.values.tolist()

      z3 = w3[['3']]
      z31=z3.values.tolist()

      z4 = w3[['4']]
      z41=z4.values.tolist()

      z5 = w3[['5']]
      z51=z5.values.tolist()

      z6 = w3[['6']]
      z61=z6.values.tolist()

      z7 = w3[['7']]
      z71=z7.values.tolist()
      

      #interval number [min，max]
      A3 = [[] for i in range(0,7)] #max-list
      A1 = [[] for i in range(0,7)] #min-list
      #3 class
      dada = [[x11,y11,z11],[x21,y21,z21],[x31,y31,z31],[x41,y41,z41],[x51,y51,z51],[x61,y61,z61],[x71,y71,z71]]

      for i in range(0,7):
            for j in range(0,3):
                  maxx = max(dada[i][j])[0]
                  A3[i].append(maxx)
                  minx = min(dada[i][j])[0]
                  A1[i].append(minx)

      FUZ = [] #Triangular fuzzy number
      for i in range(0,7):
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
                  print("sabc error")
      
      
            fuz = [sa,sb,sc,sab,sac,sbc,sabc]
            FUZ.append(fuz) 
      
            
      #--- test set ---#
      
      Test = [] #test set list
      for i in range(0,test_qua):
            test_a = X_test.iloc[i:i+1].values.tolist()
            test_t = y_test.iloc[i:i+1].values.tolist()
            test_a.append(test_t)
            Test.append(test_a)
      
      P1 = []
      P2 = []
      P3 = []
      P4 = []
      P5 = []
      P6 = []
      P7 = []
      
      MB = [] #target set

      for t in Test:
            P1.append(t[0][0])
            P2.append(t[0][1])
            P3.append(t[0][2])
            P4.append(t[0][3])
            P5.append(t[0][4])
            P6.append(t[0][5])
            P7.append(t[0][6])
            
            MB.append(t[1][0])
      SJ = [P1,P2,P3,P4,P5,P6,P7] #[[P],[Q],[R],[T]]
      BPAm = [[{} for i in range(0,7)] for j in range(0,test_qua)] #BPA
      
      
      for c in range(0,test_qua):
            #construct BPA for each data of test set
            target = 0
            alpha = 5
            for i in range(0,7): #for each attribute
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

def methodff_xiaomai(train_,BPAm,MB,method): #Calculate the accuracy of the Wheat Seeds Dataset
      global sub_
      global fr
      global GG
      global FF
      global mubiao

      test_per = round(1-train_,3)
      test_qua = int(210*test_per)

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
      cor_se = 0 
      cor_ve = 0 
      cor_vi = 0
      #F1 Score
      FN_se = 0 
      FN_ve = 0 
      FN_vi = 0
      FP_se = 0 
      FP_ve = 0 
      FP_vi = 0
      
      cor_per = 0 #Accuracy
      cor_se_per = 0 #Accuracy(Setosa)
      cor_ve_per = 0 #Accuracy(Versicolor)
      cor_vi_per = 0 #Accuracy(Virginica)
      COR = []

      FS = []
      Total = []
      
      for c in range(0,test_qua):
            
            if(method == "proposed"):
                  dict_fe = dsmass([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            elif(method == "xiao"):
                  dict_fe = dsmass_xiao([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            elif(method == "pan"):
                  dict_fe = dsmass_pan([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            elif(method == "chen"):
                  dict_fe = dsmass_chen([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            elif(method == "bjs"):
                  dict_fe = dsmass_bjs([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            elif(method == "Gao"):
                  dict_fe = dsmass_gao([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            elif(method == "Zeng"):
                  dict_fe = dsmass_zeng([BPAm[c][0],BPAm[c][1],BPAm[c][2],BPAm[c][3],BPAm[c][4],BPAm[c][5],BPAm[c][6]],7)
            else:
                  print("error")

            #Obtain the focal element of maximum belief(focal element，value)
            max_fe = max(zip(dict_fe.values(),dict_fe.keys()))
      
      
            #Whether the target is correctly identified
            if(max_fe[1] == frozenset({'a'})):
                  target = 0
            elif(max_fe[1] == frozenset({'b'})):
                  target = 1
            elif(max_fe[1] == frozenset({'c'})):
                  target = 2
            else:
                  target = 3
                  
            #target
            if(target == 0):
                  if(target == mubiao[c]): 
                        cor += 1
                        cor_se += 1 #0-0
                  else:
                        FP_se += 1 #1-0 & 2-0
            elif(target == 1):
                  if(target == mubiao[c]):
                        cor += 1
                        cor_ve += 1 #1-1
                  else:
                        FP_ve += 1 #0-1 & 2-1
            elif(target == 2):
                  if(target == mubiao[c]):
                        cor += 1
                        cor_vi += 1 #2-2
                  else:
                        FP_vi += 1 #0-2 & 1-2
            else:
                  print('error')

            if(mubiao[c] == 0):
                  if(target != 0): 
                        FN_se += 1 #0-1 & 0-2
            elif(mubiao[c] == 1):
                  if(target != 1):
                        FN_ve += 1 #1-0 & 1-2
            elif(mubiao[c] == 2):
                  if(target != 2):
                        FN_vi += 1 #2-0 & 2-1
            
            
            #Accuracy
            
            if(c == test_qua-1):
                  
                  cor_per = cor / test_qua
                  cor_se_per = cor_se / test_se
                  cor_ve_per = cor_ve / test_ve
                  cor_vi_per = cor_vi / test_vi
                  COR.append(cor_per)
                  COR.append(cor_se_per)
                  COR.append(cor_ve_per)
                  COR.append(cor_vi_per)
            '''
            #F1-Score
                  P_0 = cor_se / (cor_se + FP_se)
                  R_0 = cor_se / (cor_se + FN_se)
                  FS_0 = 2 * P_0 * R_0 / (P_0 + R_0)
                  FS.append(FS_0)
                  
                  P_1 = cor_ve / (cor_ve + FP_ve)
                  R_1 = cor_ve / (cor_ve + FN_ve)
                  FS_1 = 2 * P_1 * R_1 / (P_1 + R_1)
                  FS.append(FS_1)
                  
                  P_2 = cor_vi / (cor_vi + FP_vi)
                  R_2 = cor_vi / (cor_vi + FN_vi)
                  FS_2 = 2 * P_2 * R_2 / (P_2 + R_2)
                  FS.append(FS_2)

                  #Total = [COR,FS]
            '''

      return COR
      #return FS

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
      train__ = [0.80,0.775,0.75,0.725,0.70,0.675,0.65,0.625,0.60,0.575,0.55,0.525,0.50,0.475,0.45,0.425,0.40,0.375,0.35,0.325,0.30,0.275,0.25,0.225,0.20]

      
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
      COR_gao = [] 
      COR_gao_se = []
      COR_gao_ve = []
      COR_gao_vi = []
      COR_zeng = [] 
      COR_zeng_se = []
      COR_zeng_ve = []
      COR_zeng_vi = []
      FS_proposed_se = []
      FS_proposed_ve = []
      FS_proposed_vi = []
      FS_xiao_se = []
      FS_xiao_ve = []
      FS_xiao_vi = []
      FS_pan_se = []
      FS_pan_ve = []
      FS_pan_vi = []
      FS_chen_se = []
      FS_chen_ve = []
      FS_chen_vi = []
      FS_bjs_se = []
      FS_bjs_ve = []
      FS_bjs_vi = []
      FS_gao_se = []
      FS_gao_ve = []
      FS_gao_vi = []
      FS_zeng_se = []
      FS_zeng_ve = []
      FS_zeng_vi = []

      
      for i in range(0,len(train__)):
            
            lili = yuanweihua_in(X,y,train__[i]) 
            BPAm_ = lili[0] 
            mubiao = lili[1]

            jingdu_p = methodff(train__[i],BPAm_,mubiao,"proposed")
            #COR_proposed.append(jingdu_p[0])
            #COR_proposed_se.append(jingdu_p[1])
            #COR_proposed_ve.append(jingdu_p[2])
            #COR_proposed_vi.append(jingdu_p[3])
            FS_proposed_se.append(jingdu_p[0])
            FS_proposed_ve.append(jingdu_p[1])
            FS_proposed_vi.append(jingdu_p[2])
            #print("RB:")
            jingdu_x = methodff(train__[i],BPAm_,mubiao,"xiao")
            #COR_xiao.append(jingdu_x[0])
            #COR_xiao_se.append(jingdu_x[1])
            #COR_xiao_ve.append(jingdu_x[2])
            #COR_xiao_vi.append(jingdu_x[3])
            FS_xiao_se.append(jingdu_x[0])
            FS_xiao_ve.append(jingdu_x[1])
            FS_xiao_vi.append(jingdu_x[2])
            #print("EMJSD:")
            jingdu_s = methodff(train__[i],BPAm_,mubiao,"pan")
            #COR_pan.append(jingdu_s[0])
            #COR_pan_se.append(jingdu_s[1])
            #COR_pan_ve.append(jingdu_s[2])
            #COR_pan_vi.append(jingdu_s[3])
            FS_pan_se.append(jingdu_s[0])
            FS_pan_ve.append(jingdu_s[1])
            FS_pan_vi.append(jingdu_s[2])
            #print("MRBD:")
            jingdu_c = methodff(train__[i],BPAm_,mubiao,"chen")
            #COR_chen.append(jingdu_c[0])
            #COR_chen_se.append(jingdu_c[1])
            #COR_chen_ve.append(jingdu_c[2])
            #COR_chen_vi.append(jingdu_c[3])
            FS_chen_se.append(jingdu_c[0])
            FS_chen_ve.append(jingdu_c[1])
            FS_chen_vi.append(jingdu_c[2])
            #print("BJS:")
            jingdu_ds = methodff(train__[i],BPAm_,mubiao,"bjs")
            #COR_bjs.append(jingdu_ds[0])
            #COR_bjs_se.append(jingdu_ds[1])
            #COR_bjs_ve.append(jingdu_ds[2])
            #COR_bjs_vi.append(jingdu_ds[3])
            FS_bjs_se.append(jingdu_ds[0])
            FS_bjs_ve.append(jingdu_ds[1])
            FS_bjs_vi.append(jingdu_ds[2])
            #print("Gao:")
            jingdu_gao = methodff(train__[i],BPAm_,mubiao,"Gao")
            #COR_gao.append(jingdu_gao[0])
            #COR_gao_se.append(jingdu_gao[1])
            #COR_gao_ve.append(jingdu_gao[2])
            #COR_gao_vi.append(jingdu_gao[3])
            FS_gao_se.append(jingdu_gao[0])
            FS_gao_ve.append(jingdu_gao[1])
            FS_gao_vi.append(jingdu_gao[2])
            #print("Zeng:")
            jingdu_zeng = methodff(train__[i],BPAm_,mubiao,"Zeng")
            #COR_zeng.append(jingdu_zeng[0])
            #COR_zeng_se.append(jingdu_zeng[1])
            #COR_zeng_ve.append(jingdu_zeng[2])
            #COR_zeng_vi.append(jingdu_zeng[3])
            FS_zeng_se.append(jingdu_zeng[0])
            FS_zeng_ve.append(jingdu_zeng[1])
            FS_zeng_vi.append(jingdu_zeng[2])
            
      #If you want to calculate another metric, you can modify the output array
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
      print("gao-se:",tomean_(COR_gao_se))
      print("gao-ve:",tomean_(COR_gao_ve))
      print("gao-vi:",tomean_(COR_gao_vi))
      print("zeng-se:",tomean_(COR_zeng_se))
      print("zeng-ve:",tomean_(COR_zeng_ve))
      print("zeng-vi:",tomean_(COR_zeng_vi))
      
      #---------------------------------------
      print("proposed:",COR_proposed)
      print("xiao:",COR_xiao)
      print("pan:",COR_pan)
      print("chen:",COR_chen)
      print("bjs:",COR_bjs)
      print("gao:",COR_gao)
      print("zeng:",COR_zeng)
      #---------------------------------------
      
      
      
      #Fig8

      #Results:

      COR_proposed = [0.9, 1.0, 0.9166666666666666, 1.0, 0.9333333333333333, 0.9375, 0.9411764705882353, 1.0, 1.0, 0.9523809523809523, 0.9090909090909091, 0.9565217391304348, 0.96, 0.9230769230769231, 0.9629629629629629, 0.9642857142857143, 0.9666666666666667, 0.9354838709677419, 0.9375, 1.0, 0.9714285714285714, 0.9722222222222222, 0.8918918918918919, 0.9736842105263158, 0.975]
      COR_xiao = [1.0, 1.0, 0.9166666666666666, 1.0, 0.9333333333333333, 1.0, 0.9411764705882353, 1.0, 1.0, 0.8571428571428571, 0.9090909090909091, 0.9565217391304348, 0.96, 0.9230769230769231, 0.9259259259259259, 0.8928571428571429, 0.9333333333333333, 0.8709677419354839, 0.875, 0.9393939393939394, 0.8857142857142857, 0.9166666666666666, 0.8648648648648649, 0.9210526315789473, 0.925]
      COR_pan = [1.0, 1.0, 0.9166666666666666, 1.0, 0.9333333333333333, 1.0, 0.9411764705882353, 1.0, 1.0, 0.8571428571428571, 0.9090909090909091, 0.9565217391304348, 0.92, 0.9230769230769231, 0.9259259259259259, 0.8928571428571429, 0.9333333333333333, 0.8709677419354839, 0.875, 0.9393939393939394, 0.8857142857142857, 0.9166666666666666, 0.8648648648648649, 0.9210526315789473, 0.925]
      COR_chen = [1.0, 1.0, 0.9166666666666666, 0.9230769230769231, 0.9333333333333333, 1.0, 0.9411764705882353, 0.9444444444444444, 1.0, 0.8571428571428571, 0.8636363636363636, 0.9130434782608695, 0.92, 0.8846153846153846, 0.9259259259259259, 0.8571428571428571, 0.9, 0.8709677419354839, 0.875, 0.9393939393939394, 0.8571428571428571, 0.9166666666666666, 0.8918918918918919, 0.868421052631579, 0.9]
      COR_bjs = [0.9, 0.9090909090909091, 0.9166666666666666, 0.8461538461538461, 0.9333333333333333, 0.9375, 0.8235294117647058, 0.8888888888888888, 0.85, 0.8571428571428571, 0.9090909090909091, 0.8695652173913043, 0.88, 0.9230769230769231, 0.8888888888888888, 0.8571428571428571, 0.9, 0.8387096774193549, 0.8125, 0.9090909090909091, 0.8285714285714286, 0.8888888888888888, 0.8648648648648649, 0.9210526315789473, 0.925]
      COR_gao = [0.9, 0.8181818181818182, 0.8333333333333334, 0.8461538461538461, 0.9333333333333333, 0.9375, 0.8235294117647058, 0.8888888888888888, 0.85, 0.8095238095238095, 0.9090909090909091, 0.8260869565217391, 0.88, 0.9230769230769231, 0.8888888888888888, 0.8571428571428571, 0.8333333333333334, 0.8387096774193549, 0.8125, 0.9090909090909091, 0.8, 0.9166666666666666, 0.918918918918919, 0.8947368421052632, 0.925]
      COR_zeng = [1.0, 1.0, 0.9166666666666666, 1.0, 0.9333333333333333, 1.0, 0.9411764705882353, 1.0, 1.0, 0.8571428571428571, 0.9090909090909091, 0.9565217391304348, 0.92, 0.9230769230769231, 0.9259259259259259, 0.8928571428571429, 0.9333333333333333, 0.8709677419354839, 0.875, 0.9393939393939394, 0.8857142857142857, 0.9166666666666666, 0.8648648648648649, 0.9210526315789473, 0.925]
      
      plt.figure(figsize=(12, 6))
      x = np.linspace(0.2,0.8,len(train__))
      
      y1=COR_proposed #RFBD
      y2=COR_xiao #RB
      y3=COR_pan #EMJSD
      y4=COR_chen #MRBD
      y5=COR_bjs #BJS
      y6=COR_gao #Gao
      y7=COR_zeng #Zeng

      x_major_locator=MultipleLocator(0.1)
      y_major_locator=MultipleLocator(0.1)
      ax=plt.gca()
      ax.xaxis.set_major_locator(x_major_locator)
      ax.yaxis.set_major_locator(y_major_locator)
      
      plt.plot(x,y5,marker = '*',color='purple',label='BJS',alpha=0.5)
      plt.plot(x,y3,marker = '+',color='red',label='EMJSD',alpha=0.5)
      plt.plot(x,y4,marker = 'o',color='green',label='MRBD',alpha=0.5)
      plt.plot(x,y6,marker = 'd',color='hotpink',label='IBχ$^2$',linestyle = '-.',alpha=0.5)
      plt.plot(x,y7,marker = 'h',color='grey',label='FBD$_{SKL}$',linestyle = '-.',alpha=0.5)
      plt.plot(x,y2,marker = 'x',color='blue',label='RB',linestyle = '--',alpha=0.5)
      plt.plot(x,y1,marker = '>',color='orange',label='RFBD',linestyle = '--')
      plt.xlabel('Percentage of testing data set')
      plt.ylabel('Recognition accuracy')
      plt.xlim(0.17,0.83)
      plt.ylim(0.35,1.05)
      ticks = ax.set_xticks(x)
      labels = ax.set_xticklabels(['20%','','25%','','30%','','35%','','40%','','45%','','50%','','55%','','60%','','65%','','70%','','75%','','80%'],rotation = 0,fontsize = 'small') 
      plt.legend()
      plt.show()

      #---------------------------------------------
      #Accuracy-F1 Score (Fig9)
      
      #iris-acc iris-fs wheat-acc wheat-fs
      Y_bjs = [0.8894,0.8766,0.8804,0.8759]
      Y_pan = [0.9400,0.9302,0.8699,0.8709]
      Y_chen = [0.9195,0.9131,0.8430,0.8434]
      Y_xiao = [0.9412,0.9315,0.8821,0.8825]
      Y_gao = [0.8791,0.8634,0.8830,0.8785]
      Y_zeng = [0.9400,0.9302,0.8764,0.8770]
      Y_proposed = [0.9598,0.9528,0.8968,0.8955]
      
      plt.figure(figsize=(10, 6))
      
      x = np.arange(4)
      plt.xticks([0,1,2,3],['Iris-Accuracy','Iris-F1 Score','WS-Accuracy','WS-F1 Score'])
      plt.yticks([0.8,0.85,0.9,0.95,1],[80,85,90,95,100])
      for i in range(0,4):
            plt.text(i+0.2,Y_proposed[i]+0.002,str(Y_proposed[i]),fontsize=8.5,color='orange')
      plt.xlabel('Metric')
      plt.ylabel('Result(%)')

      plt.bar(x-0.3, Y_bjs, width=0.1, color='purple', label='BJS',alpha=0.5,ec='black',lw=0.5)
      plt.bar(x-0.2, Y_pan, width=0.1, color='red', label='EMJSD',alpha=0.5,ec='black',lw=0.5)
      plt.bar(x-0.1, Y_chen, width=0.1, color='green', label='MRBD',alpha=0.5,ec='black',lw=0.5)
      plt.bar(x, Y_xiao, width=0.1, color='blue', label='RB',alpha=0.5,ec='black',lw=0.5)
      plt.bar(x+0.1, Y_gao, width=0.1, color='hotpink', label='IBχ$^2$',alpha=0.5,ec='black',lw=0.5)
      plt.bar(x+0.2, Y_zeng, width=0.1, color='grey', label='FBD$_{SKL}$',alpha=0.5,ec='black',lw=0.5)
      plt.bar(x+0.3, Y_proposed, width=0.1, color='orange', label='RFBD',alpha=0.5,ec='black',lw=0.5)
      plt.xlim(-0.49,3.49)
      plt.ylim(0.8,1.0)
      plt.legend()
      plt.show()
      
      
      
#=================================Experiment of Wheat seeds  dataset========================================
      cols_name = ['1','2','3','4','5','6','7','class']
      df = pd.read_csv('ws.csv',names=cols_name)
      df['class'] = pd.Categorical(df['class'])
      df['class'] = df['class'].cat.codes 

      
      frame = ['a', 'b', 'c']
      sub_ = Subset(frame)
      fr = len(sub_)
      GG = ffgg(sub_)[0]
      FF = ffgg(sub_)[1]
      

      #training set and test set
      X = df[['1','2','3','4','5','6','7']]
      y = df['class']
      #percentage
      train__ = [0.80,0.775,0.75,0.725,0.70,0.675,0.65,0.625,0.60,0.575,0.55,0.525,0.50,0.475,0.45,0.425,0.40,0.375,0.35,0.325,0.30,0.275,0.25,0.225,0.20]

      
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
      COR_gao = [] 
      COR_gao_se = []
      COR_gao_ve = []
      COR_gao_vi = []
      COR_zeng = [] 
      COR_zeng_se = []
      COR_zeng_ve = []
      COR_zeng_vi = []
      FS_proposed_se = []
      FS_proposed_ve = []
      FS_proposed_vi = []
      FS_xiao_se = []
      FS_xiao_ve = []
      FS_xiao_vi = []
      FS_pan_se = []
      FS_pan_ve = []
      FS_pan_vi = []
      FS_chen_se = []
      FS_chen_ve = []
      FS_chen_vi = []
      FS_bjs_se = []
      FS_bjs_ve = []
      FS_bjs_vi = []
      FS_gao_se = []
      FS_gao_ve = []
      FS_gao_vi = []
      FS_zeng_se = []
      FS_zeng_ve = []
      FS_zeng_vi = []

      
      for i in range(0,len(train__)):
            
            lili = xiaomai_in(X,y,train__[i]) 
            BPAm_ = lili[0] 
            mubiao = lili[1]

            jingdu_p = methodff_xiaomai(train__[i],BPAm_,mubiao,"proposed")
            COR_proposed.append(jingdu_p[0])
            COR_proposed_se.append(jingdu_p[1])
            COR_proposed_ve.append(jingdu_p[2])
            COR_proposed_vi.append(jingdu_p[3])
            #FS_proposed_se.append(jingdu_p[0])
            #FS_proposed_ve.append(jingdu_p[1])
            #FS_proposed_vi.append(jingdu_p[2])
            #print("RB:")
            jingdu_x = methodff_xiaomai(train__[i],BPAm_,mubiao,"xiao")
            COR_xiao.append(jingdu_x[0])
            COR_xiao_se.append(jingdu_x[1])
            COR_xiao_ve.append(jingdu_x[2])
            COR_xiao_vi.append(jingdu_x[3])
            #FS_xiao_se.append(jingdu_x[0])
            #FS_xiao_ve.append(jingdu_x[1])
            #FS_xiao_vi.append(jingdu_x[2])
            #print("EMJSD:")
            jingdu_s = methodff_xiaomai(train__[i],BPAm_,mubiao,"pan")
            COR_pan.append(jingdu_s[0])
            COR_pan_se.append(jingdu_s[1])
            COR_pan_ve.append(jingdu_s[2])
            COR_pan_vi.append(jingdu_s[3])
            #FS_pan_se.append(jingdu_s[0])
            #FS_pan_ve.append(jingdu_s[1])
            #FS_pan_vi.append(jingdu_s[2])
            #print("MRBD:")
            jingdu_c = methodff_xiaomai(train__[i],BPAm_,mubiao,"chen")
            COR_chen.append(jingdu_c[0])
            COR_chen_se.append(jingdu_c[1])
            COR_chen_ve.append(jingdu_c[2])
            COR_chen_vi.append(jingdu_c[3])
            #FS_chen_se.append(jingdu_c[0])
            #FS_chen_ve.append(jingdu_c[1])
            #FS_chen_vi.append(jingdu_c[2])
            #print("BJS:")
            jingdu_ds = methodff_xiaomai(train__[i],BPAm_,mubiao,"bjs")
            COR_bjs.append(jingdu_ds[0])
            COR_bjs_se.append(jingdu_ds[1])
            COR_bjs_ve.append(jingdu_ds[2])
            COR_bjs_vi.append(jingdu_ds[3])
            #FS_bjs_se.append(jingdu_ds[0])
            #FS_bjs_ve.append(jingdu_ds[1])
            #FS_bjs_vi.append(jingdu_ds[2])
            #print("Gao:")
            jingdu_gao = methodff_xiaomai(train__[i],BPAm_,mubiao,"Gao")
            COR_gao.append(jingdu_gao[0])
            COR_gao_se.append(jingdu_gao[1])
            COR_gao_ve.append(jingdu_gao[2])
            COR_gao_vi.append(jingdu_gao[3])
            #FS_gao_se.append(jingdu_gao[0])
            #FS_gao_ve.append(jingdu_gao[1])
            #FS_gao_vi.append(jingdu_gao[2])
            #print("Zeng:")
            jingdu_zeng = methodff_xiaomai(train__[i],BPAm_,mubiao,"Zeng")
            COR_zeng.append(jingdu_zeng[0])
            COR_zeng_se.append(jingdu_zeng[1])
            COR_zeng_ve.append(jingdu_zeng[2])
            COR_zeng_vi.append(jingdu_zeng[3])
            #FS_zeng_se.append(jingdu_zeng[0])
            #FS_zeng_ve.append(jingdu_zeng[1])
            #FS_zeng_vi.append(jingdu_zeng[2])

      #If you want to calculate another metric, you can modify the output array
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
      print("gao-se:",tomean_(COR_gao_se))
      print("gao-ve:",tomean_(COR_gao_ve))
      print("gao-vi:",tomean_(COR_gao_vi))
      print("zeng-se:",tomean_(COR_zeng_se))
      print("zeng-ve:",tomean_(COR_zeng_ve))
      print("zeng-vi:",tomean_(COR_zeng_vi))
      #---------------------------------------
      #print("proposed:",COR_proposed)
      #print("xiao:",COR_xiao)
      #print("pan:",COR_pan)
      #print("chen:",COR_chen)
      #print("bjs:",COR_bjs)
      #print("gao:",COR_gao)
      #print("zeng:",COR_zeng)
      #---------------------------------------
      
      
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
      #frame = FRAME([M1.frame(),M2.frame()],2)
      #frame = FRAME([M3.frame(),M4.frame()],2)
      #frame = FRAME([M5.frame(),M6.frame()],2)
      #frame = FRAME([M7.frame(),M8.frame()],2)
      frame = FRAME([M9.frame(),M10.frame()],2)
      sub_ = Subset(frame)
      fr = len(sub_)

      #print(MTD([M1_,M2_],2)) #RFBD
      #print(MTD_chen([M1_,M2_],2)) #MRBD
      #print(MTD_xiao([M1_,M2_],2)) #RB
      #print(MTD_bjs([M1_,M2_],2)) #BJS
      print(MTD_pan([M9_,M10_],2)) #EMJSD
      print(MTD_gao([M9_,M10_],2)) #EMJSD
      print(MTD_fractal([M9_,M10_],2))


      
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
      MM_gao = []
      MM_zeng = []
      for i in range(0,len(MZ)):
            MD = sorted(MZ[i])
            M11 = MassFunction([({'b'}, 0.05), (MD, 0.95)]) 
            M11_ = {frozenset({'b'}):0.05, frozenset(MD):0.95}

            M12 = MassFunction([({'b'}, 0.95), (MD, 0.05)]) 
            M12_ = {frozenset({'b'}):0.95, frozenset(MD):0.05}
            frame = FRAME([M11.frame(),M12.frame()],2)
            sub_ = Subset(frame)
            fr = len(sub_)

            MM.append(MTD([M11_,M12_],2)[0][1])
            MM_xiao.append(MTD_xiao([M11_,M12_],2)[0][1])
            MM_chen.append(MTD_chen([M11_,M12_],2)[0][1])
            MM_bjs.append(MTD_bjs([M11_,M12_],2)[0][1])
            MM_pan.append(MTD_pan([M11_,M12_],2)[0][1])
            MM_gao.append(MTD_gao([M11_,M12_],2)[0][1])
            MM_zeng.append(MTD_fractal([M11_,M12_],2)[0][1])
      print("MM_chen:",MM_chen)
      print("MM_bjs:",MM_bjs)
      print("MM_xiao:",MM_xiao)
      print("pan:",MM_pan)
      print("propose:",MM)
      print("MM_gao:",MM_gao)
      print("MM_zeng:",MM_zeng)

      
      #Results
      MM = [1.194657308925069, 0.31842805019433507, 0.47645371049772217, 0.5704423173752836, 0.6325876675933586, 0.6759608783965123, 0.7072184161225014, 0.7302552901596331, 0.7475480990340648, 0.7607495055522091]
      MM_chen = [3.8231347620992264, 4.174137252729227, 4.210927514045822, 4.22326065350032, 4.229427368083758, 4.233127397154833, 4.235594083202956, 4.2373560018087595, 4.238677440763114, 4.239705226616499]
      MM_bjs = [0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436, 0.7136030428840436]
      MM_xiao = [1.1946573089250687, 0.5973286544625345, 0.6897357188972442, 0.7315752060882262, 0.7555676239140826, 0.7711479769819305, 0.7820867925264978, 0.7901915353410044, 0.7964382059415458, 0.8014004857610418]
      MM_pan = [0.8447502843349883, 0.7711479769819601, 0.6235369239568782, 0.5682401147799955, 0.5393871421030927, 0.5214046091279334, 0.5087325386030257, 0.4989769331665077, 0.49099074599206266, 0.48418000425441227]
      MM_gao = [0.8099999999999998, 0.5399999999999999, 0.46285714285714274, 0.43199999999999994, 0.41806451612903217, 0.41142857142857137, 0.40818897637795265, 0.4065882352941176, 0.4057925636007827, 0.4053958944281524]
      MM_zeng = [1.9115673810496132, 0.8469715334900791, 1.2754755615394795, 1.537270561884043, 1.7002689244660318, 1.797288705357444, 1.8517489854274107, 1.8809015527617445, 1.8960325615624285, 1.9037477593488104]

      plt.figure(figsize=(7, 6))
      #Fig3a & Fig3b
      X = np.linspace(1,len(MZ),len(MZ))
      Y1 = MM
      Y2 = MM_xiao
      Y3 = MM_chen
      Y4 = MM_pan
      Y5 = MM_bjs
      Y6 = MM_gao
      Y7 = MM_zeng

      plt.plot(X,Y1,marker = '>',markersize=4,color='orange',label='RFBD')
      plt.plot(X,Y2,marker = '*',markersize=4,color='blue',linestyle='--',label='RB')
      plt.plot(X,Y3,marker = 'o',markersize=4,color='green',label='MRBD')
      plt.plot(X,Y4,marker = '+',markersize=4,color='red',linestyle=':',label='EMJSD')
      plt.plot(X,Y5,marker = 'x',markersize=4,color='purple',label='BJS')
      plt.plot(X,Y6,marker = 'd',markersize=4,color='hotpink',linestyle='-.',label='IBχ$^2$')
      plt.plot(X,Y7,marker = 'h',markersize=4,color='grey',linestyle='-.',label='FBD$_{SKL}$')
 
      
      plt.xlabel('t')
      plt.ylabel('The divergence measure')
      
      x_major_locator=MultipleLocator(1)
      y_major_locator=MultipleLocator(0.2)
      ax=plt.gca()
      ax.xaxis.set_major_locator(x_major_locator)
      ax.yaxis.set_major_locator(y_major_locator)
      plt.xlim(0.5,10.5)
      plt.ylim(0.08,2)
      
      plt.legend(loc='right', bbox_to_anchor=(1, 0.75))
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
      
      MM = []
      MM_xiao = []
      for i in range(0,len(MZ_)):
            MD = sorted(MZ_[i])
            M13 = MassFunction([({'a'}, 0.05), (MD, 0.95)]) 
            M13_ = {frozenset({'a'}):0.05, frozenset(MD):0.95}

            M14 = MassFunction([({'g'}, 1.00)]) 
            M14_ = {frozenset({'g'}):1.00}
            frame = FRAME([M13.frame(),M14.frame()],2)
            sub_ = Subset(frame)
            fr = len(sub_)

            MM.append(MTD([M13_,M14_],2)[0][1])
            MM_xiao.append(MTD_xiao([M13_,M14_],2)[0][1])

      print("proposed:",MM)
      print("xiao:",MM_xiao)

      #------------------------------------
      
      plt.figure(figsize=(6, 5))
      X = np.linspace(1,len(MZ_),len(MZ_))

      #Results
      Y1_ = [1.396424004376894, 1.5723202245875212, 1.601546800954703, 1.6198520992030614, 1.632151637797948, 1.6405469873161103, 0.9311392606332061, 0.8016139375010333, 0.7536438956126915, 0.6852881602084078, 0.5833604114369452, 0.41920038215598154, 0.22566547338346785]
      Y2_ = [1.396424004376894, 1.4759230178921166, 1.4636709627803588, 1.4575063133851858, 1.4537949763705642, 1.451315479084229, 0.9889041089105235, 0.9240238141899891, 0.9064790152326719, 0.8795058265063024, 0.832610946677565, 0.7298369460809905, 0.22566547338346735]

      #Fig 4
      plt.plot(X,Y1_,marker = '>',markersize=4,color='orange',label='RFBD')
      plt.plot(X,Y2_,marker = '*',markersize=4,color='blue',linestyle='--',label='RB')
      
      plt.xlabel('p')
      plt.ylabel('The divergence measure')
      
      x_major_locator=MultipleLocator(1)
      y_major_locator=MultipleLocator(0.2)
      ax=plt.gca()
      ax.xaxis.set_major_locator(x_major_locator)
      ax.yaxis.set_major_locator(y_major_locator)
      plt.xlim(0.5,13.5)
      plt.ylim(-0.01,1.8)
      
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
      print('Gao:',dsmass_gao([M17_,M18_,M19_,M20_],4))
      print('Zeng:',dsmass_zeng([M17_,M18_,M19_,M20_],4))
      
      
      #======= Mass function 17 18 19 =========== 
      #frame = FRAME([M17.frame(),M18.frame(),M19.frame()],3)
      #sub_ = Subset(frame)
      #fr = len(sub_) 
      #print('RFBD:',dsmass([M17_,M18_,M19_],3))
      #print('RB:',dsmass_xiao([M17_,M18_,M19_],3))
      #print('MRBD:',dsmass_chen([M17_,M18_,M19_],3))
      #print('BJS:',dsmass_bjs([M17_,M18_,M19_],3))
      #print('EMJSD:',dsmass_pan([M17_,M18_,M19_],3))
      #print('Gao:',dsmass_gao([M17_,M18_,M19_],3))
      #print('Zeng:',dsmass_zeng([M17_,M18_,M19_],3))
      #==========================================
      
      
#=======================================================================
      #Fig7a & Fig7b
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

      data_x_6 = np.array([0.3346, 0.2827, 0.0373, 0.0871, 0.0669])#1 Gao
      data_p_6 = np.array([0.2974, 0.3050, 0.1982, 0.0961, 0.0961])#2 Gao
      
      data_x_7 = np.array([0.3963, 0.2725, 0.0512, 0.1185, 0.0673])#1 Zeng
      data_p_7 = np.array([0.3190, 0.2061, 0.0928, 0.2411, 0.1285])#2 Zeng
      
      
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
      data_x_6 = np.concatenate((data_x_6,[data_x_6[0]]))
      data_p_6 = np.concatenate((data_p_6,[data_p_6[0]]))
      data_x_7 = np.concatenate((data_x_7,[data_x_7[0]]))
      data_p_7 = np.concatenate((data_p_7,[data_p_7[0]]))
      
      
      angles = np.concatenate((angles,[angles[0]]))
      labels=np.concatenate((labels,[labels[0]])) 

      
      #--Fig7a--#
      fig = plt.figure(figsize=(7,7),facecolor="white")
      
      ax1 = plt.subplot(111,polar=True)     

      ax1.plot(angles,data_x_1,marker = 'x',markersize=4,color ='purple',label='BJS')
      ax1.fill(angles,data_x_1,facecolor='purple',alpha=0.05)  
      ax1.plot(angles,data_x_2,marker = '+',markersize=4,color ='red',linestyle = ':',label='EMJSD')
      ax1.fill(angles,data_x_2,facecolor='red',alpha=0.05)
      ax1.plot(angles,data_x_3,marker = 'o',markersize=4,color ='green',label='MRBD')
      ax1.fill(angles,data_x_3,facecolor='green',alpha=0.05)    
      ax1.plot(angles,data_x_4,marker = '*',markersize=4,color ='blue',linestyle = '--',label='RB')
      ax1.fill(angles,data_x_4,facecolor='blue',alpha=0.05)    
      ax1.plot(angles,data_x_6,marker = 'd',markersize=4,color ='hotpink',linestyle='-.',label='IBχ$^2$')
      ax1.fill(angles,data_x_6,facecolor='hotpink',alpha=0.05)
      ax1.plot(angles,data_x_7,marker = 'h',markersize=4,color ='grey',linestyle='-.',label='FBD$_{SKL}$')
      ax1.fill(angles,data_x_7,facecolor='grey',alpha=0.05)
      ax1.plot(angles,data_x_5,marker = '>',markersize=4,color ='orange',label='RFBD')
      ax1.fill(angles,data_x_5,facecolor='orange',alpha=0.05)
      
      
      ax1.set_theta_zero_location('N')
      ax1.set_rlim(0, 0.5)
      ax1.set_rticks(np.arange(0.05, 0.50, 0.05))
      ax1.set_rlabel_position(300)
      plt.thetagrids(angles*180/np.pi,labels)          
      plt.legend(loc='right', bbox_to_anchor=(1, 0.93))  

      plt.show()
      
      
      #--Fig7b--#
      ax2 = plt.subplot(111,polar=True)     
      
      ax2.plot(angles,data_p_1,marker = 'x',markersize=4,color ='purple',label='BJS')
      ax2.fill(angles,data_p_1,facecolor='purple',alpha=0.05)   
      ax2.plot(angles,data_p_2,marker = '+',markersize=4,color ='red',linestyle = ':',label='EMJSD')
      ax2.fill(angles,data_p_2,facecolor='red',alpha=0.05)
      ax2.plot(angles,data_p_3,marker = 'o',markersize=4,color ='green',label='MRBD')
      ax2.fill(angles,data_p_3,facecolor='green',alpha=0.05)    
      ax2.plot(angles,data_p_4,marker = '*',markersize=4,color ='blue',linestyle = '--',label='RB')
      ax2.fill(angles,data_p_4,facecolor='blue',alpha=0.05) 
      ax2.plot(angles,data_p_6,marker = 'd',markersize=4,color ='hotpink',linestyle='-.',label='IBχ$^2$')
      ax2.fill(angles,data_p_6,facecolor='hotpink',alpha=0.05)
      ax2.plot(angles,data_p_7,marker = 'h',markersize=4,color ='grey',linestyle='-.',label='FBD$_{SKL}$')
      ax2.fill(angles,data_p_7,facecolor='grey',alpha=0.05)
      ax2.plot(angles,data_p_5,marker = '>',markersize=4,color ='orange',label='RFBD')
      ax2.fill(angles,data_p_5,facecolor='orange',alpha=0.05)

      ax2.set_theta_zero_location('N')

      ax2.set_rlim(0, 0.5)
      ax2.set_rticks(np.arange(0.05, 0.50, 0.05))

      ax2.set_rlabel_position(300)
      plt.thetagrids(angles*180/np.pi,labels)          

      plt.legend(loc='right', bbox_to_anchor=(1, 0.93))  
      plt.show()
      
      












      
