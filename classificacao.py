#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn import svm
import numpy
import pandas as pd
import jsonlines
from matplotlib_venn import  venn2, venn2_circles, venn2_unweighted,  venn3, venn3_unweighted, venn3_circles
import scipy
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import timeit  # calcular metrica de tempo
import sys
from sklearn.metrics import silhouette_score


ini = timeit.default_timer()
SEED = 42
#dataset = "aisopos_ntua_2L"
#dataset = "reut"
#index_fold = 0

dataset = sys.argv[1]
index_fold = int(sys.argv[2])


# In[12]:


def classifica(x_train, y_train, x_test):
    estimator = svm.LinearSVC(C=1, random_state=SEED, max_iter=1000)
    #estimator = LogisticRegression(solver='liblinear')
    estimator.fit(x_train,y_train)
    return estimator.predict(x_test)    

def arredonda(number, precisao=1):
    """ Arredonda number in precision. Example: arredonda(2.1234, 2); Return='2.12'"""
    return float(f'%.{precisao}f'%(number))


# In[14]:


path = "../combinacao_atencao/"
ids = pickle.load( open(f'{path}dataset/{dataset}/splits/split_10_with_val.pkl', 'rb') )
if dataset == 'sst2':
    x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test = load_svmlight_files([open(f'{path}kaggle_tfidf/{dataset}_train{index_fold}', 'rb'), open(f'{path}kaggle_tfidf/{dataset}_val{index_fold}', 'rb'), open(f'{path}kaggle_tfidf/{dataset}_test{index_fold}', 'rb')])
else:
    x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test = load_svmlight_files([open(f'{path}kaggle_tfidf/{dataset}_tfidf_train{index_fold}', 'rb'), open(f'{path}kaggle_tfidf/{dataset}_tfidf_val{index_fold}', 'rb'), open(f'{path}kaggle_tfidf/{dataset}_tfidf_test{index_fold}', 'rb')])



#estimator = RandomForestClassifier(random_state=SEED)
#estimator = LogisticRegression(solver='liblinear')

#estimator = GridSearchCV(estimator, [{'C':  [0.01, 0.1, 1, 10]}], cv=5, scoring='f1_macro', n_jobs=-1)
#estimator = NearestCentroid()

#y_pred_tfidf = classifica(x_train_tfidf, y_train, x_test_tfidf)



#-----------------------------
# path = "../fasttext/"
# nome_experimento= f'{path}{dataset}_fasttext' 
# webkb = jsonlines.open(f'{nome_experimento}.json')
# docs = []
# for line in webkb:
#     docs.append( (line['id'], line['fasttext'], line['label']) )    
    
# X = pd.DataFrame(docs, columns=['id', 'fasttext', 'label'])
# x_train = X.query(f"id == {ids['train_idxs'][index_fold]}")
# x_val = X.query(f"id == {ids['val_idxs'][index_fold]}")
# x_test = X.query(f"id == {ids['test_idxs'][index_fold]}")
# y_pred_fasttext = classifica(list(x_train['fasttext']), y_train, list(x_test['fasttext']))
# x_train_tfidf_fasttext = hstack([ x_train_tfidf, scipy.sparse.csr_matrix( list(x_train['fasttext']) ) ])
# x_test_tfidf_fasttext = hstack([ x_test_tfidf, scipy.sparse.csr_matrix( list(x_test['fasttext'])  ) ])
# y_pred_tfidf_fasttext = classifica(x_train_tfidf_fasttext, y_train, x_test_tfidf_fasttext)

#-----------------------------
path = "../combinacao_atencao/vader/"
nome_experimento= f'{path}{dataset}_vader' 
webkb = jsonlines.open(f'{nome_experimento}.json')
docs = []
for line in webkb:
    docs.append( (line['id'], line['vader'], line['label']) )    
    
X = pd.DataFrame(docs, columns=['id', 'vader', 'label'])
x_train_vader = X.query(f"id == {ids['train_idxs'][index_fold]}")
x_val_vader = X.query(f"id == {ids['val_idxs'][index_fold]}")
x_test_vader = X.query(f"id == {ids['test_idxs'][index_fold]}")
y_pred_vader = classifica(list(x_train_vader['vader']), y_train, list(x_test_vader['vader']))
#x_train_tfidf_vader = hstack([ x_train_tfidf, scipy.sparse.csr_matrix( list(x_train_vader['vader']) ) ])
#x_test_tfidf_vader = hstack([ x_test_tfidf, scipy.sparse.csr_matrix( list(x_test_vader['vader'])  ) ])
#y_pred_tfidf_vader = classifica(x_train_tfidf_vader, y_train, x_test_tfidf_vader)


#-----------
path = "../combinacao_atencao/tfidf_svd/"
nome_experimento= f'{path}{dataset}_tfidf_svd_768_fold{index_fold}' 
webkb = jsonlines.open(f'{nome_experimento}.json')
docs = []
for line in webkb:
    docs.append( (line['id'], line['tfidf'], line['label']) )    
    
X = pd.DataFrame(docs, columns=['id', 'tfidf', 'label'])
x_train_tfidf_svd = X.query(f"id == {ids['train_idxs'][index_fold]}")
x_val_tfidf_svd = X.query(f"id == {ids['val_idxs'][index_fold]}")
x_test_tfidf_svd = X.query(f"id == {ids['test_idxs'][index_fold]}")
y_pred_tfidf_svd = classifica(list(x_train_tfidf_svd['tfidf']), y_train, list(x_test_tfidf_svd['tfidf']))

x_train_tfidf_svd_vader = hstack([ scipy.sparse.csr_matrix( list(x_train_tfidf_svd['tfidf']) ), scipy.sparse.csr_matrix( list(x_train_vader['vader']) ) ])
x_test_tfidf_svd_vader = hstack([ scipy.sparse.csr_matrix( list(x_test_tfidf_svd['tfidf']) ), scipy.sparse.csr_matrix( list(x_test_vader['vader']) ) ])
y_pred_tfidf_svd_vader = classifica(x_train_tfidf_svd_vader, y_train, x_test_tfidf_svd_vader)


#----------------
escreve =  jsonlines.open(f'pred/{dataset}{index_fold}.json', 'a')

# doc = {'index_fold' : index_fold, 'representation' : 'tfidf', 'y_pred' : list(y_pred_tfidf), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf, average='macro'),
#                        'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf, average='micro'),
#                        'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf, average='weighted'),
#                        'time' : timeit.default_timer() - ini
#                       }
# escreve.write(doc)
# doc = {'index_fold' : index_fold, 'representation': 'fasttext', 'y_pred' : list(y_pred_fasttext), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_fasttext, average='macro'),
#                        'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_fasttext, average='micro'),
#                        'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_fasttext, average='weighted'),
#                        'time' : timeit.default_timer() - ini
#                       }
# escreve.write(doc)

# doc = {'index_fold' : index_fold, 'representation': 'vader', 'y_pred' : list(y_pred_vader), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_vader, average='macro'),
#                        'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_vader, average='micro'),
#                        'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_vader, average='weighted'),
#                        'time' : timeit.default_timer() - ini
#                       }
# escreve.write(doc)

# doc = {'index_fold' : index_fold, 'representation': 'tfidf_fasttext', 'y_pred' : list(y_pred_tfidf_fasttext), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_fasttext, average='macro'),
#                        'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_fasttext, average='micro'),
#                        'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_fasttext, average='weighted'),
#                        'time' : timeit.default_timer() - ini
#                       }

# doc = {'index_fold' : index_fold, 'representation': 'tfidf_vader', 'y_pred' : list(y_pred_tfidf_vader), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_vader, average='macro'),
#                        'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_vader, average='micro'),
#                        'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_vader, average='weighted'),
#                        'time' : timeit.default_timer() - ini
#                       }
#escreve.write(doc)

doc = {'index_fold' : index_fold, 'representation' : 'tfidf_svd', 'y_pred' : list(y_pred_tfidf_svd), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_svd, average='macro'),
                       'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_svd, average='micro'),
                       'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_svd, average='weighted'),
                       'time' : timeit.default_timer() - ini
                      }
escreve.write(doc)

doc = {'index_fold' : index_fold, 'representation': 'tfidf_svd_vader', 'y_pred' : list(y_pred_tfidf_svd_vader), 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_svd_vader, average='macro'),
                       'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_svd_vader, average='micro'),
                       'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_tfidf_svd_vader, average='weighted'),
                       'time' : timeit.default_timer() - ini
                      }
escreve.write(doc)

#print(f"Macro-f1: { sklearn.metrics.f1_score( y_test, y_pred, average='macro')}" )
#print(f"Micro-f1: { sklearn.metrics.f1_score( y_test, y_pred, average='micro')}" )
#macro_lista.append(sklearn.metrics.f1_score( y_test, y_pred, average='macro'))


# In[8]:

# vader
#y_pred_fasttext = y_pred_vader
#y_pred_tfidf_fasttext = y_pred_tfidf_vader

#tfidf svd
y_pred_tfidf = y_pred_tfidf_svd
y_pred_fasttext = y_pred_vader
y_pred_tfidf_fasttext = y_pred_tfidf_svd_vader
#----------

if int(index_fold) == 0:
    acertos_tfidf = y_test == y_pred_tfidf
    acertos_tfidf = set( numpy.where(acertos_tfidf == True)[0] )
    acertos_fasttext = y_test == y_pred_fasttext
    acertos_fasttext = set( numpy.where(acertos_fasttext == True)[0] )
    acertos_tfidf_fasttext = y_test == y_pred_tfidf_fasttext
    acertos_tfidf_fasttext = set( numpy.where(acertos_tfidf_fasttext == True)[0] )

    #print(f'acertos_tfidf:          {acertos_tfidf}')
    #print(f'acertos_fasttext:       {acertos_fasttext}')
    #print(f'acertos_tfidf_fasttext: {acertos_tfidf_fasttext}')

    A = acertos_tfidf; B = acertos_fasttext; C = acertos_tfidf_fasttext
    apenas_A = A.difference(B, C) 
    apenas_B = B.difference(A, C) 
    apenas_C = C.difference(A, B) 



    osDois = A.intersection(B) 

    nenhum = len(y_test) - len(A.union(B,C))

    #print(len(apenas_A))


    #venn2(subsets = (len(acertos_tfidf), len(acertos_fasttext), len(acertos_tfidf_fasttext)), set_labels = (f'TFIDF\n\n Nenhum Acertou: {nenhum}  ({arredonda(nenhum/len(y_test)*100)}%)', 'Vader'), set_colors=("orange", "blue"),alpha=0.7)
    #print(f'No hits: {nenhum} ({arredonda(nenhum/len(y_test)*100)}%)')

    #venn2(subsets = (10, 0, 0), set_labels = ('TFIDF', 'Vader'), set_colors=("orange", "blue"),alpha=0.7)



    # A B AB C AC CB ABC
    venn3_unweighted(subsets=(len(apenas_A), len(apenas_B), len( A.intersection(B)),
    len(apenas_C), len(A.intersection(C)) , len(C.intersection(B)) , len(A.intersection(B,C) )), 
        set_labels=(f'TFIDF_SVD', 'Vader', f'Concat(TFIDF_SVD,Vader)\n\n Total: TFIDF_SVD: {len(A)}, Vader: {len(B)}, Concat: {len(C)}\n No Hit: {nenhum} ({arredonda(nenhum/len(y_test)*100)}%) '), 
        set_colors=("orange", "blue", "red"), alpha=0.7)
    # \n\n Dataset: {dataset}\n Total: TFIDF: {len(A)}, Vader: {len(B)}, Concat: {len(C)}\n No Hit: {nenhum}  ({arredonda(nenhum/len(y_test)*100)}%)

    plt.gcf().set_size_inches(5, 5)
    plt.savefig(f'fig/{dataset}_tfidf_svd.jpg', format='jpg', dpi=300) 


        # In[ ]:

from sklearn.metrics import silhouette_score

#silhouette_score(list(x_test['bert']), list(x_test['label']))

# doc = {'tfidf_silhouette': silhouette_score(x_test_tfidf, y_test),  
# 'fasttext_silhouette': silhouette_score(list(x_test['fasttext']), y_test),  
# 'tfidf_fasttext_silhouette': silhouette_score(x_test_tfidf_fasttext, y_test) }  

#doc = {'vader_silhouette': silhouette_score(list(x_test_vader['vader']), y_test),  
#'tfidf_vader_silhouette': silhouette_score(x_test_tfidf_vader, y_test) } 

doc = {'tfidf_svd_silhouette': silhouette_score(list(x_test_tfidf_svd['tfidf']), y_test),  
'tfidf_svd_vader_silhouette': silhouette_score(x_test_tfidf_svd_vader, y_test) } 

escreve =  jsonlines.open(f'result/{dataset}{index_fold}_separabilidade.json', 'a')
escreve.write(doc)
