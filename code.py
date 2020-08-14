# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:20:38 2020

@author: Gurpreet Singh
"""
def newDF(df,target,j):
    df['Target']=target
    newDF=df[df.Target==j]
    newDF=newDF.drop(columns='Target')
    df=df.drop(columns='Target')
    return newDF


import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

lfw_dataset = fetch_lfw_people(min_faces_per_person=100)
y=lfw_dataset.target
#df=pd.DataFrame(y)
arr=[0,0,0,0,0]
for i in range(0,1140):
    a=y[i]
    arr[a]+=1
"""
train_data,test_data,train_target,test_target = train_test_split(lfw_dataset.data,
lfw_dataset.target, test_size=0.3)
pca = decomposition.PCA(n_components=100, whiten=True)
train_data_pca=pca.fit_transform(train_data)
test_data_pca=pca.fit_transform(test_data)
"""
 
pca = decomposition.PCA(n_components=100, whiten=True)
data_pca=pca.fit_transform(lfw_dataset.data)
data_pca=pd.DataFrame(data_pca)
old_df=data_pca
listc=[]
for i in range(0,32):
    listc.append(i)

train_data2,test_data2,train_target2,test_target2 = train_test_split(data_pca,
lfw_dataset.target, test_size=0.3)

column_retained=0
sum=0
for i in range(0,100):
    sum+=pca.explained_variance_ratio_[i]
    column_retained+=1
    if(sum>0.8):
        break
newdf=old_df[listc]        
a,b,c,d= train_test_split(newdf,lfw_dataset.target, test_size=0.3)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(a,c)
print(classification_report(d,neigh.predict(b),target_names=lfw_dataset.target_names))




#df1=pd.DataFrame(train_data_pca)
#df2=pd.DataFrame(test_data_pca)

df1_p1=newDF(data_pca,lfw_dataset.target,0)   #first person
df1_p2=newDF(data_pca,lfw_dataset.target,2)   #second person
df1_p3=newDF(data_pca,lfw_dataset.target,4)   #third person




colormap=np.array(['r','g','b'])
tsne_2D = TSNE(n_components=2,random_state=0) 
tsne_3D = TSNE(n_components=3,random_state=0) 
df_b=pd.concat([df1_p1,df1_p2,df1_p3],ignore_index=True)
df_b=tsne_2D.fit_transform(df_b)
df_b=pd.DataFrame(df_b,columns=['ts1','ts2'])
classd=[]
for i in range(0,236):
    classd.append(0)
for i in range(0,530):
    classd.append(1)
for i in range(0,144):
    classd.append(2)    
plt.scatter(df_b['ts1'],df_b['ts2'],c=colormap[classd])    
plt.show()
df_b3=tsne_3D.fit_transform(df_b)
df_b3=pd.DataFrame(df_b3,columns=['ts1','ts2','ts3'])
ax=plt.figure().add_subplot(111,projection='3d')
ax.scatter(df_b3['ts1'],df_b3['ts2'],df_b3['ts3'],c=colormap[classd])
ax.set_xlabel('ts1')
ax.set_ylabel('ts2')
ax.set_zlabel('ts3')
plt.show() 

#plt.imshow(pca.mean_.reshape(lfw_dataset.images[0].shape),cmap=plt.cm.bone)

fig = plt.figure(figsize=(6, 6))
for i in range(20):
    ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(lfw_dataset.images[0].shape),
              cmap=plt.cm.bone)


neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_data_pca, train_target)
z=neigh.predict(test_data_pca)
print(classification_report(test_target,z,target_names=lfw_dataset.target_names))
#neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_data, train_target)
z2=neigh.predict(test_data)
print(classification_report(test_target,z2,target_names=lfw_dataset.target_names))
neigh.fit(train_data2, train_target2)
z2=neigh.predict(test_data2)
print(classification_report(test_target2,z2,target_names=lfw_dataset.target_names))


