# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:35:18 2020

@author: Rohan Dixit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
def process(img):
    
    image = np.resize(img,(1,8748))
    image = np.reshape(image,(1,-1))
    return image
li_train=[]
for i in range(1,41):
    for j in range(1,11):
        img = cv2.imread(f"orl_face//u{i}//{j}.png")
        image=process(img)
        li_train.append(image)
    
li_label=[]
for i in range(1,41):
    for j in range(1,11):
        li_label.append(i)
        
X=pd.DataFrame(li_train[0])
for i in range(1,400):
    df=pd.DataFrame(li_train[i])
    X=pd.concat([X,df],ignore_index=True)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,li_label,test_size=0.3,random_state=3)
from sklearn.svm import SVC
s=SVC(kernel='linear')
s.fit(X_train,y_train)
p=s.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,p))
# for best parameter and best estimator

param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001], 'kernel':['poly']}
from sklearn.model_selection import GridSearchCV
g=GridSearchCV(SVC(),param_grid,refit=True,verbose=3)#verbose is a quality
g.fit(X_train,y_train)
print(g.best_params_)
print(g.best_estimator_)
gp=g.predict(X_test)
print(confusion_matrix(y_test,gp))
print(classification_report(y_test,gp))
#save the model

import joblib
joblib.dump(g,"model.pkl")
# Load the Model
joblib.load("model.pkl")
# input  the image
img = cv2.imread(f"orl_face//u{32}//{7}.png")
image=process(img)
#predict the image
gp=g.predict(image)
#result 
print("This image is belong to class :", gp)



