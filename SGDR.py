import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
n=x
from sklearn.preprocessing import *

s=StandardScaler()
x=s.fit_transform(x)

def learning_schedule(t):
    t1=5;t2=50
    return t1/(t+t2)

def SGD(theta,x,y):
    n_epochs=50
    m=len(x)
    lr=0.01
    for epoch in range(n_epochs):
        for i in range(m):
            random_index=np.random.randint(m)
            xi=x[random_index:random_index+1]
            yi=y[random_index:random_index+1]
            
            gradients=xi.T.dot(xi.dot(theta)-yi)
            #lr=learning_schedule(epoch*m+i)
            theta=theta-lr*gradients
    return theta


x=np.c_[np.ones((len(x),1)),x]
theta=np.random.randn(x.shape[1],1)

th=SGD(theta,x,y)

y_pred=x.dot(th)

acc=np.mean(y==y_pred)


from sklearn.metrics import *

err=mean_absolute_error(y,y_pred)
print(err)

plt.scatter(n,y,c='r')
plt.plot(n,y_pred,c='b')

#
#from sklearn.linear_model import *
#reg=SGDRegressor(max_iter=50)
#reg.fit(x,y)
#y_hat=reg.predict(x)
#
#
#er=mean_absolute_error(y,y_hat)



