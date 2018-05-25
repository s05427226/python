import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')
print(iris.head())

sns.pairplot(iris,hue="species")
plt.show()

x = iris.values[:,:4]
y = iris.values[:,4]

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=0)

def one_hot_encode_object_array(arr):
    uniques,ids = np.unique(arr,return_inverse=True)
    return np_utils.to_categorical(ids,len(uniques))

def one_hot_encode_object_pandas(arr):
    return pd.get_dummies(arr).values

y_train_ohe = one_hot_encode_object_pandas(y_train)
y_test_ohe = one_hot_encode_object_pandas(y_test)

model = Sequential()

model.add(Dense(16,input_shape=(4,)))
model.add(Activation('sigmoid'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train_ohe,epochs=100,batch_size=1,verbose=1,validation_split=0.2)

loss,accuracy = model.evaluate(x_test,y_test_ohe)

print('Accuracy = {:.2f}'.format(accuracy))



