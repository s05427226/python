import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils

(X_train,y_train) ,(X_test,y_test) = mnist.load_data("./data")

print(X_train.shape)
print(X_test.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i],cmap='gray',interpolation='none')
    plt.title("class {}".format(y_train[i]))
plt.show()

X_train = X_train.reshape(len(X_train),-1)
X_test = X_test.reshape(len(X_test),-1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = (X_train - 127) / 127
X_test = (X_test - 127)/127

nb_classes = 10

y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

model = Sequential()

model.add(Dense(512,input_shape=(784,),kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(512,kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(nb_classes))
model.add(Activation("softmax"))

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size = 64,epochs=20,verbose = 1,validation_split=0.05)


loss,accuracy = model.evaluate(X_test,y_test)

print("Total loss:",loss)
print("Accuracy:",accuracy)
