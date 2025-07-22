import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
X=np.linspace(1,10,100)
Y=2*X+10+np.random.randn(X.shape[0])
print(X)
print(Y)
X.shape
X.reshape(-1,1)
X.shape
model=Sequential()
model.add(Dense(1,input_dim=1,activation='linear'))
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
model.fit(X,Y,epochs=10,verbose=1)
pred=model.predict(X)
plt.scatter(X,Y,label="original data")
plt.plot(X,pred)
plt.show()
