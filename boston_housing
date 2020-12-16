import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split

# Download data from https://www.kaggle.com/puxama/bostoncsv
data = pd.read_csv(r'C:\File_path\Boston.csv')

# Prepare data
x1 = data['rm'].values 
x2 = data['lstat'].values
Y = data['medv'].values
m = len(x1)
nx = 2  # number of features
Y = Y.reshape(m, 1)
X = np.block([x1.reshape(m, 1), x2.reshape(m, 1)])

# Split the training and test data set 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Create (compile) model
alpha = .0001
epoch_n = 25
model = Sequential()
model.add(Dense(units=1, activation='linear', input_dim=nx))
sgd=optimizers.SGD(learning_rate=alpha)
model.compile(optimizer=sgd, loss='mse')

# Train (fit) model
mfit = model.fit(X_train, Y_train, batch_size=5, verbose=0, epochs=epoch_n, shuffle=False)

# Plot the Cost Function vs Iterations
J = mfit.history['loss']
plt.plot(J[2:-1]) 
plt.title('J Plot')
plt.ylabel('Cost Function J')
plt.xlabel('Iteration')
print('The final train value of J is ', J[epoch_n - 1])
