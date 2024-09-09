# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: V NAVEENKUMAR
### Register Number:212221230068
~~~

from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('SD2').sheet1
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'X':'int'})
df = df.astype({'Y':'int'})

x = df[["X"]] .values
y = df[["Y"]].values

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

ai = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=3000)

loss_plot = pd.DataFrame(ai.history.history)
loss_plot.plot()



err = rmse()
preds = ai.predict(x_test)
err(y_test,preds)
x_n1 = [[30]]
x_n_n = scaler.transform(x_n1)
ai.predict(x_n_n)
~~~

## Dataset Information

![image](https://github.com/user-attachments/assets/70df72b8-cd08-4233-a41b-59c2123f46e2)


## OUTPUT

### Training Loss Vs Iteration Plot

![363577161-1ae7337a-bb6b-470e-85d4-5906731e120e](https://github.com/user-attachments/assets/38245132-385d-42e1-abf1-fa410198e00f)


### Test Data Root Mean Squared Error
![363577294-41a6d340-dd55-421f-a37f-2327f0fc7119](https://github.com/user-attachments/assets/432d89bf-f4d8-47a7-8d13-f72420b263ea)


### New Sample Data Prediction
![363577368-4f0b07bc-7291-4eb2-b12a-215d0e9f4f85](https://github.com/user-attachments/assets/92599975-f1ba-4733-97b0-e5cb836f9728)


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.

