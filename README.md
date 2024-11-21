# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value

## Neural Network Model


![image](https://github.com/user-attachments/assets/989bf911-b444-4e27-b16e-545f341a044a)


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
### Name: VARSHA AJITH
### Register Number:212221230118
```python

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

```
## Dataset Information


![image](https://github.com/user-attachments/assets/ad78cc6e-7638-4035-95cf-39d2f3698c07)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/1ae7337a-bb6b-470e-85d4-5906731e120e)


### Test Data Root Mean Squared Error


![image](https://github.com/user-attachments/assets/41a6d340-dd55-421f-a37f-2327f0fc7119)


### New Sample Data Prediction


![image](https://github.com/user-attachments/assets/4f0b07bc-7291-4eb2-b12a-215d0e9f4f85)


## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
