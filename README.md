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
### Name: SANTHOSH KUMAR R
### Register Number:212223240153
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_excel('/content/deeplearn.xlsx')
X = dataset1[['input']].values
y = dataset1[['output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_santhosh= NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (ai_santhosh. parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=4000) :
  for epoch in range (epochs) :
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train), y_train)
    loss. backward()
    optimizer.step()
    ai_brain. history['loss'] .append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_santhosh, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_santhosh(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')



loss_df = pd.DataFrame(ai_santhosh.history)


import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_santhosh(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
## Dataset Information
<img width="226" height="449" alt="image" src="https://github.com/user-attachments/assets/36719cee-1734-4225-8e92-0d2f60ba7e33" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="596" height="404" alt="image" src="https://github.com/user-attachments/assets/73b6ba64-4ff0-45de-b0a2-8a518019f019" />

### New Sample Data Prediction

<img width="240" height="60" alt="image" src="https://github.com/user-attachments/assets/fe4e1ff4-ec2d-475e-8a23-ea0ce866d636" />

## RESULT

Include your result here
