import torch
import torch.nn as nn
import torch.nn.functional as F


# create a model class that inherits nn.module
class Model(nn.Module): # A class(nn.Module) that inherits a subclass(Model)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3): # constructor method of the model
        super().__init__() # constructor of the parent class
        self.fc1 = nn.Linear(in_features, h1) # instance of the nn.Linear class (Fully Connected Linear Layer)
        self.fc2 = nn.Linear(h1, h2) # Another instance (Fully connected Linear Layer)
        #self.fc3 = nn.Linear(h2, out_features) # Another instance (Fully Connected Linear Layer)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
    
        return x


# pick a randsom seed for randomization
torch.manual_seed(41)
# create an instance of model
model = Model()


import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')



url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
#from sklearn import datasets
#import pandas as pd
#iris = datasets.load_iris()
#my_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# Change Last Column From Strng to int
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 1.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 2.0)



# Train test and split
X = my_df.drop('variety', axis=1)
y = my_df['variety']


# convert this into numpy arrays
X = X.values
y = y.values


from sklearn.model_selection import train_test_split


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)



import torch


# convert X labels to FloatTensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)



# Convert y labels to tensor long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)




import torch.nn as nn



# Set the criterion of model to measure the error, how far off the predictions are from:
criterion = nn.CrossEntropyLoss()
# choose the optimizer (Adam Optimizer), lr = learning rate (if error does not go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Train our Model
epoch = 300 # Epochs (one run through all the training data in our network)
losses = [] 
for i in range(epoch): 
    y_pred = model.forward(X_train) # Go forward and get a prediction
    loss = criterion(y_pred, y_train) # Measure the error
    losses.append(loss.detach().numpy()) # keep track of our loss
    if i%10 == 0:
        print(f"Epoch: {i} and loss: {loss}") 

    # Backpropagation: take the error rate data from the forward propagation and feed it back through 
    # the network to fine tune the weights
    optimizer.zero_grad 
    loss.backward()
    optimizer.step()
    


plt.plot(range(epoch), losses)
plt.ylabel("loss/error")
plt.xlabel("epochs")



# Evaluate our model on test data set
with torch.no_grad(): # Basically turn off back propagation
    y_eval = model.forward(X_test) # X-test are features are features from our test set, y_eaval is 
    loss = criterion(y_eval, y_test) # find the loss or error


print(loss)



correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f"{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}") 
        # will tell us what type of flower class our network th

        # correct or not
        if y_val.argmax().item() == y_test[i]:
            correct+=1
print(f"we got {correct} correct")



# Find information from new data
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])



with torch.no_grad():
    y_eval = model(new_iris)

    if y_test[i] == 0:
        x = "Setosa"
    elif y_test[i] == 1:
        x = "Versicolor"
    else:
        x = "Viginica"
        
    print(f'{str(y_val)} \t {x}')



newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])


with torch.no_grad():
    y_val = model(newer_iris)

    if y_test[i] == 0:
        x = "Setosa"
    elif y_test[i] == 1:
        x = "Versicolor"
    else:
        x = "Viginica"
        
    print(f'{str(y_val)} \t {x}')

