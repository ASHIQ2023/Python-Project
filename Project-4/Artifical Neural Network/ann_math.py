import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# import the dataset
df = pd.read_csv("C:\\Users\\user\\Desktop\\Temp code\\Research\\Neural Network\\ann_math.csv")

# Separating features and target variable
X = df[['X']]
y = df['Y'].values.ravel()

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.24, random_state=20)

# Initializing and training the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(2, 2), activation='relu', solver='lbfgs', learning_rate='adaptive')
mlp.fit(x_train, y_train)

# Making predictions
predictions = mlp.predict(x_test)

# Creating DataFrame for test features and predictions
x_test_df = pd.DataFrame(x_test, columns=['X_TEST'])
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Y'])
final_result = pd.concat([x_test_df, predictions_df], axis=1)

# Accuracy
scr=mlp.score(x_test, y_test)
print(f"\n {scr}")

# Mean Value Calculation
meanf=X.mean()
print(f"\n {meanf}")


#standard deviation
stdf = X.std()
print(f"\n {stdf}")


#print(final_result)
weights=mlp.coefs_
weights_flattened=[]
for layer in weights:
  weights_flattened.extend(layer.flatten())
weights_df=pd.DataFrame(weights_flattened)
columns = [f"Layer_{i+1}" for i in range(len(weights_flattened))]
weights_df=pd.DataFrame(weights_flattened, columns=['Weights'])
print(f"\n {weights_df}")

#Bias
bias=mlp.intercepts_
bias_flattened=[]
for layer in bias:
  bias_flattened.extend(layer.flatten())
columns=[f"Layer_{i+1}" for i in range(len(bias_flattened))]
bias_df=pd.DataFrame(bias_flattened, columns=['bias'])
print(bias_df)

# Accuracy
scr=mlp.score(x_test, y_test)
print(f"\n {scr}")

#Final Result
print(f"\n  {final_result}")

