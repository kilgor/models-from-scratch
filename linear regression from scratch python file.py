# %% [markdown]
# Video: https://colab.research.google.com/drive/17bs_UTvsNSkdnBMwzhxGqN_eHbu6YBkl?usp=sharing#scrollTo=GmDgN3q0rw6y
# 
# 
# Linear Regression:
# 
# y = wx + b
# 
# y --> Depended Variable
# 
# x --> Independed Variable
# 
# w --> weight
# 
# b --> bias
# 
# -------------------------------------------------
# 
# Gradient Descent:
# Gradient descent is an optimization algorithm used for minimizing the loss function in various machine learning algorithms. It is
# used for updating the parameters of learning model.
# 
# w = w - a * dw
# 
# b = b- a * db
# 
# 
# -------------------------------------------------
# 
# 

# %%
#importing necessary libraries (numpy, pandas)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# %% [markdown]
# Linear regression model creation

# %%
#now create a class under tthe name Linear_Regression
class Linear_Regression():
    
    #we are calling __init__ method to initialize the object
    #initiating the parameters (learning rate and no of iterations)
    def __init__(self, learning_rate, no_of_iterations):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
        
        
        
    #we are calling fit method to train the model
    def fit(self, x, y):
        
        #number of training examples and numer of features         
        self.m, self.n = x.shape #number of rows and columns (m=rows, n=columns)
        
        #initiating the weights and bias of our model
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        
        #implementing the gradient descent algorithm
        for i in range(self.no_of_iterations):
            self.update_weights()         
        
        
        
        
    #we are greating ne function for update the weights
    def update_weights(self,):
        
        y_pred = self.predict(self.x)
        
        #calculating the gradients
        dw = -(2 * (self.x.T).dot(self.y - y_pred)) / self.m
        #m = number of training examples
        #dw = derivative of weights
        #T = transpose of matrix
        #y = actual values
        #y_pred = predicted values
        #x = features
        
        db = -2 * np.sum(self.y - y_pred) / self.m
        #db = derivative of bias
        
        #updating the weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
        #self.w = weights
        #self.b = bias
        #self.learning_rate = learning rate
        #dw = derivative of weights
        #db = derivative of bias
        
        
        
        
    #we are creation predict function to predict the values
    def predict(self, x):
        
        return x.dot(self.w) + self.b
        
        
        

        

# %% [markdown]
# Using Linear Regression model for prediction that we created right abowe
# 

# %% [markdown]
# 1) Set Learning Rate and numbver of iterations; initiate random weight and bias value
# 2) Build linear regression model
# 3) find the "y pred" value for given x value for the corresponding weight and value
# 4) check the loss function for the parameter of the values (difference between "y pred" and "true y")
# 5) update the parameter of the values using gradient descent. (new weight and bias value)
# 6) Step 3, 4, 5 till we get minimum loss function
# Finally we will get the best model (best veight and bias value) as it has minimum loss function
# 

# %%
#now we are uploading the dataset to test our model
df = pd.read_csv(r'D:\python\Machine learning practice\models from scratch\salary_data.csv')
df.head()

# %%
df.shape

# %%
#checking missing parameters
df.isnull().sum()

# %%
#splitting the target column (y) and feature column (x)
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# %%
#splitting data set into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# %% [markdown]
# trainig our model
# 

# %%
#we are monually entering the learning rate and no of iterations
model  = Linear_Regression(learning_rate=0.002 , no_of_iterations=1000)
model.fit(x_train, y_train)


# %%
#printing the weights and bias of our model
print('weight: ' ,model.w[0])
print('bias: ', model.b)

# %% [markdown]
# y = 11068(x) + 16150
# 
# salary = y =11068(years of experience) + 16150
# 
# for train_size=0.66, random_state=0 and learning_rate=0.002 , no_of_iterations=1000

# %%
#predicting salary value for test data
test_data_pred = model.predict(x_test)
#now calculating difference between actual and predicted values
diff = test_data_pred - y_test
diff

# %% [markdown]
# visualizing actual and predicted values of test data

# %%
#visualizing actual and predicted values
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, test_data_pred, color='orange')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# %%
#as for the result, do you think the model is good enough to predict the salary based on years of experience?
#I am asking you copilot not others
#yes, the model is good enough to predict the salary based on years of experience
#because the predicted values are very close to the actual values
#and the model is able to capture the linear relationship between the features and target variable
#and the model is able to generalize well on unseen data

#thank you colpilot for your help
#you are welcome
#I am always here to help you
#(: 
#have a nice day
#you too







# %%



