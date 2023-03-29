from Desktop import project
drive.mount('/content/project')
# Import os: means import operating system 
# import is a python keyword used to import libraries (libraries are a collection of many useful functions and classes for typically one type)
# https://docs.python.org/3/library/os.html
import os

# numpy is one of the most commonly used matrix manipulation library. Make sure to check out the official documentations for in-depth understanding
# https://numpy.org/doc/stable/user/whatisnumpy.html
import numpy as np

# Let's make an import in order to plot something
# we import matplotlib to plot the data
import matplotlib.pyplot as plt
%matplotlib inline

# to handle dataframes
# https://pandas.pydata.org/docs/user_guide/index.html
import pandas as pd

# to visualize plots more beautifully 
# https://seaborn.pydata.org/tutorial.html
import seaborn as sns

# library for machine learning
# https://scikit-learn.org/stable/getting_started.html
import sklearn
## Suppress Filter Warnings
import warnings
warnings.filterwarnings('ignore')
## Create the data frame
dataset_path = "C:\Users\ACER\Desktop\project\Data.csv"
star_df = pd.read_csv(dataset_path)
star_df.info()
## Check the first 10 rows of data
star_df.head(10)
## Check the input counts for star color 
print("Color                 Count\n-----------------------------\n",
star_df['Star color'].value_counts())
## Visulaising the Star color data

# Adjusting figure size
plt.figure(figsize = (13, 6))

# Sorting in descending format for an eye soothing visualisation
color = pd.DataFrame(star_df['Star color'].value_counts().sort_values(ascending=False)) # New df to avoid changes in main

# Create bar plot for star color
ax = sns.barplot(x = color.index, y = 'Star color' , data = color, palette='magma') # Saving in ax variable to use it later

# Decorate the plot
plt.title("Visualising Star Colors", color = "m", fontsize = 18)  # Add title
plt.ylabel('Star color', color = 'b', fontsize = 15)              # Add y label
plt.xticks(fontsize = 14)                                         # Change x ticks size
plt.yticks(fontsize = 14)                                         # Change y ticks size
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)      # Rotate xticklabels by 90 degrees using ax variable

##to avoid repeating
## Simplest way of doing it manually 
star_df.replace({"Star color" : {"Blue-white" : "Blue-White", 
                                  "Blue White" : "Blue-White", 
                                  "Blue white" : "Blue-White", 
                                  "Blue white " : "Blue-White", 
                                  "Blue-White" : "Blue-White",
                                  "yellow-white" : "Yellow-White",
                                  "Yellowish White" : "Yellow-White",
                                  "yellowish" : "Yellow-White",
                                  "White-Yellow" : "Yellow-White",
                                  "Yellowish" : "Yellow-White",
                                  "white" : "White",
                                  "Whitish" : "White",
                                  "Blue " : "Blue",
                                  "Pale yellow orange" : "Red",
                                  "Orange" : "Red",
                                  "Orange-Red" : "Red"}},
                                  inplace = True)

## Display the final Star color column after cleaning it
print("Color           Count\n-----------------------------\n", star_df['Star color'].value_counts())

##moving to spectral class
## Check the input count for Spectral Class
print("Class Count\n-------------------\n", star_df['Spectral Class'].value_counts())

## Create new df which sorts spectral class in descending way
spectral_class = pd.DataFrame(star_df['Spectral Class'].value_counts().sort_values(ascending=False)) # New df to avoid
                                                                                                     # changes in main data
print("\n\n")

## Create the plot

# Set the size of the figure
plt.figure(figsize=(10, 6))

# Create barplot for spectral class
ax = sns.barplot(x = spectral_class.index, y = 'Spectral Class' , data = spectral_class, palette='magma')

# Decorate the plot
plt.title("Visualising Spectral Class", color = "m", fontsize = 18)  # title
plt.ylabel('Spectral Class', color = 'b', fontsize = 15)             # y label
plt.xticks(fontsize = 14)                                            # resize x ticks
plt.yticks(fontsize = 14)                                            # resize y ticks
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)         # rotate x labels by 45 deg

## Cat codes

# Converting star color into category and then performing cat codes encoding
star_df['Star color'] = star_df['Star color'].astype('category').cat.codes

# Converting spectral class into category and then performing cat codes encoding
star_df['Spectral Class'] = star_df['Spectral Class'].astype('category').cat.codes

# Display the encoded output
print("Star color Encoded Output ~\n", star_df['Star color'].value_counts(),
     "\n\n---------------------------------------------------------------",
      "\n\nSpectral Class Encoded Output ~\n", star_df['Spectral Class'].value_counts())


# Pearson Correlation Matrix
pc_mat = star_df.corr()

# Setting the figure size
fig, ax = plt.subplots(figsize = (10, 10))

# Create heatmap to visualise the correlation
sns.heatmap(pc_mat, cmap = 'Set2', cbar = False, annot = True, annot_kws = {"size" : 14},ax = ax,
           fmt = '.2f', linewidth = 2, square = True)

# Add title to plot
fig = plt.title("Pearson's Correlation", fontsize = 15, color = 'red')

##checking if data is balanced

plt.figure(figsize = (8,8))
splot = sns.barplot(x = star_df['Star type'].unique(), y = star_df['Star type'].value_counts(),
           palette = 'viridis')
plt.title("Visualize if the classes are balanced", color = 'r', fontsize = 14)
plt.bar_label(splot.containers[0],color = 'tab:cyan', weight = 'bold')
plt.ylabel("Count")
plt.xlabel('Star Type')
plt.show()

## Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

## Perform the split - 90% for training, 10% for testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
## Check the shape
print('X_train_shape', X_train.shape)
print('y_train_shape', y_train.shape)
print('X_test_shape', X_test.shape)
print('y_test_shape', y_test.shape)

## For Logistic Regression
from sklearn.linear_model import LogisticRegression

## Build the Classifier
logreg = LogisticRegression(random_state=42, max_iter=10000)
logreg.fit(X_train, y_train)

## Metrics for evaluation of the model
from sklearn.metrics import classification_report

## Predictions + Report
predictions = logreg.predict(X_test)
print(classification_report(y_test, predictions))

##comparing

## Target values in test dataset
test = y_test.values
print("This is how the test target values are:-")
print(test)

## Predicted values
print("\nThis is how the predicted values are:-")
print(predictions)