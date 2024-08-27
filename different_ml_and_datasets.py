
import streamlit as st  # Importing the Streamlit library for web application development
import numpy as np  # Importing NumPy for numerical computations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from sklearn import datasets  # Importing datasets from scikit-learn
from sklearn.model_selection import train_test_split  # Importing function to split data into training and testing sets
from sklearn.decomposition import PCA  # Importing PCA for dimensionality reduction
from sklearn.neighbors import KNeighborsClassifier  # Importing K-Nearest Neighbors classifier
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest classifier
from sklearn.metrics import accuracy_score  # Importing accuracy score metric

# Title and description for the web application
st.write('''
# Explore different ML models and datasets 
Daikhtay hain kon sa best hai in may say?
''')

# Sidebar dropdown for selecting the dataset
dataset_name = st.sidebar.selectbox(
    'Select Dataset',  # Label for the dropdown
    ('Iris', 'Wine', 'Breast Cancer')  # Options for the dropdown
)

# Function to load the dataset based on the user's selection
def get_dataset(dataset_name):
    data = None  # Initialize the variable to hold the dataset
    if dataset_name == 'Iris':  # If the selected dataset is 'Iris'
        data = datasets.load_iris()  # Load the Iris dataset
    elif dataset_name == 'Wine':  # If the selected dataset is 'Wine'
        data = datasets.load_wine()  # Load the Wine dataset
    else:  # Otherwise, load the Breast Cancer dataset
        data = datasets.load_breast_cancer()
    x = data.data  # Features of the dataset
    y = data.target  # Labels of the dataset
    return x, y  # Return the features and labels

# Get the selected dataset's features and labels
x, y = get_dataset(dataset_name)

# Display the shape of the dataset (number of samples and features)
st.write('Shape of dataset:', x.shape)
# Display the number of unique classes in the dataset
st.write('Number of classes:', len(np.unique(y)))

# Function to add parameter selection widgets in the sidebar based on the classifier
def add_parameter_ui(classifier_name):
    params = dict()  # Empty dictionary to store parameters
    if classifier_name == 'SVM':  # If the selected classifier is 'SVM'
        C = st.sidebar.slider('C', 0.01, 10.0)  # Slider to select the 'C' parameter
        params['C'] = C  # Add 'C' to the parameters dictionary
    elif classifier_name == 'KNN':  # If the selected classifier is 'KNN'
        K = st.sidebar.slider('K', 1, 15)  # Slider to select the number of neighbors 'K'
        params['K'] = K  # Add 'K' to the parameters dictionary
    else:  # If the selected classifier is 'Random Forest'
        max_depth = st.sidebar.slider('max_depth', 2, 15)  # Slider to select the maximum depth of the trees
        params['max_depth'] = max_depth  # Add 'max_depth' to the parameters dictionary
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)  # Slider to select the number of trees in the forest
        params['n_estimators'] = n_estimators  # Add 'n_estimators' to the parameters dictionary
    return params  # Return the dictionary of parameters

# Sidebar dropdown for selecting the classifier
classifier_name = st.sidebar.selectbox(
    'Select Classifier',  # Label for the dropdown
    ('KNN', 'SVM', 'Random Forest')  # Options for the dropdown
)

# Get the parameters for the selected classifier
params = add_parameter_ui(classifier_name)

# Function to create the classifier based on the user's selection
def get_classifier(classifier_name, params):
    clf = None  # Initialize the variable to hold the classifier
    if classifier_name == 'SVM':  # If the selected classifier is 'SVM'
        clf = SVC(C=params['C'])  # Create an SVM classifier with the selected 'C' parameter
    elif classifier_name == 'KNN':  # If the selected classifier is 'KNN'
        clf = KNeighborsClassifier(n_neighbors=params['K'])  # Create a KNN classifier with the selected 'K' parameter
    else:  # If the selected classifier is 'Random Forest'
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)  # Create a Random Forest classifier with the selected parameters
    return clf  # Return the classifier

# Create the classifier with the selected parameters
clf = get_classifier(classifier_name, params)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Predict the labels for the testing data
y_pred = clf.predict(x_test)

# Calculate the accuracy of the classifier
acc = accuracy_score(y_test, y_pred)

# Display the selected classifier's name and its accuracy
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')

# PCA for dimensionality reduction to 2D for visualization
pca = PCA(2)
x_projected = pca.fit_transform(x)  # Project the data onto 2 principal components

x1 = x_projected[:, 0]  # First principal component
x2 = x_projected[:, 1]  # Second principal component

# Create a scatter plot of the projected data
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')  # Scatter plot with colors corresponding to the labels
plt.xlabel("Principal Component 1")  # Label for the x-axis
plt.ylabel('Principal Component 2')  # Label for the y-axis
plt.colorbar()  # Add a color bar to show the mapping of labels to colors

# Display the plot in the Streamlit app
st.pyplot(fig)
