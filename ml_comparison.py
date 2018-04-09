# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import random
import time

# Print dataset resume
def print_dataset(dataset, groupby='class'):
    # Shape
    print("\nShape: ")
    print(dataset.shape)

    # Head
    print("\nHead:")
    print(dataset.head(20))

    # Descriptions
    print("\nDescriptions:")
    print(dataset.describe())

    # Class distribution
    print("\nClass distribution:")
    print(dataset.groupby(groupby).size())

def plot_box_whisker(dataset):
    dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
    plt.show()

def plot_histograms(dataset):
    dataset.hist()
    plt.show()

def plot_scatter_matrix(dataset):
    scatter_matrix(dataset)
    plt.show()

def train_and_validation_sets(dataset, seed=7):
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:len(array[0])-1]
    Y = array[:,len(array[0])-1]
    validation_size = 0.20
    return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

def build_and_check_models(models, X_train, Y_train, seed=7):
    # Test options and evaluation metric
    scoring = 'accuracy'
    splits = 10

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=splits, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)

    return results, names

def compare_algorithms(results, names):
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def predictions_to_models(models, X_train, X_validation, Y_train, Y_validation, verbose = False):
    # Make predictions on validation dataset
    for name, model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)

        if verbose:
            print("==========================================================================")
            print("==> Prediction for model %s", name)
            print("\nAccuracy:")
            print(accuracy_score(Y_validation, predictions))
            print("\nConfusion Matrix:")
            print(confusion_matrix(Y_validation, predictions))
            print("\nClassification Report:")
            print(classification_report(Y_validation, predictions))
        else:
            acc = 100. * accuracy_score(Y_validation, predictions)
            print("%s: %.2f%%" % (name, acc))

# Running
if __name__ == '__main__':
    print("==> Running iris flower example...")

    print("==> Loading Dataset")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)

    print("==> Dataset Summary")
    print_dataset(dataset, 'class')

    print("==> Whisker plot")
    plot_box_whisker(dataset)

    print("==> Histogram plot")
    plot_histograms(dataset)

    print("==> Scatter Plot Matrix")
    plot_scatter_matrix(dataset)

    # Spot Check Algorithms
    models = []
    models.append(('Logistic Regression', LogisticRegression()))
    models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighbors Classifier', KNeighborsClassifier()))
    models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
    models.append(('Gaussian NB', GaussianNB()))
    models.append(('SVM', SVC()))

    print("==> Setting seed")
    # seed = 7
    random.seed(time.time())
    seed = random.randint(0, 2**32 - 1)

    print("==> Getting validation and train sets")
    X_train, X_validation, Y_train, Y_validation = train_and_validation_sets(dataset, seed)

    print("\n==> Building and Checking our models")
    results, names = build_and_check_models(models, X_train, Y_train, seed)

    print("\n==> Comparing results")
    compare_algorithms(results, names)

    print("\n==> Making Accuracy predictions to specific models")
    # predictions_to_models(models, X_train, X_validation, Y_train, Y_validation, True)
    predictions_to_models(models, X_train, X_validation, Y_train, Y_validation)
