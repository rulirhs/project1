import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from deap import base, creator, tools, algorithms
import random

data = pd.read_csv('test.csv')

le = LabelEncoder()
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data = data.dropna()

X = data.drop(['id', 'satisfaction'], axis=1)
y = data['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred)
nb_precision = precision_score(y_test, y_pred)
nb_recall = recall_score(y_test, y_pred)

print(f"Naive Bayes - Accuracy: {nb_accuracy*100:.2f}%")
print(f"Naive Bayes - Precision: {nb_precision*100:.2f}%")
print(f"Naive Bayes - Recall: {nb_recall*100:.2f}%")

def eval_genome(genome):
    selected_features = [index for index in range(len(genome)) if genome[index] == 1]
    if len(selected_features) == 0:
        return 0,
    
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    nb_model = GaussianNB()
    nb_model.fit(X_train_selected, y_train)
    y_pred = nb_model.predict(X_test_selected)
    
    return accuracy_score(y_test, y_pred),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_genome)

population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=False)

best_individual = tools.selBest(population, k=1)[0]
best_features = [index for index in range(len(best_individual)) if best_individual[index] == 1]

X_train_selected = X_train.iloc[:, best_features]
X_test_selected = X_test.iloc[:, best_features]

nb_model.fit(X_train_selected, y_train)
y_pred_ga = nb_model.predict(X_test_selected)

ga_nb_accuracy = accuracy_score(y_test, y_pred_ga)
ga_nb_precision = precision_score(y_test, y_pred_ga)
ga_nb_recall = recall_score(y_test, y_pred_ga)

print(f"GA + Naive Bayes - Accuracy: {ga_nb_accuracy*100:.2f}%")
print(f"GA + Naive Bayes - Precision: {ga_nb_precision*100:.2f}%")
print(f"GA + Naive Bayes - Recall: {ga_nb_recall*100:.2f}%")
