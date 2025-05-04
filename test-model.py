import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('IRIS.csv')

def train_and_save_model():
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open('test-model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    feature_names = X.columns.tolist()
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    return model

if __name__ == "__main__":
    train_and_save_model()










# Assume the target column is named 'species' and rest are features
#X = df.drop('species', axis=1)
#y = df['species']

# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train RandomForestClassifier
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)

#y_pred = model.predict(X_test)
#print("Predictions: ", y_pred)

#print(model.score(X_test, y_test))

#joblib.dump(clf, 'test-model.pkl')