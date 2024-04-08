import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

#model = LogisticRegression().fit(X, y)
#model = LogisticRegression(max_iter=1000, random_state=42).fit(X_scaled, y)
model = GaussianNB().fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
    
    
