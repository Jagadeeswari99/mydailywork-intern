
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

data = {
    "Genre": ["Action","Drama","Comedy","Thriller","Action"],
    "Director": ["A","B","C","D","A"],
    "Votes": [1000,2000,1500,1800,2500],
    "Duration": [120,140,100,110,150],
    "Rating": [8.5,7.2,6.8,7.9,8.8]
}

df = pd.DataFrame(data)

le1 = LabelEncoder()
le2 = LabelEncoder()

df["Genre"] = le1.fit_transform(df["Genre"])
df["Director"] = le2.fit_transform(df["Director"])

X = df.drop("Rating", axis=1)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Predictions:", preds)
print("MAE:", mean_absolute_error(y_test, preds))
