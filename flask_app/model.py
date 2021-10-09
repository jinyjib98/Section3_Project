import os
import pickle
import sqlite3

from flask import Blueprint

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


model_bp = Blueprint('model', __name__)

df = pd.read_csv('flask_app/data/student-mat.csv')

# G3와 correlation이 높은 G1, G2 삭제: Multicollinearity(다중공선성) 방지
data = df.drop(['G1', 'G2'], axis=1)

# categorical data -> One-Hot encoding
data = pd.get_dummies(data)

# dimentionality reduction -> feature 수가 너무 많으면 과적합 발생(high variance)
correlation = data.corr().abs()['G3'].sort_values(ascending=False)
top10_corr = correlation[:11]

final = data.loc[:, top10_corr.index]

# 굳이 필요없는 feature 삭제
final = final.drop(['higher_no', 'romantic_no'], axis=1)
data = final.reindex(columns=['age', 'failures', 'higher_yes', 'romantic_yes', 'Medu', 'Fedu', 'goout', 'traveltime', 'G3']) 

# Train - Test set split
label = 'G3'
features = np.array(data.drop(label, axis=1))
target = np.array(data[label])

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=42)

# Modeling - Linear Regression
model = LinearRegression()

# train set 학습
model.fit(X_train, y_train)

# test set 예측
y_pred = model.predict(X_test)

# evaluation matrix
mae = mean_absolute_error(y_test, y_pred) # 3.306
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # 4.21

# prediction model save
pickle.dump(model, open('flask_app/model/grade_model.pickle', 'wb'))


# save data into db
DATABASE_PATH = os.path.join(os.getcwd(), 'flask_app/data/data.db')

conn = sqlite3.connect(DATABASE_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS Info;")
cur.execute("""CREATE TABLE Info(
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            age INTEGER NOT NULL,
            failures INTEGER NOT NULL,
            higher INTEGER NOT NULL,
            romantic INTEGER NOT NULL,
            M_edu INTEGER NOT NULL,
            F_edu INTEGER NOT NULL,
            go_out INTEGER NOT NULL,
            travel INTEGER NOT NULL,
            grade INTEGER NOT NULL
            );""")


for i in range(len(data)):
    pick = list(data.loc[i])
    cur.execute("INSERT INTO Info(age, failures, higher, romantic, M_edu, F_edu, go_out, travel, grade) VALUES('{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(pick[0], pick[1], pick[2], pick[3], pick[4], pick[5], pick[6], pick[7], pick[8]))

conn.commit()
cur.close()




