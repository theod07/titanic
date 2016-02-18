import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('./data/train.csv')

print 'train_df.describe(): '
print train_df.describe()
print 'train_df.info():'
print train_df.info()

# list(train_df.columns)
# ['PassengerId',
#  'Survived',
#  'Pclass',
#  'Name',
#  'Sex',
#  'Age',           #
#  'SibSp',
#  'Parch',         # 7 unique values. No null values
#  'Ticket',        # 681 unique values. No null values
#  'Fare',          # 248 unique values. No null values
#  'Cabin',         # 687 nulls. May be able to map cabin to fare
#  'Embarked']      # fill with mode 'S'

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train_df.Parch.unique()                         # array([0, 1, 2, 5, 3, 4, 6])
train_df['Fare'].isnull().sum()                 # 0
train_df[train_df['Cabin'].isnull()].count()    # 687 may be able to map cabin to fare
train_df[train_df['Age'].isnull()].count()      # 177
train_df[train_df['Embarked'].isnull()].count()  # 2

train_df['Embarked'].unique()                   # ['S', 'C', 'Q', nan]
train_df[train_df['Embarked'] == 'S'].count()   # 644
train_df[train_df['Embarked'] == 'C'].count()   # 168
train_df[train_df['Embarked'] == 'Q'].count()   # 77

embarked_mode = train_df['Embarked'].mode()[0]
train_df['Embarked'].fillna(embarked_mode, inplace=True)
age_mode = train_df.Age.mode().values
age_median = train_df.Age.dropna().median()
# train_df['Age'].fillna(age_mode, inplace=True)
train_df['Age'].fillna(age_median, inplace=True)

train_df['embarked_cat'] = train_df['Embarked'].map( {'S': 1, 'C': 2, 'Q':3} ).astype(int)

# model features: Pclass, Gender, Age, embarked_cat
# fill NaN age with age_mode
# results: 0.57895

# model features: Pclass, Gender, Age, embarked_cat
# fill Nan age with age_median
# results:


X_train = train_df[['Pclass', 'Gender', 'Age', 'embarked_cat']]
y_train = train_df['Survived']
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


test_df = pd.read_csv('./data/test.csv')
test_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['Embarked'].fillna(embarked_mode, inplace=True)
# test_df['Age'].fillna(age_mode, inplace=True)
test_df['Age'].fillna(age_median, inplace=True)
test_df['embarked_cat'] = train_df['Embarked'].map( {'S': 1, 'C': 2, 'Q':3} ).astype(int)
X_test = test_df[['Pclass', 'Gender', 'Age', 'embarked_cat']]

test_df['Survived'] = rf.predict(X_test)
test_df.to_csv('prediction.csv', index_label=False, index=False, columns=['PassengerId', 'Survived'])
