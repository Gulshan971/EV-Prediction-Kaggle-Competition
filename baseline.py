import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier , AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop(columns = ["Will_Buy_EV" , "id"])
y = train["Will_Buy_EV"].map({"Yes" : 1 , "No" : 0})

X_kaggle_test = test.drop(columns=["id"])

print("Training features shape" , X.shape)
print("Training outcome shape : " ,y.shape)

print("Total Distribution : \n")
print(y.value_counts(normalize=True))

numerical_columns = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level"
]

categorical_columns = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]

for column in categorical_columns:
    X[column] = X[column].astype("category")
    X_kaggle_test[column] = X_kaggle_test[column].astype("category")

X_train , X_test , y_train , y_test = train_test_split(
     X , 
     y ,
    test_size = 0.2 ,
    random_state = 42 ,
    stratify = y
)


print("Taining shape" , X_train.shape)
print("Testing shape" , X_test.shape)
print("Theoretical mean" , y_train.mean())
print("Actual mean" , y_test.mean())



preprocessor_tree = ColumnTransformer(
    transformers = [
        # ("num" , StandardScaler() ,numerical_columns ),
        ("cat" ,
        OrdinalEncoder(handle_unknown = "use_encoded_value" , unknown_value = -1 ),
        categorical_columns),
    ],
  remainder="passthrough"   
)


tree_model = Pipeline(
    steps = [
        ("preprocessor" , preprocessor_tree),
        ("classifier",
         HistGradientBoostingClassifier(
             max_iter=400,
             random_state=42,
             learning_rate=0.1,
             max_leaf_nodes=31
         )
         )
    ]
)


tree_model.fit(X_train,y_train)

tree_model.predict(X_test)

tree_probabilities = tree_model.predict_proba(X_test)[:, 1]


tree_auc = roc_auc_score(
    y_test,
    tree_probabilities
)

print("Tree ROC-AUC:",tree_auc)


"""
print("Number of predictions:", len(tree_probabilities))
print("First 10 probabilities:", tree_probabilities[:10])

submission = pd.DataFrame({
    "id": test["id"],
    "Will_Buy_EV": tree_probabilities
})

submission.to_csv("submission.csv", index=False)

print(submission.head())
print(submission.shape)

print("Missing values:")
print(submission.isnull().sum())

print("Minimum probability:", submission["Will_Buy_EV"].min())
print("Maximum probability:", submission["Will_Buy_EV"].max())

print("Unique IDs:", submission["id"].nunique())

"""

# result = permutation_importance(
#     tree_model,
#     X_test,
#     y_test,
#     scoring="roc_auc",
#     n_repeats=3,
#     random_state=42
# )

# importance = pd.Series(
#     result.importances_mean,
#     index=X_test.columns
# ).sort_values(ascending=False)

# print(importance)

# ada_model = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor_tree),
#         ("classifier", AdaBoostClassifier(
#             n_estimators=200,
#             learning_rate=0.1,
#             random_state=42
#         ))
#     ]
# )

# ada_model.fit(X_train, y_train)

# ada_probabilities = ada_model.predict_proba(X_test)[:, 1]

# ada_auc = roc_auc_score(
#     y_test,
#     ada_probabilities
# )

# print("AdaBoost ROC-AUC:", ada_auc)



"""
LOGISTIC REGRESSION FOR THIS CLASSIFIER
preprocessor = ColumnTransformer(
    transformers = [
        ("num" , StandardScaler() , numerical_columns),
        ("cat" , OneHotEncoder() , categorical_columns)
    ]
)
model = Pipeline(
    steps = [
        ("preprocessor" , preprocessor),
        ("classifier" , LogisticRegression(max_iter= 1000))
    ]
)

model.fit(X_train , y_train)

model.predict(X_test)

valid_probabilities = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(
    y_test,
    valid_probabilities
)


"""


"""
hISTBOOSTGRADIENTCLASSIFIER NATIVE EXAMPLE
native_model = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42,
    categorical_features="from_dtype"
)

native_model.fit(X_train , y_train)

native_model.predict(X_test)

native_prob = native_model.predict_proba(X_test)[: , 1]
"""