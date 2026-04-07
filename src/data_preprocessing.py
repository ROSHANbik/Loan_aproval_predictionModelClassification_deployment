from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , RobustScaler , LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score , precision_score , f1_score, recall_score , classification_report
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import OrderedDict

# data_preprocessing
  #1) drop the duplicates
  #2) split the X and y i.e input feature and target
  #3) split the train test split
  #4) use scaler ( pipeline)
  #5) use the SMOTE 
def data_preprocessing(df):
    df = df.drop_duplicates()
    X = df.drop(columns=['loan_id',' no_of_dependents',' education',' loan_status'],errors='ignore')
    y = df[' loan_status']

    X_train , X_test,y_train,y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=3)

    numerical_col = X.select_dtypes(exclude='object').columns
    categorical_col = X.select_dtypes(include='object').columns

    num_pipe = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",RobustScaler())
    ])

    cat_pipe =Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("Encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num",num_pipe,numerical_col),
        ("cat",cat_pipe,categorical_col)
    ])
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train ,X_test ,y_train,y_test 