#import scikit-learn labraries for the model_building
import pickle
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score , precision_score , f1_score, recall_score , classification_report

def model_building(X_train , X_test,y_train,y_test,pickle_path="model_best.pkl"):
    models=({
         "XGBRegressor":RandomForestClassifier(n_estimators=300 , 
                                    max_depth=6)
    })
    for model_name , model in models.items():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test,y_pred) 

    print(f"model name:{model_name} classification is :{report}")

    with open(pickle_path,"wb") as f:
       pickle.dump(model,f)
      
    return report  , model
