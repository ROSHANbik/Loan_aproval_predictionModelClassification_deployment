from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_building
  
def main():
    df = data_ingestion()
    X_train ,X_test ,y_train,y_test = data_preprocessing(df)
    classification_report,model   = model_building(X_train , X_test,y_train,y_test,pickle_path="model_best.pkl")

main()
