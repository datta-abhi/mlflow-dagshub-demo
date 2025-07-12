import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import mlflow.sklearn
import seaborn as sns

from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

# load the dataset
data = load_iris()
X = data.data
y = data.target

# splitting into train and test sets
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=28)

# define params for Random Forest
params = {'max_depth': 4,
        # 'n_estimators': 10
        }

# dagshub repo connection
import dagshub
dagshub.init(repo_owner='datta-abhi', repo_name='mlflow-dagshub-demo', mlflow=True)

# set mlflow uri for tracking
mlflow.set_tracking_uri("https://dagshub.com/datta-abhi/mlflow-dagshub-demo.mlflow")

# apply mlflow
mlflow.set_experiment('iris-dt')
with mlflow.start_run():
 
    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train,y_train)
    
    y_pred = dt.predict(X_test)
    
    accuracy = accuracy_score(y_pred,y_test)
    
    # logging metrics
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_params(params)
    
    # create and log artifacts
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True, fmt = 'd',cmap = 'Blues',xticklabels=data.target_names,yticklabels= data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # tracking code file as artifact
    mlflow.log_artifact(__file__)
    
    # tracking model
    mlflow.sklearn.log_model(sk_model=dt,name = 'decision tree')
    mlflow.set_tags({'author': 'Abhigyan',
                     'algo': 'DecisionTree'})
    
    print(f"Accuracy:{accuracy:.4f}")