import json
import  joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    iris = load_iris()
    X , y= iris.data, iris.target
    print(X)
    print(y)
    print(iris.target_names)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)

    os.mkdir("artifacts")
    model_path = os.path.join("artifacts","model.pkl")
    joblib.dump(model,model_path)

    acc = model.score(X_test, y_test)
    metrics = {"accuracy":float(acc)}

    with open(os.path.join("artifacts","metrics.json"),"w") as f:
        json.dump(metrics,f)
    print(acc)


if __name__ == '__main__':
    main()