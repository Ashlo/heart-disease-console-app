import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('heart.csv')

X = df.iloc[:,2:13]
y = df.iloc[:,13:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

print("Program to Classify whether you have a heart disease or not")


cp = int(input("The Chest Pain Experience : 1. Typical angina 2:Atypical Angina 3. non-Angical pain 4 . asymptotic"))
trestbps = int(input("Resting Blood Pressure, Enter between 94 to 200 "))
chol = int(input("Cholestrol measure in mg/dl  Enter between 126 to 564 " ))
fbs = int(input('fasting blood sugar if > 120 mg/dl 1 = true 0 = False'))
restecg = int(input("Resting Electrocardiographic measurement 0 = normal , 1 = having stt wave abnormalit 2 = showing probable or definite left ventricular"))
thalach = int(input('maximum heart rate achieved Enter Between 71 to 282'))
exang = int(input("Exercised induced engina 1 = yes 2 = no"))
oldpeak = int(input("St depression induced by Exercise Enter between 0 to 6 "))
slope = int(input("1 = upslopiing 2 = flat 3 = downslopping"))
ca = int(input('the number of major vessels (0-3)'))
thal = int(input("blood Disorder 1 = normal 2 = fixed defest 3 = reversable defect"))

testcase = [cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

testcase = pd.DataFrame(testcase)

testcase = pd.DataFrame.transpose(testcase)











def treeClass():
    tree = DecisionTreeClassifier()
    tree.fit(X_train,y_train)
    prediction = tree.predict(testcase)
   # print(f"Confusion Matrix {confusion_matrix(y_test,prediction)}")
   #print(f"Accuracy Score {accuracy_score(y_test,prediction) * 100}")
   #print(f"Classification report {classification_report(y_test,prediction)}")
    print(prediction)

def logistic():
    regressor = LogisticRegression()
    regressor.fit(X_train,y_train)
    prediction = regressor.predict(testcase)
   # print(f"Confusion Matrix {confusion_matrix(y_test,prediction)}")
   # print(f"Accuracy Score {(accuracy_score(y_test,prediction)) * 100}")
   # print(f"Classification report {classification_report(y_test,prediction)}")
    print(prediction)
def linear():
    regressor =  LinearRegression()
    regressor.fit(X_train,y_train)
    prediction = regressor.predict(testcase)
   # print(f"Accuracy : {(1 - mean_squared_error(y_test,prediction)) * 100} % ")
    print(prediction)


print("1; Decision tree Classifier \n 2: Logistic Regression 3: Linear Regression")
choice =  int(input("Enter your Choice"))
if choice == 1:
    treeClass()
elif choice ==2:
    logistic()
elif choice ==3:
    linear()

