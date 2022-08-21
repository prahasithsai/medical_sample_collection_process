# Import the required libraries
import pandas as pd
import numpy as np
import pickle
data = pd.read_excel(r"C:\Users\Sai\Desktop\P2\flask\samplecollection.xlsx")

# Data Preprocessing: 
# Labelleing for categorical columns 
data['Patient_Gender'] = data['Patient_Gender'].map({'Male':1,'Female':2})
data['Name_Of_Test'] = data['Name_Of_Test'].map({'Complete Urinalysis':1,'CBC':2,'Acute kidney profile':3,'Fasting blood sugar':4,'RTPCR':5,'TSH':6,'Vitamin D-25Hydroxy':7,'Lipid Profile':8,'H1N1':9,'HbA1c':10})
data['Sample'] = data['Sample'].map({'Blood':1,'Swab':2,'Urine':3,'blood':4})
data['Way_Of_Storage_Of_Sample'] = data['Way_Of_Storage_Of_Sample'].map({'Normal':1,'Advanced':2})
data['Cut-off Schedule'] = data['Cut-off Schedule'].map({'Daily':1,'Sample by 5pm':2})
data['Traffic_Conditions'] = data['Traffic_Conditions'].map({'Medium Traffic':1,'Low Traffic':2,'High Traffic':3})
data['Mode_Of_Transport'] = data['Mode_Of_Transport'].map({'BIKE':1}) # 'Mode of Transport' has Zero Variance feature
data['Reached_On_Time'] = data['Reached_On_Time'].map({'Y':1,'N':0}) # 'Mode of Transport' has Zero Variance feature
data.columns
# Drop the irrevelant columns - Based on correlation coefficient & k-best algorithm
data.drop(data.columns[[0,6,8,10,11,14,17,19]], axis = 1, inplace = True)
data['Patient_Gender'].value_counts()
# Taking predictors and target
X = np.array(data.iloc[:,:-1])
y = np.array(data.iloc[:,-1])

# Train-Test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Model Building: Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier as DT
model = DT()
model.fit(X_train, y_train)

# Evaluation on Test Data
from sklearn.metrics import classification_report, confusion_matrix
pred_test = model.predict(X_test)
clr_test = classification_report(y_test,pred_test)
cnf_test = confusion_matrix(y_test,pred_test)

# Evaluation on Train Data
pred_train = model.predict(X_train)
clr_train = classification_report(y_train, pred_train)
cnf_train = confusion_matrix(y_train, pred_train)

# Saving the model
pickle.dump(model, open('model.pkl', 'wb'))


