# Project DS_69:
# Sample Collection dataset
# Import the required libraries
import pandas as pd
import numpy as np
data = pd.read_excel(r"C:\Users\RAJA\Downloads\lms\Project_DS\DS_Project_Team_69\samplecollection.xlsx") 
data.info     
data.dtypes # Return datatype of all columns
data.columns # Return all the column names in the dataset
# Data Preprocessing: 
# Label the scores for all the given columns in the dataset
data['Delay in Job'].value_counts()
data['Patient_Gender'] = data['Patient_Gender'].map({'Male':0,"Female":1})
data['Name_Of_Test'] = data['Name_Of_Test'].map({'Complete Urinalysis':1,'CBC':2,'Acute kidney profile':3,'Fasting blood sugar':4,'RTPCR':5,'TSH':6,'Vitamin D-25Hydroxy':7,'Lipid Profile':8,'H1N1':9,'HbA1c':10})
data['Sample'] = data['Sample'].map({'Blood':1,"Swab":2,'Urine':3,'blood':1})
data['Way_Of_Storage_Of_Sample'] = data['Way_Of_Storage_Of_Sample'].map({'Normal':1,'Advanced':2})

# Used to convert the difference in terms of years
data['Delay in Job'] = (((data.Sample_Collection_Date - data.Test_Booking_Date)/np.timedelta64(1, 'Y')))
data['Delay in Job'] = data['Delay in Job'].astype(int)
data['Delay in Job']

from datetime import datetime
data['Sample_Collection_Date'] = pd.to_datetime(data.Sample_Collection_Date)
data['Test_Booking_Date'] = pd.to_datetime(data.Test_Booking_Date)
data['Delay in Job'] = (data.Sample_Collection_Date - data.Test_Booking_Date)/np.timedelta64(1, 'D')).astype(int)

data['Delay in Job'] = data['Delay in Job'].str.split(' ').str[0]

data['Delay in Job'] = data['Delay in Job'].map(lambda x: x.lstrip('Year ').rstrip(''))
data['What year are you in now?'] = data['What year are you in now?'].astype(int)

data['Gender'] = data['Gender'].map({'Girl':0,"Boy":1})

data.rename(columns={'1. What did you eat for breakfast YESTERDAY?': 'breakfast YESTERDAY'}, inplace=True)
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].str.split(';').str[0]
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].str.split('and').str[0]
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].str.split('-').str[0]
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].map(lambda x: x.rstrip(' ')).str.lower()
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["crackerbread","pankaces","breadsticks","tostie s","brioch","waffles with chocolate spread","hot cross bun","toast","croissants with cheese","bacon muffins","bacon bap","pancake","pancakes","jam sandwich","croissants","sausage roll","wrap","sausage s","bread roll","waffles","brioche","a toastie","2 x pancakes","egg waffles","beans on toast","snacks","biscuits","homemade pancakes","brioche roll","freshly baked bread","mackrel on toast","bagel"],"Toast Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["boiled sausages","healthy cereal e.g. porridge, weetabix, readybrek, muesli, branflakes, cornflakes","fruit;toast;yoghurt","boiled eggs","shreddies","marshmallow mates","egg","eggs","sausage & egg s","cheese","sardines"],"Healthy Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["fruit","fruit;toast;yoghurt","yoghurt","apple, pear, grapes, yogurt, brioche"],"Fruit Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["danish","waffle","pain au choclat","pan au chocolate","pain au chocolat","sugary cereal e.g. cocopops, frosties, sugar puffs, chocolate cereals","chocolate crepe","chocolate finger","pain au chocolate","crumpets with chocolate spread","croissants","jam s","pancakes with chocolate spread","nutella","pop tarts","chocolate spread on bread","pano chocolate"],"Sugary Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["nothing"],"No Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["egg on toast","waffles mum made them","pancakes & strawberries","chesse","healthy shake","tea","cheesey beans","smoothy"],"Other Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].replace(dict.fromkeys(["crumpet","crumpets","cooked breakfast","cheese omlet","crumpets with marmite","homemade pancakes with fruit","homemade pancakes","belgian pancakes (it was my mums birthday)"],"Cooked Food"))
data['breakfast YESTERDAY'] = data['breakfast YESTERDAY'].map({'No Food':0,"Healthy Food":1,"Toast Food":2,"Sugary Food":3,"Fruit Food":4,"Cooked Food":5,"Other Food":6})

data['2. Did you eat any fruit and vegetables YESTERDAY? '] = data['2. Did you eat any fruit and vegetables YESTERDAY? '].map({'2 Or More Fruit and Veg':2,"1 Piece":1,"No":0})

data4 = data.iloc[:,12:14]
data4['4. What time did you fall asleep YESTERDAY (to the nearest half hour)?'] = data4['4. What time did you fall asleep YESTERDAY (to the nearest half hour)?'].map({'10:00pm':2,"9:30pm":2.5,"9:00pm":3,'10:30pm':1.5,'8:30pm':3.5,
      '11:00pm':1,'11:30pm':0.5,'8:00pm':4,'12:00am':0,'12:30am':0.5,'7:30pm':4.5,'2:00am':2,'7:00pm':5,'1:00am':1,'1:30am':1.5})
data4['5. What time did you wake up TODAY (to the nearest half hour)?'] = data4['5. What time did you wake up TODAY (to the nearest half hour)?'].map({'8:00am':8,'7:30am':7,'8:30am':8.5,'9:00am':9,'7:00am':7,'9:30am':9.5,'6:30am':6.5,
      '10:00am':10,'6:00am':6,'10:30am':10.5,'11:30am':11.5,'11:00am':11,'5:30am':5.5,'5:00am':5})
data4['sleeping hours'] = data4['4. What time did you fall asleep YESTERDAY (to the nearest half hour)?'] + data4['5. What time did you wake up TODAY (to the nearest half hour)?']

data5 = data.iloc[:,14:21]
data5.columns = ["activities","you watch TV","feel tired","school work","drink","sweets","Chinese takeaway"]
data5["activities"] = data5["activities"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})
data5["you watch TV"] = data5["you watch TV"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})
data5["feel tired"] = data5["feel tired"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})
data5["school work"] = data5["school work"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})
data5["drink"] = data5["drink"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})
data5["sweets"] = data5["sweets"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})
data5["Chinese takeaway"] = data5["Chinese takeaway"].map({'7 days':7,"3-4 days":4,"5-6 days":6,'1-2 days':2,'0 days':0})

data["14. From your house, can you easily walk to a park (for example a field or grassy area)?"] = data["14. From your house, can you easily walk to a park (for example a field or grassy area)?"].map({'Yes':1,"No":0})
data["15. From your house, can you easily walk to somewhere you can play?"] = data["15. From your house, can you easily walk to somewhere you can play?"].map({'Yes':1,"No":0})
data["16. Do you have a garden?"] = data["16. Do you have a garden?"].map({'Yes':1,"No":0})
data['17. How often do you go out to play outside?'] = data['17. How often do you go out to play outside?'].map({'Most days':4,"I don't play":3,"Hardly ever":2,'A few days each week':1})
data['17. How often do you go out to play outside?'] = data['17. How often do you go out to play outside?'].fillna(0)
data['18. Do you have enough time for play?'] = data['18. Do you have enough time for play?'].map({'Yes, I have loads':3,"Yes, it's just about enough":4,"No, I would like to have a bit more":2,'No, I need a lot more':1})
data['18. Do you have enough time for play?'] = data['18. Do you have enough time for play?'].fillna(0)
data['19. What type of places do you play in?'] = data['19. What type of places do you play in?'].str.split(';').str[0]
data['19. What type of places do you play in?'] = data['19. What type of places do you play in?'].map({"In my house":3,"In my garden":2,"In a place with bushes, trees and flowers":1,"On the bike or skate park":5,"On a local grassy area":4,"In the street":6,"Out the front of my house":6,"In the woods near my house":6})
data['20. Can you play in all the places you would like to?'] = data['20. Can you play in all the places you would like to?'].map({'I can play in some of the places I would like to':3,"I can play in all the places I would like to":4,"I can only play in a few places I would like to":2,'I can hardly play in any of the places I would like to':1})
data['21. Do you have somewhere at home where you have space to relax?'] = data['21. Do you have somewhere at home where you have space to relax?'].map({'Yes':2,"Sometimes but not all the time":1,"No":0})

data3 = data.iloc[:,30:34]
data3.rename(columns={"22. Tell us if you agree or disagree with the following: [I am doing well with my school work]": 'well with my school work',
                      "22. Tell us if you agree or disagree with the following: [I feel part of my school community]": 'feel part of my school community',
                      "22. Tell us if you agree or disagree with the following: [I have lots of choice over things that are important to me]" : 'important to me',
                      "22. Tell us if you agree or disagree with the following: [There are lots of things I'm good at]": 'things Iam good at'}, inplace=True)
data3['well with my school work'] = data3['well with my school work'].map({'Agree':2,'Strongly agree':1,"Don't agree or disagree":5,'Disagree':3,'Strongly disagree':4})                                                                                              
data3['feel part of my school community'] = data3['feel part of my school community'].map({'Agree':2,'Strongly agree':1,"Don't agree or disagree":5,'Disagree':3,'Strongly disagree':4})
data3['important to me'] = data3['important to me'].map({'Agree':2,'Strongly agree':1,"Don't agree or disagree":5,'Disagree':3,'Strongly disagree':4})
data3['things Iam good at'] = data3['things Iam good at'].map({'Agree':2,'Strongly agree':1,"Don't agree or disagree":5,'Disagree':3,'Strongly disagree':4})

data1 = data.iloc[:,40:56]
data1.columns =['I feel lonely','I cry a lot','I am unhappy','I feel nobody likes me','I worry a lot','I have problems sleeping','I wake up in the night',
                'I am shy','I feel scared','I worry when I am at school','I get very angry','I lose my temper','I hit out when I am angry','I do things to hurt people',
                'I am calm','I break things on purpose']
data1['I am calm'] = data1['I am calm'].map({'Never':2,'Sometimes':1,'Always':0})
data2 = data1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15]]
data2.replace({'Never':0,'Sometimes':1,'Always':2}, inplace=True)
data1 = pd.concat([data2,data1['I am calm']],axis='columns')
data1["emmotion"] = data1["I feel lonely"] + data1["I cry a lot"] + data1["I am unhappy"] + data1["I feel nobody likes me"] + data1["I worry a lot"] + data1["I have problems sleeping"] + data1["I wake up in the night"] + data1["I am shy"] + data1["I feel scared"] + data1["I worry when I am at school"]
data1["behaviour"] = data1['I get very angry'] + data1["I lose my temper"] + data1["I hit out when I am angry"] + data1["I do things to hurt people"] + data1["I am calm"] + data1["I break things on purpose"]
data1["emmotion_scale"]="<=5"
data1.loc[data1["emmotion"]<=9,"emmotion_scale"]="expected difficulties"
data1.loc[data1["emmotion"]==10,"emmotion_scale"]="borderline difficulties"
data1.loc[data1["emmotion"]==11,"emmotion_scale"]="borderline difficulties"
data1.loc[data1["emmotion"]>=12,"emmotion_scale"]="clinically significant difficulties"
data1["behaviour_scale"]="<=5"
data1.loc[data1["emmotion"]<=5,"behaviour_scale"]="expected difficulties"
data1.loc[data1["emmotion"]==6,"behaviour_scale"]="borderline difficulties"
data1.loc[data1["emmotion"]>=7,"behaviour_scale"]="clinically significant difficulties"
mmf = data1.iloc[:,18:]
data.rename(columns={"25. Are you able to keep in touch with your family that you don't live with? (grand parents, Uncle, Aunt, Cousins, etc)": 'keep in touch with family',"26. Are you able to keep in touch with your friends?":'keep in touch with your friends'}, inplace=True)
data['keep in touch with family'] = data['keep in touch with family'].map({'Yes':1,'No':0})
data['keep in touch with your friends'] = data['keep in touch with your friends'].map({'Yes':1,'No':0})
data.rename(columns={'27. If yes, how are you keeping in touch (tick all you use)?': 'keep in touch by other mode'}, inplace=True)
data['keep in touch by other mode'] = data['keep in touch by other mode'].map({'By phone (texting, calling or video calling)':2,'By phone (texting, calling or video calling);On games consoles':2,
    'I live near them so I can see them (at a social distance);By phone (texting, calling or video calling)':1,
    'By phone (texting, calling or video calling);On social media;On games consoles':2,'By phone (texting, calling or video calling);On social media':2,
    'I live near them so I can see them (at a social distance);By phone (texting, calling or video calling);On social media;On games consoles':1,
    'I live near them so I can see them (at a social distance);By phone (texting, calling or video calling);On games consoles':1,
    'I live near them so I can see them (at a social distance);By phone (texting, calling or video calling);On social media':1,'On games consoles':3,
    'On social media':3,'I live near them so I can see them (at a social distance)':1,'I live near them so I can see them (at a social distance);On games consoles':1,
    'On social media;On games consoles':3,'I live near them so I can see them (at a social distance);On social media':1})                                                                                              
# Drop the irrevelant columns
data.drop(data.columns[[0,6,7,8,12,13,14,15,16,17,18,19,20,30,31,32,33,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]], axis = 1, inplace = True)
# a) Prepare the final data with output as emmotional column
model_emt = pd.concat([data,data4['sleeping hours'],data5,data3,mmf['emmotion_scale']],axis='columns')
model_emt.columns = ['Are you still going to school','Any children with you','No of people with you','Age','Gender',"Yesterday's breakfast",'Did you ate fruits,veggies','No of times brushing teeth',
                     "Is it safe playing in your area",'Can you walk to park','Can you walk where you play','Do you have garden','Do you go to play outside','Do you have time to play','Where you will play',
                     'Can you play in all places','Do you have place to relax','Your health','Your school','Your family','Your friends','Your appearance','Your life','keep in touch with family', 'keep in touch with your friends',
                     'keep in touch by other mode','Sleeping hours','Activities','Do you watch TV','Do you get tired','Do you finish you home work','Do you have drinks','Do you have sweets','Do you have chineese takeaway','Are you doing well your homework',
                     'Are you part of school community','Important things',"I'm good at",'emmotinal_scale']
model_emt['emmotinal_scale'] = model_emt['emmotinal_scale'].map({'expected difficulties':1,"borderline difficulties":2,"clinically significant difficulties":3})
 
# Handling Missing values:
model_emt.isna().sum() # NaN values are found
from sklearn.impute import SimpleImputer
s_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent',verbose=0)
s_imputer=s_imputer.fit(model_emt.iloc[:,:])
model_emt.iloc[:,:]=s_imputer.transform(model_emt.iloc[:,:])
model_emt.isna().sum().sum()

# Feature Selection: Using "K-Best & Chi2" Algorithm 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_emt = model_emt.iloc[:,:-1];Y_emt = model_emt.iloc[:,-1]
best_ftr = SelectKBest(score_func=chi2,k='all')
allftr_scores = best_ftr.fit(X_emt,Y_emt)
allftr_scores = pd.DataFrame(allftr_scores.scores_,columns=['Score'])
allftr_names = pd.DataFrame(X_emt.columns,columns=['Features'])
bst_ftrs = pd.concat([allftr_names,allftr_scores],axis=1).sort_values(by='Score',ascending = False)
bst_ftrs
model_emtf =model_emt.drop(['Do you have garden','Age',"Yesterday's breakfast",'Do you have time to play','Where you will play','Are you still going to school',
                           'Can you walk where you play','No of people with you','Can you walk to park','Sleeping hours','keep in touch by other mode',
                           'Do you have sweets','Any children with you','Gender','Do you have place to relax','Do you watch TV','Do you have chineese takeaway',
                           'Can you play in all places'],axis=1)
top_bst_ftrs = bst_ftrs.nlargest(15,'Score')

# Checking for correlation between input features
# import seaborn as sns
# sns.pairplot(model_emtf.iloc[:,:])
# find and remove correlated features
corr=abs(model_emtf.iloc[:,:-1].corr())
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
threshold = 0.8 # for absolute values of r, 0-0.19 is 'very weak', 0.2-0.39 is 'weak', 0.40-0.59 is 'moderate', 0.6-0.79 'stronger correlation' and 0.8-1 is 'strongest correlation'
correlation(model_emtf.iloc[:,:-1],threshold) # No values are found above threshold limit

# Model Building:
# (1) Logistic Regression with StratifiedKFold:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
classifier = LogisticRegression(multi_class = "ovr", solver = "newton-cg")
from sklearn.model_selection import StratifiedKFold
accuracy=[]
skf = StratifiedKFold(n_splits=30,random_state=None)
X_emtf = model_emtf.iloc[:,:-1];Y_emt
skf.get_n_splits(X_emtf,Y_emt)
# x is the feature set and y is the target
for train_index,test_index in skf.split(X_emtf,Y_emt):
    print("Train:",train_index,"Validation:",test_index)
    x_train1,x_test1 = X_emtf.iloc[train_index],X_emtf.iloc[test_index]
    y_train1,y_test1 = Y_emt.iloc[train_index],Y_emt.iloc[test_index]
    classifier.fit(x_train1,y_train1)
    prediction = classifier.predict(x_test1)
    score = accuracy_score(prediction,y_test1)
    accuracy.append(score)
np.array(accuracy).mean()
# Evaluation on Test Data
print(confusion_matrix(y_test1,prediction))
print(classification_report(y_test1,prediction))

# (2) Decision Tree with (XGBoosting + RandomizedSearchCV):
# Hyper Parameter Optimization
from sklearn import tree
import xgboost as xgb
clftree = tree.DecisionTreeClassifier()
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(X_emtf, Y_emt, test_size = 0.2,random_state=42)
params ={"learning_rate"    :[0.05,0.1,0.15,0.2,0.25,0.3],
        "max_depth"         :[3,4,5,6,7,8,9,10,12,15],
        "min_child_weight"  :[1,3,5,7],
        "gamma"             :[0,0.1,0.2,0.3,0.4],
        "colsample_bytree"  :[0.3,0.4,0.5,0.7]} 
classifier1 = xgb.XGBClassifier(n_jobs = 1,objective = 'multi:softmax',silent=1,tree_method='approx')
random = RandomizedSearchCV(classifier1,param_distributions=params,n_iter=5,n_jobs = -1, cv = 10,verbose=3)
random.fit(x_train2,y_train2)
random.best_params_
xg_clf_random = random.best_estimator_
# Evaluation on Test Data
print(confusion_matrix(y_test2, xg_clf_random.predict(x_test2)))
print(classification_report(y_test2, xg_clf_random.predict(x_test2)))


# (3) RandomForest Classifier with StratifiedKFold:
from sklearn.ensemble import RandomForestClassifier
x_train3, x_test3, y_train3, y_test3 = train_test_split(X_emtf,Y_emt, test_size = 0.2,random_state=42)
X_emtf = model_emtf.iloc[:,:-1]
Y_emt = pd.DataFrame(model_emtf.iloc[:,-1])
classifier2 = RandomForestClassifier(max_depth=2, random_state=42)
from sklearn.model_selection import StratifiedKFold
accuracy1=[]
skf = StratifiedKFold(n_splits=30,random_state=None)
skf.get_n_splits(X_emtf,Y_emt)
# x is the feature set and y is the target
for train_index,test_index in skf.split(X_emtf,Y_emt):
    print("Train:",train_index,"Validation:",test_index)
    x_train3,x_test3 = X_emtf.iloc[train_index],X_emtf.iloc[test_index]
    y_train3,y_test3 = Y_emt.iloc[train_index],Y_emt.iloc[test_index]
    classifier2.fit(x_train3,y_train3)
    prediction1 = classifier2.predict(x_test3)
    score1 = accuracy_score(prediction1,y_test3)
    accuracy1.append(score1)
np.array(accuracy1).mean()
# Evaluation on Test Data
print(confusion_matrix(y_test3,prediction1))
print(classification_report(y_test3,prediction1))

# creating the bar plot for Accuracy scores
import matplotlib.pyplot as plt
dat = {'Logistic Regression':0.91,'Decision Tree':0.89,'RandomForest':0.91}
modelname = list(dat.keys())
accuracyvalue = list(dat.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(modelname,accuracyvalue,color ='maroon',width = 0.2)
plt.xlabel("modelname")
plt.ylabel("accuracyvalue")
plt.title("Accuracy scores of different models")
plt.show()
# creating the bar plot for f1-scores
dat1 = {'Logistic Regression':0.86,'Decision Tree':0.85,'RandomForest':0.86}
modelname = list(dat.keys())
f1score = list(dat1.values())
fig = plt.figure(figsize = (10,5))
plt.bar(modelname,f1score,color ='maroon',width = 0.2)
plt.xlabel("modelname")
plt.ylabel("f1-score")
plt.title("f1-scores of different models")
plt.show()



















































    


    
    
    
    
    
    
    

