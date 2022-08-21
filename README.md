# medical_sample_collection_process
* Project Title: "medical_sample_collection_process"
* Project Management Methodology used - CRISP ML (Q)
* Scope of the Project: To Determine whether the samples of patient collected by the agent for test can deliver to the lab within time or not.  
***********************************************************************************************************************************************************************
* a) Business Understanding & Data Understanding:
* Business Objective: To determine whether the agent can deliver the samples of the patient to the lab within time or not.
* Business constraints: Choosing the top most relevant data.
***********************************************************************************************************************************************************************
* b) Data dictionary: (Secondary Data Source)
*	      **Column**           		**Dtype**     		**Relevance**         
	  1   Patient ID	    			Discrete-Nominal	 Irrelevant
	  2   Patient Age				Numeric-Ratio		 Relevant 
	  3   Patient Gender	    		Discrete-Binary	         Relevant  		 
	  4   Test Name	  	    		Discrete-Nominal   	 Relevant   
	  5   Sample		 	    	Discrete-Nominal	 Relevant
	  6   Way of Storage of Sample		Discrete-Nominal	 Relevant
	  7   Test Booking Date		    	Datetime64	  	 Irrelevant
	  8   Test Booking Time,HH MM		Numeric-Ratio		 Relevant
	  9   Sample Collection Date		Datetime64	 	 Irrelevant
	  10  Scheduled Sample Collection Time,HH MM Numeric-Ratio    	 Relevant
	  11  Cut-off Schedule			Discrete-Ordinal	 Irrelevant
	  12  Cut-off time,HH MM			Numeric	Ratio	   	 Irrelevant
	  13  Agent ID				Discrete-Ordinal 	 Relevant
	  14  Traffic Conditions			Discrete-Ordinal 	 Relevant
	  15  Agent Location,KM			Numeric-Ratio	 	 Irrelevant
	  16  Time Taken To Reach Patient,MM	Numeric-Ratio	 	 Relevant
	  17  Time For Sample Collection,MM	Numeric-Ratio	 	 Relevant
	  18  Lab Location,KM			Numeric-Ratio		 Irrelevant
	  19  Time Taken To Reach Lab,MM		Numeric-Ratio	         Relevant
	  20  Mode of Transport			Discrete-Ordinal	 Irrelevant
	  21  Reached on Time			Discrete-Binary		 Relevant
***********************************************************************************************************************************************************************
* c) Data Preprocessing:
*	Drop the attributes which are irrelevant from the dataset.
*	Customize the column names
*	Perform Text Mining to get relevant information for corresponding columns
*	Map the labels as per the their scores in the given columns of the dataset
*	Perform label encoding for the categorical features where ever it is applicable
*	Perform all the required operations for the given columns as mentioned in the Sheet2 of the given dataset 
*	Check for NaN values, if any present perform imputation. 
*	Check for zero variance & near zero variance features and drop it.
***********************************************************************************************************************************************************************
* d) Exploratory Data Analysis (EDA):
*	Perform pair plot for the finalized dataset & identify for co linearity between predictors.
*	Used Bar plot for plotting accuracy scores & f1-scores of different models.
*	Build correlation coefficient matrix for dataset for checking of co linearity problem.  
***********************************************************************************************************************************************************************
* e) Feature Engineering:
*	Using "K-Best & Chi2" Algorithm - Gives most significant features with respect to target variable
*	Identify the features which are most relevant for model building based on their scores & drop the remaining features from the dataset
*	Check the correlation coefficient,(|r|) between input features, if any present (|r| > 0.85)  drop those features which has least impact on target variable
***********************************************************************************************************************************************************************
* f) Model Building: 
*	Classification algorithm used is:Decision Tree Classifier
*	Import the required libraries for model building
*	Split the entire dataset into train & test (with test size = 0.3) 
*	Use the Grid Search CV technique to choose the best hyper parameters in Decision Tree model.
*	Run the model using test data & compare the prediction values with actual values of test data
*	Check for accuracy scores & f1-scores for goodness of the different models.
*	Store the file into pickle format viz., while using it in creating flask API, HTML, CSS.
*	Deploy the model & evaluate it’s performance. 
*	Attachment of python code for model building is available with documentation
***********************************************************************************************************************************************************************
* g) Results: 
*	Evaluation Metrics:
*   		Model							Accuracy			f1-score
									Train		Test		Train		Test
		Decision Tree Classifier with (Grid Search CV)	1.0		0.9673		1.0		0.97
*		Confusion Matrix:
		Train Dataset				Predicted Values		Test Dataset		Predicted Values
		Actual Values				No	Yes			Actual Values		No		Yes
		No					139	0			No			139		0
		Yes					0	574			Yes			0		574
*	Highly Significant features for predicting is obtained: 
	1)	After performing Correlation coefficient between input features
	2)	Using "K-Best & Chi2" Algorithm
*	      **Features**           		**Score**     		         
	  1   Time_Taken_To_Reach_Lab_MM 		50047.80
	  2   Time_Taken_To_Reach_Patient_MM	2887.70
	  3   Time_For_Sample_Collection_MM	381.19
	  4   Scheduled_Sample_Collection_Time_HH_MM10.99
	  5   Traffic_Conditions			7.46
	  6   Agent_ID				5.04
	  7   Test_Name				3.25
	  8   Patient_Age				1.65
	  9   Sample				0.99
	 10   Way_Of_Storage_Of_Sample	 	0.33
	 11   Test_Booking_Time_HH_MM		0.13
	 12   Patient_Gender			0.01
* Summary: The above features should be given most importance in order to reach samples of patient within time.	 
***********************************************************************************************************************************************************************
* Attachments/Links
* Note: please refer project documentation in 'CRISP ML Q' file & coding in 'model file'
* Link: https://github.com/prahasithsai/medical_sample_collection_process
*	Deployment Link: https://medicalsamplecollection.herokuapp.com/
* Used Libraries: pandas, numpy,matplotlib,seaborn,sklearn,flask.
