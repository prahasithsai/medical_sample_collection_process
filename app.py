from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def result():
    Patient_Age = request.form.get('Patient_Age')
    Patient_Gender = request.form.get('Patient_Gender')
    Name_Of_Test = request.form.get('Name_Of_Test')
    Sample = request.form.get('Sample')
    Way_Of_Storage_Of_Sample = request.form.get('Way_Of_Storage_Of_Sample')
    Test_Booking_Time_HH_MM = request.form.get('Test_Booking_Time_HH_MM')
    Scheduled_Sample_Collection_Time_HH_MM = request.form.get('Scheduled_Sample_Collection_Time_HH_MM')
    Agent_ID = request.form.get('Agent_ID')
    Traffic_Conditions = request.form.get('Traffic_Conditions')
    Time_Taken_To_Reach_Patient_MM = request.form.get('Time_Taken_To_Reach_Patient_MM')
    Time_For_Sample_Collection_MM = request.form.get('Time_For_Sample_Collection_MM')
    Time_Taken_To_Reach_Lab_MM = request.form.get('Time_Taken_To_Reach_Lab_MM')
    
    result = model.predict([[Patient_Age,Patient_Gender,Name_Of_Test,Sample,Way_Of_Storage_Of_Sample,Test_Booking_Time_HH_MM,Scheduled_Sample_Collection_Time_HH_MM,Agent_ID,Traffic_Conditions,Time_Taken_To_Reach_Patient_MM,Time_For_Sample_Collection_MM,Time_Taken_To_Reach_Lab_MM]])[0]

    if result==1:
        return render_template('index.html',label=1)
    else:
        return render_template('index.html',label=0)

if __name__=='__main__':
    app.run(debug=False)
