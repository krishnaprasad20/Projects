import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment ("patient_RandomForest_classifier")

df = pd.read_csv('Patients Data ( Used for Heart Disease Prediction ).csv')

df.drop(columns=['PatientID'], inplace=True)
df.drop(columns=['HeightInMeters', 'WeightInKilograms'], inplace=True)
df.drop_duplicates(inplace=True)
age_map = {
    'Age 18 to 24': 21,
    'Age 25 to 29': 27,
    'Age 30 to 34': 32,
    'Age 35 to 39': 37,
    'Age 40 to 44': 42,
    'Age 45 to 49': 47,
    'Age 50 to 54': 52,
    'Age 55 to 59': 57,
    'Age 60 to 64': 62,
    'Age 65 to 69': 67,
    'Age 70 to 74': 72,
    'Age 75 to 79': 77,
    'Age 80 or older': 82
}

df['Age'] = df['AgeCategory'].map(age_map)
df.drop(columns=['AgeCategory'], inplace=True)
health_map = {
    'Excellent': 5,
    'Very good': 4,
    'Good': 3,
    'Fair': 2,
    'Poor': 1
}

df['HealthScore'] = df['GeneralHealth'].map(health_map)
df.drop(columns=['GeneralHealth'], inplace=True)

smoker_map = {
    'Never smoked': 0,
    'Former smoker': 1,
    'Current smoker': 2
}

df['SmokerRisk'] = df['SmokerStatus'].map(smoker_map)
df.drop(columns=['SmokerStatus'], inplace=True)

def bmi_category(bmi):
    if bmi < 18.5:
        return 0   # Underweight
    elif bmi < 25:
        return 1   # Normal
    elif bmi < 30:
        return 2   # Overweight
    else:
        return 3   # Obese

df['BMI_Category'] = df['BMI'].apply(bmi_category)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['State']=le.fit_transform(df['State'])

df['ECigaretteUsage']=df['ECigaretteUsage'].replace(['Never used e-cigarettes in my entire life', 'Not at all (right now)',
 'Use them some days' ,'Use them every day'],[1,2,3,4])

df['RaceEthnicityCategory']=df['RaceEthnicityCategory'].replace(['White only, Non-Hispanic', 'Black only, Non-Hispanic',
 'Other race only, Non-Hispanic', 'Multiracial, Non-Hispanic' ,'Hispanic'],[1,2,3,4,5])

df['HadDiabetes']=df['HadDiabetes'].replace(['Yes', 'No', 'No, pre-diabetes or borderline diabetes',
 'Yes, but only during pregnancy (female)'],[1,2,3,4])

df['TetanusLast10Tdap']=df['TetanusLast10Tdap'].replace(['No, did not receive any tetanus shot in the past 10 years',
 'Yes, received Tdap', 'Yes, received tetanus shot but not sure what type',
 'Yes, received tetanus shot, but not Tdap'],[1,2,3,4])

dummies=pd.get_dummies(df["Sex"]).astype(int)
dummies

df = pd.concat([df,dummies],axis="columns")
df.drop(["Sex"],axis="columns",inplace=True)

df['SmokerRisk'].hist()
df['SmokerRisk'].skew()

df['SmokerRisk'].fillna(df['SmokerRisk'].mode()[0],inplace=True)

from imblearn.over_sampling import SMOTE

X = df.drop('HadHeartAttack',axis=1)
y = df['HadHeartAttack']

smote=SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X,y)

df=pd.concat([X_resampled,y_resampled],axis=1)

df.drop(["Male","Female"],axis=1,inplace=True)

df.drop(["BMI_Category",'FluVaxLast12','DifficultyDressingBathing','DifficultyErrands','DifficultyWalking',
         'DifficultyConcentrating','BlindOrVisionDifficulty','DeafOrHardOfHearing','HadArthritis','HadKidneyDisease',
         'HadCOPD','HadSkinCancer','HadStroke','BMI','State'],axis=1,inplace=True)



X = df.drop('HadHeartAttack',axis = 1)
y = df['HadHeartAttack']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

with mlflow.start_run(run_name="RandomForestClassifier_Run_6"):
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='log2',
        random_state=42

    )

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model","RandomForestClassifier")
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("max_depth",12)
    mlflow.log_param("min_samples_split",15)
    mlflow.log_param("min_samples_leaf",8)
    mlflow.log_param("max_features",'log2')
    mlflow.log_param("random_state",42)
    mlflow.log_param("test_size",0.25)
    
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model,"Random_Forest_model")

    print("Accuracy : ", acc)