import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle



data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
data.head(2)
df = data.copy()

# # Transform the dataset to a Machine-Readeable Language

# df = data.copy()

# from sklearn.preprocessing import LabelEncoder
# def transformer(datadf):
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     scaler = StandardScaler()
#     encoder = LabelEncoder()

#     for i in datadf.columns:
#         if datadf[i].dtypes != 'O':
#             datadf[i] = scaler.fit_transform(datadf[[i]])
#         else:
#             datadf[i] = encoder.fit_transform(datadf[i])
#     return datadf

# y = df.DEATH_EVENT
# x = df.drop(['DEATH_EVENT'], axis = 1)

# sel_cols = ['time', 'platelets', 'creatinine_phosphokinase', 'serum_creatinine', 'ejection_fraction', 'age', 'serum_sodium']
# new_df = df[sel_cols]
# new_df = pd.concat([new_df, df['DEATH_EVENT']], axis = 1)
# new_df.head()



# # - split into train and test
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.metrics import classification_report

# y = df.DEATH_EVENT
# x = df.drop('DEATH_EVENT', axis =1)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 79, stratify = y)
# print(f'x_train: {x_train.shape}')
# print(f'x_test: {x_test.shape}')
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))

# # Create a Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)

# # Train the classifier
# model.fit(x_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(x_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)

# pickle.dump(model, open('Model.pkl', "wb"))


import streamlit as st
import pickle
model = pickle.load(open('Model.pkl', "rb"))

st.sidebar.image('pngwing.com (12).png', width = 300,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Heart Failure Prediction</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>Developing an Accurate Heart failure Prediction Model Using Machine Learning to Enhance Early Detection and Improve Patient Outcomes.</h6>",unsafe_allow_html=True)
    st.image('main.png',  width = 650)

    # Background story
    st.markdown("<h3 style = 'margin: -15px; color: #2B2A4C; text-align: left; font-family:montserrat'>Background to the story</h3>",unsafe_allow_html=True)
    st.markdown("<p>he underlying mechanisms vary depending on the disease. It is estimated that dietary risk factors are associated with 53 percent of CVD deaths. Coronary artery disease, stroke, and peripheral artery disease involve atherosclerosis. This may be caused by high blood pressure, smoking, diabetes mellitus, lack of exercise, obesity, high blood cholesterol, poor diet, excessive alcohol consumption, and poor sleep, among other things.</p>", unsafe_allow_html = True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>IThe age of the patient in years</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Anaemia</h3>", unsafe_allow_html=True)
    st.markdown("<p>A binary indicator of whether the patient has a lower than normal red blood cell count (0 = no, 1 = yes)</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>creatinine phosphokinase (CPK)</h3>", unsafe_allow_html=True)
    st.markdown("<p>The level of the enzyme creatinine phosphokinase (CPK) in the patient’s blood in mcg/L. </p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Diabetes</h3>", unsafe_allow_html=True)
    st.markdown("<p>A binary indicator of whether the patient has diabetes (0 = no, 1 = yes)</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>ejection_fraction</h3>", unsafe_allow_html=True)
    st.markdown("<p>The percentage of blood leaving the heart at each contraction. A lower ejection fraction means a weaker heart pump.</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>serum_sodium</h3>", unsafe_allow_html=True)
    st.markdown("<p>The level of sodium in the patient’s blood in mEq/L.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>ejection_fraction</h3>", unsafe_allow_html=True)
    st.markdown("<p>The percentage of blood leaving the heart at each contraction. A lower ejection fraction means a weaker heart pump.</p>", unsafe_allow_html=True)

    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by Tivanny africa </p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(df.head())
    # st.sidebar.image('pngwing.com (13).png', width = 300,  caption = 'customer and deliver agent info')


if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()


if selected_page == "Modeling":
    st.sidebar.markdown("Add your modeling content here")
    time = st.sidebar.number_input("time", 0,1000) 
    platelets = st.sidebar.number_input("platelets", 0,1000)
    creatinine_phosphokinase = st.sidebar.number_input("creatinine_phosphokinase", 0,1000) 
    serum_creatinine = st.sidebar.number_input("serum_creatinine", 0,1000) 
    ejection_fraction = st.sidebar.number_input("ejection_fraction", 0,1000) 
    age = st.sidebar.number_input("age", 0,1000)  
    serum_sodium = st.sidebar.number_input("serum_sodium", 0,1000)
    st.sidebar.markdown('<br>', unsafe_allow_html= True)


    input_variables = pd.DataFrame([{
        'time':time,
        'platelets': platelets,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'serum_creatinine': serum_creatinine,
        'ejection_fraction': ejection_fraction,
        'age': age,
        'serum_sodium': serum_sodium,
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #2B2A4C;'>Inputed Variables</h2>", unsafe_allow_html=True)
    st.write(input_variables)
    # st.write(input_variables)
    cat = input_variables.select_dtypes(include = ['object', 'category'])
    num = input_variables.select_dtypes(include = 'number')

    # Standard Scale the Input Variable.
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    for i in input_variables.columns:
        if i in num.columns:
            input_variables[i] = StandardScaler().fit_transform(input_variables[[i]])
    for i in input_variables.columns:
        if i in cat.columns: 
            input_variables[i] = LabelEncoder().fit_transform(input_variables[i])

    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Press To Predict'):
        st.markdown("<h4 style = 'color: #2B2A4C; text-align: left; font-family: montserrat '>Model Report</h4>", unsafe_allow_html = True)
        predicted = model.predict(input_variables)
        st.toast('bank_account Predicted')
        st.image('check icon.png', width = 100)
        st.success(f'Model Predicted {predicted}')
        if predicted == 0:
            st.success('The person is not likely to have Heart failure')
        else:
            st.success('The person is likely to have Heart failure')


    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>FINANCIAL INCLUSION BUILT BY Tivanny Africa</h8>",unsafe_allow_html=True)


    