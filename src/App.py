import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt

# Title and Project Description
st.title("COVID-19 Medical Ventilator Predictor")
st.subheader("COVID-19 continues to be an issue nearly 3 years after it first became relevant. Healthcare professionals continue to face many challenging decisions every day regarding these patients. More information and resources can only help these professionals in giving their patients the best possible outcome.")
st.text("")

# Predictor Form
st.header("Patient Predictor Form")

with st.form("my_form"):
   st.write("Patient Form")
   col1, col2 = st.columns(2)

   age_val = col1.number_input('Age', min_value=0, max_value=100, step=1)
   bmi_val = col1.number_input('BMI', min_value=0, max_value=100, step=1)
   creat_max_val = col1.number_input('Creatinine Max', min_value=0, max_value=30, step=1)
   crp_max_val = col1.number_input('C Reactive Protein', min_value=0, max_value=600, step=1)
   ddimer_max_val = col2.number_input('D-dimer', min_value=0, max_value=100, step=1)
   plt_min_val = col2.number_input('Platelet Count', min_value=0, max_value=600, step=1)
   tbili_max_val = col2.number_input('Total Bilirubin', min_value=0, max_value=40, step=1)
   race_val = col2.selectbox(
      'Race',
      ('White or Caucasian', 'Black or African American', 'Asian', 'Other')
   )

   st.write("Select all that apply")
   
   alc_abuse_val = st.checkbox('Alcohol Abuse')
   smoker_val = st.checkbox('Smoker')
   htn_val = st.checkbox('HTN')
   hld_val = st.checkbox('HLD')
   dm_val = st.checkbox('DM')
   cad_val = st.checkbox('CAD')
   hf_val = st.checkbox('HF')
   chronic_lung_val = st.checkbox('Chronic Lung Disease')

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   
# set values from values submitted in form
if submitted:
   if age_val == 0:
      age_val = 64

   if bmi_val == 0:
      bmi_val = 31

   if creat_max_val == 0:
      creat_max_val = 2

   if crp_max_val == 0:
      crp_max_val = 144

   if ddimer_max_val == 0:
      ddimer_max_val = 2552
   else:
      ddimer_max_val = ddimer_max_val * 100

   if plt_min_val == 0:
      plt_min_val = 199

   if tbili_max_val == 0:
      tbili_max_val = 1

   if race_val == 'White or Caucasian':
      race_val = 1
   elif race_val == 'Black or African American':
      race_val = 2
   elif race_val == 'Asian':
      race_val = 3
   else:
      race_val = 0

   if alc_abuse_val == True:
      alc_abuse_val = 2
   else:
      alc_abuse_val == 0
   
   if smoker_val == True:
      smoker_val = 2
   else:
      smoker_val = 0

   if htn_val == True:
      htn_val = 1
   else:
      htn_val = 0
   
   if hld_val == True:
      hld_val = 1
   else:
      hld_val = 0
   
   if dm_val == True:
      dm_val = 1
   else:
      dm_val = 0

   if cad_val == True:
      cad_val = 1
   else: 
      cad_val = 0

   if hf_val == True:
      hf_val = 1
   else:
      hf_val = 0

   if chronic_lung_val == True:
      chronic_lung_val = 1
   else:
      chronic_lung_val = 0

   # Vented Patients
   model = xgb.Booster()
   model.load_model("../models/vented_model.bin")

   data = np.array([[age_val,race_val,bmi_val,alc_abuse_val,smoker_val,htn_val,hld_val,dm_val,
      cad_val,hf_val,chronic_lung_val,creat_max_val,crp_max_val,ddimer_max_val,plt_min_val,tbili_max_val]])
   dm = xgb.DMatrix(data)
   chance_death = model.predict( dm )
   chance_survival = 1 - chance_death


   # Not-Vented Patients
   model = xgb.Booster()
   model.load_model("../models/not_vented_model.bin")

   data2 = np.array([[age_val,race_val,bmi_val,alc_abuse_val,smoker_val,htn_val,hld_val,dm_val,
      cad_val,hf_val,chronic_lung_val,creat_max_val,crp_max_val,ddimer_max_val,plt_min_val,tbili_max_val]])
   dm2 = xgb.DMatrix(data)
   chance_death2 = model.predict( dm2 )
   chance_survival2 = 1 - chance_death2


   # Create Pie Charts
   labels = 'Survival', 'Death'
   sizes = [chance_survival[0], chance_death[0]]
   sizes2 = [chance_survival2[0], chance_death2[0]]
   explode = (0.1, 0.1) 


   fig, (ax1, ax2) = plt.subplots(1, 2)
   fig.suptitle('Results')
   ax1.set_title("Chances on Ventilator")
   ax1.pie(sizes, explode=explode, shadow=True, autopct='%1.1f%%')
   ax1.legend(labels)
   ax2.set_title("Chances off of Ventilator")
   ax2.pie(sizes2, explode=explode, shadow=True, autopct='%1.1f%%')
   ax2.legend(labels)
   st.pyplot(fig)

   if chance_survival >= chance_survival2:
      st.subheader("Based on the results found, the patient should be put onto the ventilator")
   else:
      st.subheader("Based on the results found, the patient should remain off of the ventilator")


