# app.py — PyTorch version of your Streamlit TF app
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

# Load encoders and scaler



with open (r'onehot_encoder_geo.pkl','rb') as f:
    label_encoder_geo=pickle.load(f)
with open (r'label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)
with open (r'scaler.pkl','rb') as f:
    scaler=pickle.load(f)

# 2) Recreate the ANN architecture EXACTLY as trained
class ANN_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net =nn.Sequential(
            nn.Linear(input_dim,64), #HL1
            nn.ReLU(),
            nn.Linear(64,32), #HL2
            nn.ReLU(),
            nn.Linear(32,1), #Output layer
            nn.Sigmoid(),   #Binary Classification
        )
    def forward(self,x):
        return self.net(x)

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
input_dim= getattr(scaler,"n_features_in_", None)
if input_dim is None:
    raise ValueError("Set input_dim to the feature count used during training.")

model =ANN_Model(input_dim=input_dim).to(device)

# Replace the filename with whatever you saved (e.g., 'model.pth' or 'best_weights.pth')
state_dict=torch.load(r'C:\Users\E40056416\Documents\Extra\learning\AIML\GenAI\10_Section_13\model.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Streamlit UI
st.title('Customer churn Prediction')

# User input
geography =st.selectbox('Geography',label_encoder_geo.categories_[0] )
gender =st.selectbox('Gender',label_encoder_gender.classes_)
age =st.slider('Age',18,92)
balance =st.number_input('Balance')
credit_score =st.number_input('Credit Score')
estimated_salary =st.number_input('Estimated Salary')
tenure =st.slider('Tenure',0,10)
num_of_product =st.slider('Number of Products',1,4)
has_cr_card =st.selectbox('Has Credit Card',[0,1])
is_active_member =st.selectbox('Is Active Member',[0,1])

# Prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

#scale the input data
input_data_scaled=scaler.transform(input_data)

# Predict (PyTorch)
# make sure it's float32 and on the right device
x= torch.from_numpy(input_data_scaled.astype(np.float32)).to(device)
with torch.no_grad():
    probs=model(x).squeeze(1).cpu().numpy() # shape: (N,)

# Get a scalar
prob= float(probs[0] if probs.ndim>0 else float(probs.item()))
pred=int(prob>=0.5)
st.write(f"Churn probability: {prob:.2f}")

if pred==1:
    st.warning("The customer is likely to churn.")
else:

    st.success("The customer is not likely to churn.")
