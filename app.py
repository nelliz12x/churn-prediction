# Import all necessary libaries
import pandas as pd
import numpy as np
import streamlit as st
import pypickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler # Import package for standard scaler



load_model = pypickle.load("churn_model.pkl")


def prediction(data):

    label = LabelEncoder()
    df = pd.DataFrame(data)

    #col = [0, 1]
    #for i in col:
    #    df.iloc[:, i] = label.fit_transform(df.iloc[:, i])
    #df.iloc[0] = label.fit_transform(df.iloc[0]) # Fit and transform the column
    df.iloc[2].replace({"D 3-6 month": 3, "E 6-9 month": 6, "K > 24 month": 24, "I 18-21 month": 18,
                   "H 15-18 month": 15, "G 12-15 month": 12, "J 21-24 month": 21, "F 9-12 month": 9}, 
                   inplace=True)
    
    df.iloc[1] = label.fit_transform(df.iloc[1])
    df.iloc[16] = label.fit_transform(df.iloc[16])
    num_data = df.drop([0, 14]).values.reshape(1, -1)
    #sc = StandardScaler() # Create object for standard scaler
    #X = sc.fit_transform(num_data)
    #num_data = X

    pred = load_model.predict(num_data)

    if pred[0] == 1:
        return "The Customer will Churn"
    else:
        return "The Customer will not Churn"
    

def main():
    st.title("Customer Churn Predictive Model")
    user_id = st.text_input("Customer ID: ")
    Region = st.text_input("Client's Region: ")
    Tenure = st.text_input("duration in the network")
    Montant = st.number_input("Top-up Amount: ")
    frequency_Rech = st.number_input("Number of times the customer refilled")
    Revenue = st.number_input("Monthly income of each client: ")
    Arpu_Segment = st.number_input("income over 90 days / 3")
    frequency = st.number_input("number of times the client has made an income")
    Data_Volume = st.number_input("number of connections")
    On_net = st.number_input("inter expresso call: ")
    Orange = st.number_input("call to orange: ")
    Tigo = st.number_input("call to Tigo")
    Zone1 = st.number_input("call to zones1: ")
    Zone2 = st.number_input("call to zones2: ")
    MRG = st.text_input("a client who is going: ")
    Reqularity = st.number_input("number of times the client is active for 90 days")
    Top_pack = st.text_input("the most active packs")
    Freq_Top_pack = st.number_input("number of times the client has activated the top pack packages")

    Churn = ""

    if st.button("Result"):
        Churn = prediction([user_id, Region, Tenure, Montant, frequency_Rech, Revenue, Arpu_Segment,
                            frequency, Data_Volume, On_net, Orange, Tigo, Zone1, Zone2, MRG, Reqularity, Top_pack,
                            Freq_Top_pack])
        
    st.success(Churn)


if __name__ == "__main__":
    main()

