import streamlit as st
import pandas as pd
import numpy as np
import pickle
import meta
from utils.st import (
    remote_css,
    local_css,

)
from utils.utils import load_image_from_local
def main():
    st.set_page_config(
        page_title="Bike Price Predictor",
        page_icon="🛵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    remote_css("https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Poppins:wght@600&display=swap")
    local_css("assets/css/style.css")
    # importing the pickle file
    model = pickle.load(open('Linearregression_bike.pkl', 'rb'))
    # Loading the bike file
    bike = pd.read_csv('bike_cleaned.csv')



    col1, col2 = st.columns([6, 4])
    with col2:
        st.image(load_image_from_local("assets/images/logo2.png"),width=450)

        with st.expander("How Mr.Patel got this power?", expanded=True):
            st.markdown(meta.STORY, unsafe_allow_html=True)

    with col1:
        st.markdown(meta.HEADER_INFO, unsafe_allow_html=True)

        st.markdown(meta.BIKE_INFO, unsafe_allow_html=True)
        # Brand name box
        brands = np.sort(bike['company'].unique())
        Brand = st.selectbox('Brand Name', brands)

        # Model of the bike
        models = np.sort(bike[bike['name'].str.split(' ').str.get(0) == Brand]['name'].unique())
        Model = st.selectbox('Model Name', models)

        # Year bought
        Year = st.number_input('Purchase Year', min_value=1995, max_value=2023)

        # Kms Travelled
        Kms = st.number_input('Kms Travelled', min_value=0, max_value=200000)

        # Prediction button
        def predict_price(Brand, Model, Year, Kms):
            input = np.array([Brand, Model, Year, Kms]).reshape(1, 4)
            columns = ['company', 'name', 'year', 'travelled']
            prediction = model.predict(pd.DataFrame(columns=columns, data=input))

            return int(prediction)

        if st.button("Predict the Price"):
            output = predict_price(Brand, Model, Year, Kms)
            st.success('The Estimated Price is -{} 💵'.format(output))



if __name__ == '__main__':
    main()