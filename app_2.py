from numpy import array
import numpy as np
import streamlit as st 
import folium 
from streamlit_folium import st_folium , folium_static # Untuk membuat MAP
from streamlit_option_menu import option_menu # Untuk membuat SIDE BAR
import pandas as pd
import PIL.Image as img
import time
import pickle
from time_predict import run_ml

#Tampilan halaman utama
st.title('Porter Delivery Time Estimation')


image = img.open('porterimg2.jpg')
st.image(image)


desc_1 = """
        Porter is a tech-enabled logistics company offering a variety of intracity 
        and intercity delivery services. From a pin to an entire house, Porter will 
        deliver anything, anywhere, anytime. However, Porter does not deliver item(s) 
        which are illegal, dangerous or hazardous in nature or which are prohibited by 
        any statue or law. 
        """

desc_2 = """
        This app is made, as a Final Project from Digital Skola, by Team Honeycomb. 
        This app will do Delivery Time Estimation on Porter delivery services. 
        The whole process upon making this app started from thorough data preprocessing, 
        model selection & training, model evaluation until model tuning. We believe the 
        prediction the app generates is accurate as we made it from the best of our knowledge.
        """
df = pd.DataFrame({'Team Member': 
                   ['Jessica Manna Febriani Nadeak',
                    'Ignatius Evans Erlangga',
                    'Hapsari Warih Utami',
                    'Feri Aditya Ridwan Mas',
                    'Feliks Wijaya Santoso',
                    'Fela Suvah'
                    ]})


page = option_menu('MENU' , ['About Porter', 'About Us'])
if page == 'About Porter':
   st.caption(desc_1)
elif page == 'About Us':
   st.caption (desc_2)
   st.subheader('Team Honeycomb consists of :')
   st.dataframe(df.style.hide())

def main():
   menu = ['Home', 'Delivery Time Prediction']
   choice = st.sidebar.selectbox('Menu', menu)
    
   if choice == 'Home':
      print('')
   elif choice == 'Delivery Time Prediction':
      run_ml()
    
if __name__== '__main__':
   main()

    

