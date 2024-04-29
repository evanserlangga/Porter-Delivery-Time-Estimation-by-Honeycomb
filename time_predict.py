import streamlit as st
import numpy as np

#load ML Package
import joblib
import os
import xgboost as xgb
import LinearRegression

attribut_info = """
                - Total Item: How many food you ordered
                - Market_ID: Restaurant Location
                - Order_Protocol: How do you order
                - Store_Primary_Category: What kind of restaurant
                """

mar_id = {'Region 1': 1,
          'Region 2': 2,
          'Region 3': 3,
          'Region 4': 4,
          'Region 5': 5,
          'Region 6': 6
          }

ord_pro = {'Through Porter': 1,
           'Call to Restaurant' : 2,
           'Pre-booked': 3,
           'Third Party': 4,
           'Others': 5,
           'Others': 6,
           'Others': 7,
            }

sto_ctgr = {'Ethnic Based Food': 1,
            'Specialized Food': 2,
            'Dietary Based Food': 3,
            'Others': 4
            }

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value



#def predict_delivery_time(total_item, market_id, order_protocol, restaurant_category):
    # Menggunakan rumus sederhana untuk estimasi waktu pengiriman
    #estimated_time = model_.predict([[total_item, market_id, order_protocol, restaurant_category]])
    #return estimated_time


@st.cache_data
def load_model (model):
    loaded_model = joblib.load(open(os.path.join(model),'rb'))
    return loaded_model

def run_ml():
    st.subheader('Your Food Delivery Time')
    st.caption('Please input these information for predict')
    with st.expander('What do I need to input?'):
        st.markdown(attribut_info)
    
    total_item = st.number_input('How many food did you order?', 1, 500,)
    total_distinct = st.number_input('How many types of food did you order', 1, 500)
    subtotal = st.number_input('How much does your food cost', 1, 30000)
    market_id = st.selectbox('Where the restaurant located at?',
                       ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5','Region 6'])
    order_protocol = st.selectbox('How did you order the Food?', ['Through Porter', 'Call to Restaurant', 'Pre-booked', 'Third Party',
                                    'Others'])
    rest_cate = st.selectbox('What kind of food the restaurant sell?', ['Ethnic Based Food','Specialized Food',
                                 'Dietary Based Food','Others'])
    
    with st.expander("Your Selected Options"):
        result = {
            'Total Item': total_item,
            'Total Types of Item': total_distinct,
            'Price': subtotal,
            'Location': market_id,
            'Order by': order_protocol,
            'Restaurant Type':rest_cate
            }

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5', 'Region 6']:
            res = get_value(i, mar_id)
            encoded_result.append(res)
        elif i in ['Through Porter', 'Call to Restaurant', 'Pre-booked', 'Third Party','Others']:
            res = get_value(i, ord_pro)
            encoded_result.append(res)
        elif i in ['Ethnic Based Food','Specialized Food','Dietary Based Food','Others', 'Others', 'Others']:
            res = get_value(i, sto_ctgr)
            encoded_result.append(res)
    
    single_sample = np.array(encoded_result).reshape(1,-1)
    st.write(single_sample)

    model_ = load_model('XGB_model.pkl')
    predict = model_.predict(single_sample)

    st.subheader('Your Foods Are Arriving in')
    if st.button('Predict Delivery Time'):
        # Prediksi waktu pengiriman
        predict
        st.success(f"Estimated Delivery Time: {predict} minutes")