# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import datetime
import plotly.express as px

from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load best models
best_classification_model = joblib.load('models/best_model_classification.pkl')
best_regression_model = joblib.load('models/best_model_regression.pkl')
best_clustering_model = joblib.load('models/best_model_clustering.pkl')

# --------------- Preprocessing Function --------------- #

#global_variables



def preprocess_uploaded_data(df,type="file"):
    # Step 1: Drop 'year'
    #df.drop('year', axis=1, inplace=True)
    
    # Step 2: Country Mapping
    country_mapping = {
    1: "Australia", 2: "Austria", 3: "Belgium", 4: "British Virgin Islands", 
    5: "Cayman Islands", 6: "Christmas Island", 7: "Croatia", 8: "Cyprus", 
    9: "Czech Republic", 10: "Denmark", 11: "Estonia", 12: "unidentified", 
    13: "Faroe Islands", 14: "Finland", 15: "France", 16: "Germany", 
    17: "Greece", 18: "Hungary", 19: "Iceland", 20: "India", 21: "Ireland",
    22: "Italy", 23: "Latvia", 24: "Lithuania", 25: "Luxembourg", 26: "Mexico", 
    27: "Netherlands", 28: "Norway", 29: "Poland", 30: "Portugal", 
    31: "Romania", 32: "Russia", 33: "San Marino", 34: "Slovakia", 
    35: "Slovenia", 36: "Spain", 37: "Sweden", 38: "Switzerland", 
    39: "Ukraine", 40: "United Arab Emirates", 41: "United Kingdom", 
    42: "USA", 43: "biz (.biz)", 44: "com (.com)", 45: "int (.int)", 
    46: "net (.net)", 47: "org (*.org)"
}
    
    def group_countries(country_name):
        eu_countries = ["Austria", "Belgium", "Croatia", "Cyprus", "Czech Republic", 
                   "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", 
                   "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", 
                   "Luxembourg", "Netherlands", "Portugal", "Romania", 
                   "Slovakia", "Slovenia", "Spain", "Sweden"]
    
        non_eu_europe = ["Norway", "Switzerland", "United Kingdom", "Ukraine", 
                        "Russia", "Iceland", "Faroe Islands"]

        if country_name == "Poland":
            return "Poland"
        elif country_name in eu_countries:
            return "EU Countries"
        elif country_name in non_eu_europe:
            return "Non-EU Countries"
        else:
            return "Countries outside Europe"

    df['country_group'] = df['country'].map(country_mapping).apply(group_countries)
    country_encoded = pd.get_dummies(df['country_group'], prefix='country_group')
    df = pd.concat([df, country_encoded], axis=1)

    # Step 3: page1_main_category Encoding
    product_mapping = {1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"}
    
    df['page1_main_category'] = df['page1_main_category'].map(product_mapping)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df[['page1_main_category']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['page1_main_category']), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)


    # Step 4: page2_clothing_model Encoding (Frequency)
    page2_freq = df['page2_clothing_model'].value_counts(normalize=True)
    df['page2_clothing_model_encoded'] = df['page2_clothing_model'].map(page2_freq)

    # Step 5: Colour Encoding    
    color_group_mapping = {
    1: "neutral", 2: "dark", 3: "cool", 4: "dark", 5: "warm",
    6: "neutral", 7: "cool", 8: "cool", 9: "multi", 10: "warm",
    11: "warm", 12: "warm", 13: "cool", 14: "neutral"}

    df['colour_group'] = df['colour'].map(color_group_mapping)
    df = pd.get_dummies(df, columns=['colour_group'])

    # Step 6: Location -> x_axis, y_axis
    location_mapping = {
    1: (-1, 1),   # Top Left
    2: (0, 1),    # Top Middle
    3: (1, 1),    # Top Right
    4: (-1, -1),  # Bottom Left
    5: (0, -1),   # Bottom Middle
    6: (1, -1)    # Bottom Right
}
    df[['x_axis', 'y_axis']] = df['location'].apply(lambda x: pd.Series(location_mapping[x]))

    # Step 7: Model Photography
    df['model_photography_encoded'] = df['model_photography'].apply(lambda x: 0 if x == 1 else 1)

    # Step 8: Price2
    df['price_2_encoded'] = df['price_2'].apply(lambda x: 1 if x == 1 else 0)

    # Step 9: Session Features
    df['total_clicks'] = df.groupby('session_id')['order'].transform('count')
    df['avg_price'] = df.groupby('session_id')['price'].transform('mean')
    df['unique_products'] = df.groupby('session_id')['page2_clothing_model'].transform('nunique')
    df['browsing_depth'] = df.groupby('session_id')['page'].transform('max')
    df['session_duration'] = df.groupby('session_id')['order'].transform(lambda x: x.max() - x.min())
    df['click_density'] = df['total_clicks'] / (df['session_duration'] + 1e-6)

    # Step 10: Weekday
    df['date'] = pd.to_datetime(df['year'].astype(str)+ '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
    df['weekday'] = df['date'].dt.weekday.apply(lambda x: 1 if x < 5 else 0)
    df.drop('date', axis=1, inplace=True)

    # Step 11: High Price Preference
    median_price = df['price'].median()
    df['high_price_preference'] = (df['avg_price'] > median_price).astype(int)

    # Step 12: Scale numerical features
    numerical_cols = ['order', 'price', 'page', 'total_clicks', 'avg_price', 
                     'unique_products', 'browsing_depth', 'session_duration', 'click_density']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    
    if type=="form":
        expected_cols = ['month', 'day', 'order', 'session_id', 'price', 'page',
                 'country_group_Countries outside Europe', 'country_group_EU Countries',
                 'country_group_Non-EU Countries', 'country_group_Poland',
                 'page1_main_category_sale', 'page1_main_category_skirts',
                 'page1_main_category_trousers', 'page2_clothing_model_encoded',
                 'colour_group_cool', 'colour_group_dark', 'colour_group_multi',
                 'colour_group_neutral', 'colour_group_warm', 'x_axis', 'y_axis',
                 'model_photography_encoded', 'price_2_encoded', 'total_clicks',
                 'avg_price', 'unique_products', 'browsing_depth', 'session_duration',
                 'click_density', 'weekday', 'high_price_preference']
        existing_columns = df.columns.to_list()
        missing_columns = [col for col in expected_cols if col not in existing_columns]
        for col in missing_columns:
            df[col] = 0
        
    # Step 13: Final Columns to Match Model Input
    final_features = ['month', 'day', 'order', 'price', 'page', 'country_group_Countries outside Europe', 'country_group_EU Countries', 'country_group_Non-EU Countries', 'country_group_Poland', 'page1_main_category_sale', 'page1_main_category_skirts', 'page1_main_category_trousers', 'page2_clothing_model_encoded', 'colour_group_cool', 'colour_group_dark', 'colour_group_multi', 'colour_group_neutral', 'colour_group_warm', 'x_axis', 'y_axis', 'model_photography_encoded', 'price_2_encoded', 'total_clicks', 'avg_price', 'unique_products', 'browsing_depth', 'session_duration', 'click_density', 'weekday', 'high_price_preference']
    processed_df = df[final_features]    
    
    return processed_df

# --------------- UI Layout --------------- #

st.set_page_config(page_title="🛒 Smart Customer Session Predictor", layout="wide")

st.title("🛒 Customer Session Prediction App")
st.markdown("""
Welcome to the **Smart Customer Session Predictor**!  
You can either **upload a session CSV file** or **manually input customer session details** below.
""")

st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select how you want to provide input:", ("Upload CSV", "Manual Entry"))

# Upload CSV Mode
if input_method == "Upload CSV":
    st.subheader("📂 Upload your Customer Session CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        raw_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded successfully! {raw_data.shape[0]} records found.")

        # Preprocessing
        processed_data = preprocess_uploaded_data(raw_data.copy())
        st.info(f"🔍 Preprocessed {processed_data.shape[0]} rows successfully.")

        # Predictions
        purchase_preds = best_classification_model.predict(processed_data)
        revenue_preds = best_regression_model.predict(processed_data)
        cluster_preds = best_clustering_model.predict(processed_data)

        # Attach Predictions
        raw_data['Predicted_Purchase'] = purchase_preds
        raw_data['Predicted_Revenue'] = np.round(revenue_preds, 2)
        raw_data['Predicted_Cluster'] = cluster_preds

        st.subheader("🔮 Sample Predictions")
        st.dataframe(raw_data.head())

         # Visualization Section
        st.subheader("📊 Visualizations")

        # Pie chart for Cluster Distribution
        st.markdown("### 🎯 Predicted Cluster Distribution")
        cluster_counts = raw_data['Predicted_Cluster'].value_counts()

        # Create a mapping dictionary for your cluster labels
        cluster_label_mapping = {
            0: "Budget-oriented / Window Shoppers",
            1: "High-value / Likely Buyers"
        }

        # Map the numerical cluster labels to the descriptive labels
        labeled_index = cluster_counts.index.map(cluster_label_mapping)

        st.plotly_chart(
            px.pie(
                names=labeled_index, 
                values=cluster_counts.values, 
                title='Predicted Customer Clusters',
                hole=0.4
            )
        )

        # Histogram for Predicted Revenue
        st.markdown("### 💰 Predicted Revenue Distribution")
        st.plotly_chart(
            px.histogram(
                raw_data, 
                x='Predicted_Revenue', 
                nbins=30, 
                title='Distribution of Predicted Revenue',
                color_discrete_sequence=['#636EFA']
            )
        )

        # Bar chart for Purchase Prediction
        st.markdown("### 🛍️ Purchase Prediction Counts")
        purchase_counts = raw_data['Predicted_Purchase'].value_counts()
        st.plotly_chart(
            px.bar(
                x=purchase_counts.index.map({0: 'No Purchase', 1: 'Purchase'}),
                y=purchase_counts.values,
                labels={'x': 'Purchase Outcome', 'y': 'Number of Sessions'},
                title='Purchase vs No Purchase'
            )
        )

        # Downloadable CSV
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(raw_data)

        st.download_button(
            label="⬇️ Download Full Predictions CSV",
            data=csv,
            file_name='customer_predictions.csv',
            mime='text/csv'
        )

    else:
        st.warning("👈 Please upload a file to proceed.")

    

# Manual Entry Mode
else:
    st.subheader("📝 Manual Entry Form")
    st.markdown("Fill in session details below to predict purchase, revenue, and cluster group.")

    country_list=["Australia", "Austria", "Belgium", "British Virgin Islands", "Cayman Islands", "Christmas Island", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "unidentified", "Faroe Islands", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "India", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Arab Emirates", "United Kingdom", "USA", "biz (.biz)", "com (.com)", "int (.int)", "net (.net)", "org (*.org)"]
    page2_clothing_model_list=['A1', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A2', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A3', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A4', 'A40', 'A41', 'A42', 'A43', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B19', 'B2', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B3', 'B30', 'B31', 'B32', 'B33', 'B34', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C4', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C5', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C59', 'C6', 'C7', 'C8', 'C9', 'P1', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P2', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P29', 'P3', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P4', 'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P5', 'P50', 'P51', 'P52', 'P53', 'P54', 'P55', 'P56', 'P57', 'P58', 'P59', 'P6', 'P60', 'P61', 'P62', 'P63', 'P64', 'P65', 'P66', 'P67', 'P68', 'P69', 'P7', 'P70', 'P71', 'P72', 'P73', 'P74', 'P75', 'P76', 'P77', 'P78', 'P79', 'P8', 'P80', 'P81', 'P82', 'P9']
    page1_main_category_list=['sale', 'skirts', 'trousers','blouses']
    colors_list=["beige", "black", "blue", "brown", "burgundy", "gray", "green", "navy blue", "of many colors", "olive", "pink", "red", "violet", "white"]
    location_list=["top left", "top in the middle", "top right", "bottom left", "bottom in the middle", "bottom right"]
    model_photography_list=['en face', 'profile']


    # Form for manual input
    with st.form(key='manual_entry_form'):
        d = st.date_input("select date, month, year", min_value=datetime.date(2000, 1, 1), value= datetime.datetime.now())
        item=str(d).split(" ")[0].split("-")
        year=item[0]
        month = item[1]
        day = item[2]
        order = st.number_input('Order (sequence of clicks during one session)', min_value=1, value=5)
        country = st.selectbox('Country', country_list)
        session_id = st.number_input('Session id', min_value=1, value=5)
        page1_main_category = st.selectbox('Main Category',page1_main_category_list )
        page2_clothing_model = st.selectbox('page2_clothing_model', page2_clothing_model_list)
        colors= st.selectbox('Colors', colors_list)
        location=st.selectbox('Position in page', location_list)
        model_photography = st.selectbox('Model Photography', model_photography_list )
        price = st.number_input('Price in US dollars', min_value=0.0, value=29.99)
        price_2 = st.selectbox('Price 2 (The average price for the entire product category)', ['Yes', 'No'])
        page = st.number_input('Page Number', min_value=1, max_value=5,value=2)

        submitted = st.form_submit_button("Predict Session")

    if submitted:
        # Create single-row DataFrame
        form_data = {
            'year':int(year),
            'month': int(month),
            'day': int(day),
            'order': order,
            'country': country_list.index(country)+1,
            'session_id': session_id,
            'page1_main_category' : page1_main_category_list.index(page1_main_category)+1,
            'page2_clothing_model' : page2_clothing_model_list.index(page2_clothing_model)+1,
            'colour': colors_list.index(colors)+1,
            'location':location_list.index(location)+1,
            'model_photography': model_photography_list.index(model_photography)+1,
            'price':price,
            'price_2':1 if price_2=='Yes' else 2,
            'page':page
            
        }

        single_input = pd.DataFrame([form_data])

        # Preprocessing
        processed_data1 = preprocess_uploaded_data(single_input,type="form")
        st.info(f"🔍 Preprocessed Submitted inputs successfully.")

        # Predictions
        purchase_pred = best_classification_model.predict(processed_data1)
        revenue_pred = str(best_regression_model.predict(processed_data1))
        cluster_pred = str(best_clustering_model.predict(processed_data1))

        st.info(f"""✅ Purchase Prediction: **{'Yes' if purchase_pred==1 else 'No'}**""")
        st.info(f"""💰 Estimated Revenue: **${revenue_pred[1:-1]}**""")
        st.info(f"👥 Cluster Group Assigned: **Cluster {cluster_pred[1:-1]}**")

# --------------- Sidebar Notes --------------- #

with st.sidebar.expander("ℹ️ Clustered into 2 groups"):
    st.markdown("""
We used **K=2** for customer clustering to simplify groups into:
- **Cluster 0:** Budget-oriented / Window Shoppers
- **Cluster 1:** High-value / Likely Buyers
""")
