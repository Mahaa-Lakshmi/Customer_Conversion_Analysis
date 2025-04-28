import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from scipy import stats

# Load data
train_data = pd.read_csv("E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/data/train.csv")
#test_data = pd.read_csv("E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/data/test.csv")

print("dataset loaded and preprocess started")

# Drop unnecessary columns
train_data.drop(["year"], axis=1, inplace=True)

## ----------------------
## 1. COUNTRY ENCODING
## ----------------------
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

train_data['country_group'] = train_data['country'].map(country_mapping).apply(group_countries)
encoded_country = pd.get_dummies(train_data['country_group'], prefix='country_group', dtype=int)
train_data = pd.concat([train_data, encoded_country], axis=1)

## ----------------------
## 2. PRODUCT CATEGORY ENCODING (PAGE1)
## ----------------------
product_mapping = {1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"}
train_data['page1_main_category'] = train_data['page1_main_category'].map(product_mapping)
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(train_data[['page1_main_category']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['page1_main_category']), index=train_data.index)
train_data = pd.concat([train_data, encoded_df], axis=1)

## ----------------------
## 3. PRODUCT MODEL ENCODING (PAGE2) - FREQUENCY ENCODING
## ----------------------
page2_freq = train_data['page2_clothing_model'].value_counts(normalize=True)
train_data['page2_clothing_model_encoded'] = train_data['page2_clothing_model'].map(page2_freq)

## ----------------------
## 4. COLOR ENCODING
## ----------------------
color_group_mapping = {
    1: "neutral", 2: "dark", 3: "cool", 4: "dark", 5: "warm",
    6: "neutral", 7: "cool", 8: "cool", 9: "multi", 10: "warm",
    11: "warm", 12: "warm", 13: "cool", 14: "neutral"
}
train_data['colour_group'] = train_data['colour'].map(color_group_mapping)
train_data = pd.get_dummies(train_data, columns=['colour_group'], dtype=int)

## ----------------------
## 5. LOCATION ENCODING (POSITION COORDINATES)
## ----------------------
position_mapping = {
    1: (-1, 1),   # Top Left
    2: (0, 1),    # Top Middle
    3: (1, 1),    # Top Right
    4: (-1, -1),  # Bottom Left
    5: (0, -1),   # Bottom Middle
    6: (1, -1)    # Bottom Right
}
train_data[['x_axis', 'y_axis']] = train_data['location'].apply(lambda pos: pd.Series(position_mapping[pos]))

## ----------------------
## 6. MODEL PHOTOGRAPHY & PRICE2 ENCODING
## ----------------------
train_data['model_photography_encoded'] = train_data['model_photography'].apply(lambda x: 0 if x == 1 else 1)
train_data['price_2_encoded'] = train_data['price_2'].apply(lambda x: 1 if x == 1 else 0)

## ----------------------
## 7. SESSION-LEVEL FEATURE ENGINEERING
## ----------------------
# Total clicks per session
train_data['total_clicks'] = train_data.groupby('session_id')['order'].transform('count')

# Average price per session
train_data['avg_price'] = train_data.groupby('session_id')['price'].transform('mean')

# Unique products viewed per session
train_data['unique_products'] = train_data.groupby('session_id')['page2_clothing_model'].transform('nunique')

# Browsing depth (max page reached)
train_data['browsing_depth'] = train_data.groupby('session_id')['page'].transform('max')

# Session duration (assuming 'order' represents chronological sequence)
train_data['session_duration'] = train_data.groupby('session_id')['order'].transform(lambda x: x.max() - x.min())

# Click density (clicks per minute)
train_data['click_density'] = train_data['total_clicks'] / (train_data['session_duration'] + 1e-6)

# Weekday vs Weekend (after creating proper date)
train_data['date'] = pd.to_datetime('2008-' + train_data['month'].astype(str) + '-' + train_data['day'].astype(str))
train_data['weekday'] = train_data['date'].dt.weekday.apply(lambda x: 1 if x < 5 else 0)
train_data.drop('date', axis=1, inplace=True)

# High price preference
median_price = train_data['price'].median()
train_data['high_price_preference'] = (train_data['avg_price'] > median_price).astype(int)

## ----------------------
## 8. TARGET VARIABLE CREATION
## ----------------------
# Classification target: Purchase (1 if last action in session)
train_data['purchase'] = train_data.groupby('session_id')['order'].transform(lambda x: (x == x.max()).astype(int))

# Regression target: Total revenue per session
train_data['revenue'] = train_data.groupby('session_id')['price'].transform('sum')

## ----------------------
## 9. FINAL CLEANUP & SCALING
## ----------------------
# Drop redundant columns
cols_to_drop = ['page1_main_category','country','country_group' ,'page2_clothing_model', 'location', 'model_photography', 'price_2', 'colour']
train_data.drop(cols_to_drop, axis=1, inplace=True)


# Scale numerical features
numerical_cols = ['order', 'price', 'page', 'total_clicks', 'avg_price', 
                 'unique_products', 'browsing_depth', 'session_duration', 'click_density']
scaler = StandardScaler()
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# Save processed data
train_data.to_csv("E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/processedData/train_data_processed.csv", index=False)

print("Preprocessing completed")