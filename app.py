import pickle
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file) 
    
def predict(model, X):
    return model.predict(X)

def get_airlines():
    df = pd.read_csv('./datasets/Clean_Dataset.csv')
    return df['airline'].unique().tolist()

def get_time_slot(time):
    hour, minute = map(int, time.split(':'))
    if hour < 3:
        return 'Late_Night'
    elif hour < 6:
        return 'Early_Morning'
    elif hour < 12:
        return 'Morning'
    elif hour < 17:
        return 'Afternoon'
    elif hour < 20:
        return 'Evening'
    else:
        return 'Night'
    
def predict_duration(source_city, destination_city, stops):
    city_pair_duration = {
        ('Delhi', 'Mumbai') : 2.33,
        ('Delhi', 'Bangalore'): 2.83,
        ('Delhi', 'Kolkata') : 2.17,
        ('Delhi', 'Hyderabad') : 2.25,
        ('Delhi', 'Chennai') : 2.83,
        ('Mumbai', 'Delhi') : 2.33,
        ('Mumbai', 'Bangalore') : 1.75,
        ('Mumbai', 'Kolkata') : 2.67,
        ('Mumbai', 'Hyderabad') : 1.5,
        ('Mumbai', 'Chennai') : 2,
        ('Bangalore', 'Delhi') : 2.83,
        ('Bangalore', 'Mumbai') : 1.75,
        ('Bangalore', 'Kolkata') : 2.5,
        ('Bangalore', 'Hyderabad') : 1.25,
        ('Bangalore', 'Chennai') : 1,
        ('Kolkata', 'Delhi') : 2.17,
        ('Kolkata', 'Mumbai') : 2.67,
        ('Kolkata', 'Bangalore') : 2.5,
        ('Kolkata', 'Hyderabad') : 2.25,
        ('Kolkata', 'Chennai') : 2.42,
        ('Hyderabad', 'Delhi') : 2.25,
        ('Hyderabad', 'Mumbai') : 1.5,
        ('Hyderabad', 'Bangalore') : 1.25,
        ('Hyderabad', 'Kolkata') : 2.25,
        ('Hyderabad', 'Chennai') : 1.33,
        ('Chennai', 'Delhi') : 2.83,
        ('Chennai', 'Mumbai') : 2,
        ('Chennai', 'Bangalore') : 1,
        ('Chennai', 'Kolkata') : 2.42,
        ('Chennai', 'Hyderabad') : 1.33
    }   
    stop_duration = 1.5
    base_duration = city_pair_duration.get((source_city, destination_city), 0)
    total_duration = base_duration + (stops * stop_duration)
    return round(total_duration, 2)  

def get_encodings():
    clean_data = pd.read_csv("./datasets/Clean_Dataset.csv")
    processed_data = pd.read_csv("./datasets/final_dataset.csv")  
    categorical_columns = ['flight', 'departure_time', 'arrival_time', 'class'] 
    encodings = {}
    for column in categorical_columns:
        original_values = clean_data[column].unique()
        encoded_values = processed_data[column].unique()

        mapping_df = pd.DataFrame({
            'Original Value': original_values,
            'Encoded Value': encoded_values
        })
        encodings[column] = mapping_df
    return encodings

def get_categorical_encodings():
    categorical_encoding = {}
    airline_cols = ['airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara']
    i = 0
    categorical_encoding['airline'] = {}
    for airline in airline_cols:
        l = [0] * len(airline_cols)
        l[i] = 1
        i += 1
        airline = airline.split('airline_')[1]
        categorical_encoding['airline'][airline] = l
        
    source_city_cols = ['source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai']
    i = 0
    categorical_encoding['source_city'] = {}
    for city in source_city_cols:
        l = [0] * len(source_city_cols)
        l[i] = 1
        i += 1
        city = city.split('source_city_')[1]
        categorical_encoding['source_city'][city] = l
        
    destination_city_cols = ['destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai']
    i = 0
    categorical_encoding['destination_city'] = {}
    for city in destination_city_cols:
        l = [0] * len(destination_city_cols)
        l[i] = 1
        i += 1
        city = city.split('destination_city_')[1]
        categorical_encoding['destination_city'][city] = l
        
    return categorical_encoding

def get_encoded_value(class_name, mapping_df):
    encoded_value = mapping_df.loc[mapping_df['Original Value'] == class_name, 'Encoded Value']
    if not encoded_value.empty:
        return int(encoded_value.values[0])
    else:
        return None 
    
def get_decoded_value(encoded_value, mapping_df):
    original_value = mapping_df.loc[mapping_df['Encoded Value'] == encoded_value, 'Original Value']
    if not original_value.empty:
        return original_value.values[0]
    else:
        return None
    
def categorical_encode(data, encodings):
    return encodings[data]

def extract_flight(airline):
    d = {
        'AirAsia': 1213,
        'Air_India': 878,
        'GO_FIRST': 1013,
        'Indigo': 245,
        'SpiceJet': 1408,
        'Vistara': 1559
    }
    return d[airline]

model = load_model('./checkpoint/random_forest.pkl')
encodings = get_encodings()
categorical_encodings = get_categorical_encodings()
airlines = get_airlines()
src_city = input('Enter source: ').title()
src = categorical_encode(src_city, categorical_encodings['source_city'])
dst_city = input('Enter destination: ').title()
dst = categorical_encode(dst_city, categorical_encodings['destination_city'])
date_of_departure = input('Enter date of departure [DD/MM/YYYY]: ')
departure_date = datetime.strptime(date_of_departure, '%d/%m/%Y')
today_date = datetime.now()
days_left = (departure_date - today_date).days
departure_time = get_encoded_value(get_time_slot(input('Enter departure time [HH:MM]: ')), encodings['departure_time'])
date_of_arrival = input('Enter date of arrival [DD/MM/YYYY]: ')
arrival_time = get_encoded_value(get_time_slot(input('Enter arrival time [HH:MM]: ')), encodings['arrival_time'])
seat_class = get_encoded_value(input('Enter seat class [Economy/Business]: '), encodings['class'])
stops = min(int(input('Enter number of preferred stops: ')), 2)
duration = predict_duration(src_city, dst_city, stops)

data = {}

columns = [
    "flight", "departure_time", "stops", "arrival_time", "class", "duration", "days_left", "airline_AirAsia",
    "airline_Air_India", "airline_GO_FIRST", "airline_Indigo", "airline_SpiceJet", "airline_Vistara", "source_city_Bangalore",
    "source_city_Chennai", "source_city_Delhi", "source_city_Hyderabad", "source_city_Kolkata", "source_city_Mumbai", "destination_city_Bangalore",
    "destination_city_Chennai", "destination_city_Delhi", "destination_city_Hyderabad", "destination_city_Kolkata",
    "destination_city_Mumbai"
]

min_price = float('inf')
best_airline = None
min_price_day = None

for i in range(days_left, -1, -1):
    data[i] = {}
    for j in airlines:
        flight = extract_flight(j)
        airline = categorical_encode(j, categorical_encodings['airline'])
        features = [flight, departure_time, stops, arrival_time, seat_class, duration, i] + airline + src + dst
        features = pd.Series(features, index=columns)
        features = features.values.reshape(1, -1)
        price = predict(model, features)[0]
        data[i][j] = price
        if price < min_price:
            min_price = price
            best_airline = j
            min_price_day = i
            
print(f'Best Airline: {best_airline}')
print(f'Price: {min_price}')
best_date = departure_date - timedelta(days=min_price_day)
print(f'Best Date: {best_date.strftime("%d/%m/%Y")}')

fig, ax = plt.subplots(figsize=(10, 6))

for airline in airlines:
    airline_prices = [data[days][airline] for days in range(days_left, -1, -1)]
    ax.plot(range(days_left, -1, -1), airline_prices, label=airline)

ax.axvline(x=min_price_day, color='red', linestyle='--', label=f'Min Price: {min_price}')

ax.set_xlabel('Days Left')
ax.set_ylabel('Price')
ax.set_title('Price vs. Days Left for Each Airline')
ax.legend()

plt.grid(True)
plt.show()