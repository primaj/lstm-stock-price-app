import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

trained = False

# Define the sidebar ---------------------------------------------------------------------------------------------------
st.sidebar.markdown('# Options')
company = st.sidebar.text_input('Enter a ticker:', 'AAPL')
plot_history = st.sidebar.checkbox("Plot History?")

st.sidebar.markdown('### Training Data Range')
training_start = st.sidebar.date_input('Start', dt.datetime(2012, 1, 1))
training_end = st.sidebar.date_input('End', dt.datetime(2020, 1, 1))

st.sidebar.markdown('### Testing Data Range')
st.sidebar.markdown("Defaults to the end of the training data till the most recent data Yahoo has")
testing_start = st.sidebar.date_input('Start', training_end)
testing_end = st.sidebar.date_input('End', dt.datetime.now())

st.sidebar.markdown('## Model Parameters 📊')
prediction_days = st.sidebar.number_input('Prediction days:', 2, 100, 60)
epochs = st.sidebar.number_input('Epochs:', 2, 100, 25)
batch_size = st.sidebar.number_input('Batch size:', 2, 100, 32)
units_per_lstm_layer = st.sidebar.number_input('Units per LSTM Layer:', 2, 100, 50)
dropout_portion = st.sidebar.slider('Dropout Portion per LSTM Layer:', 0.0, 1.0, 0.2)

combine_training = st.sidebar.checkbox("Combine Training?")

# Main Panel -----------------------------------------------------------------------------------------------------------
st.title('🚀 LSTM Stock Price Prediction 🚀')
st.markdown(f'## For ticker: {company}')

st.markdown('*Obligatory this is not financial advice*')
# Read the data from yahoo
data = web.DataReader(company, 'yahoo', training_start, training_end)

st.markdown(f'### Data sample')
st.write(data.head())

if plot_history:
    f'''### Historic Price'''
    fig = px.line(data, x=data.index, y=data.Close, title=f'{company} Closing Price History')
    st.plotly_chart(fig)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

f'''
### Scaled Data
We are going to predict the closing price, which must first be scaled between 0 and 1 using: {scaler}

'''

#
X_train = []
y_train = []

for X in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[X - prediction_days:X, 0])
    y_train.append(scaled_data[X, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(units=units_per_lstm_layer, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(dropout_portion))
model.add(LSTM(units=units_per_lstm_layer, return_sequences=True))
model.add(Dropout(dropout_portion))
model.add(LSTM(units=units_per_lstm_layer))
model.add(Dropout(dropout_portion))
model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

f'''
## Model Definition
### Code
'''
st.code(
    '''
    model = Sequential()
    model.add(LSTM(units=units_per_lstm_layer, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_portion))
    model.add(LSTM(units=units_per_lstm_layer, return_sequences=True))
    model.add(Dropout(dropout_portion))
    model.add(LSTM(units=units_per_lstm_layer))
    model.add(Dropout(dropout_portion))
    model.add(Dense(units=1))
''')
'''### Summary'''
st.code(short_model_summary)

train = st.button("Train Model!")

if train:
    with st.spinner("Model training..."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        st.success("Model Trained!")
        trained = True

# Only do if we have a model trained
if trained:
    # Read the test data from yahoo
    test_data = web.DataReader(company, 'yahoo', testing_start, testing_end)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    # Define the data to use for predictions
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    X_test = [
        model_inputs[X - prediction_days:X, 0]
        for X in range(prediction_days, len(model_inputs))
    ]

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict on the data and inverse transform
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Create the plotting data
    test_data['Predicted Price'] = predicted_prices
    test_data.rename(columns={'Close': 'Actual Price (Testing)'}, inplace=True)
    test_data['date'] = test_data.index

    plot_data = pd.melt(test_data, id_vars=['date'],
                        value_vars=['Actual Price (Testing)', 'Predicted Price'], value_name='value')

    # add the training data to plot data if selected
    if combine_training:
        data['date'] = data.index
        data.rename(columns={'Close': 'Actual Price (Training)'}, inplace=True)
        long_training = pd.melt(data, id_vars=['date'], value_vars=['Actual Price (Training)'], value_name='value')
        plot_data = pd.concat([plot_data, long_training])

    # Define plotly plot
    pred_fig = px.line(plot_data, x='date', y='value', color='variable')
    st.plotly_chart(pred_fig)

    # Predicting next day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    prediction_date = (max(test_data['date']) + dt.timedelta(days=1)).date()

    f'''
    ### Prediction for next day ({prediction_date}):
    **$ {round(float(prediction[0]), 2)}**
    '''
