# Created by: Brian Guevara
# WGU Computer Science Capstone C964

# List of Imports needed for the program to run:
import os
from datetime import date, datetime
import streamlit as st
from dateutil.relativedelta import relativedelta
from plotly import graph_objs as go
import pandas as pd
from pandas_datareader import data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set the dates for our info when grabbing data. We are only taking data within the last 8 months, so it is relevant.
# 8 months of data is enough for our training model and information. If we take in too much data, then the prediction
# will have more unexpected results.
TODAY = date.today().strftime("%Y-%m-%d")
START = date.today() + relativedelta(months=-8)

# Set the title of the web page both in the browser tab and page
st.set_page_config(page_title='Stock Predictor')
st.title("Stock Prediction App")

# INTERACTIVE QUERY
# Create our input field for the user to select any stock they want
stock = st.text_input("Type Stock Name", value='SPY').upper()

# Create a number adjustment 'slider' that allows the user to change the Moving Average they would like to use.
# We use a default of 14 to match with RSI, along with a minimum of 10 and a maximum 0f 200
moving_avg_period = st.number_input("Exponential Moving Average (Default 14, Min 10, Max 200)", value=14, min_value=10,
                                    max_value=200)

# There is some text under the interactive query that lets the user know if data was loaded, waiting to load, or
# if another stock name needs to be chosen.
data_load_state = st.text("Loading stock information...")
moving_avg_title = str(moving_avg_period) + ' EMA'


# This method is used to calculate the Relative Strength Index of a Stock
# RSI is usually calculated in 14 day intervals
def rsi(data_frame):
    close_delta = data_frame['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    moving_average_up = up.ewm(com=14 - 1, adjust=True, min_periods=14).mean()
    moving_average_down = down.ewm(com=14 - 1, adjust=True, min_periods=14).mean()

    relative_strength_index = moving_average_up / moving_average_down
    relative_strength_index = 100 - (100 / (1 + relative_strength_index))
    return relative_strength_index


# This method will import data from yahoo finance and add/remove necessary columns
# THIS IS OUR DESCRIPTIVE METHOD AS IT PARSES/ADDS/CHANGES DATA
def import_data(stock_ticker):
    data_frame = data.DataReader(stock_ticker, 'yahoo', START, TODAY)
    data_frame.reset_index(inplace=True)
    data_frame.drop(['High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    data_frame['RSI'] = rsi(data_frame)
    data_frame[moving_avg_title] = data_frame['Close'].ewm(span=moving_avg_period, adjust=False).mean()
    return data_frame


# This is the start of creating our displayed data frame, adding the technical indicators, and making a prediction.
df = None
# Try to grab the stock data
try:
    df = import_data(stock)
    data_load_state.text("Data Loaded!")
# If we cannot grab stock data, then the program will request the user to try another ticker/stock name
except:
    data_load_state.text("Please Try Another Stock/Ticker")

try:
    # Print the current price of the stock
    cur_price = df['Close'][int(len(df) - 1)]
    st.subheader("Current Price: $" + str(round(cur_price, 2)))
    st.text("Read below for Next Business Day\'s close prediction and \nBuy/Sell/Neutral Recommendation")

    # MACHINE LEARNING SECTION #
    # THIS IS OUR PREDICTIVE METHOD. It is used to predict the next day's closing price
    # For this program, we will use a simple LinearRegression algorithm since it has broad uses and is easy to implement

    # Split our data and use 20% of the data to validate/test our model
    X_train, X_test, y_train, y_test = train_test_split(df[['Close']], df[[moving_avg_title]], test_size=20,
                                                        random_state=1)

    # The following will create the Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # To have consistent values in our data (ignore null values), we will reduce the amount of data
    # according to the size of the y_test array/model
    df = df.iloc[int(len(df)) - int(len(y_pred)):]
    # Add the new prediction values as a column to our table
    df['Predicted Next Day Close'] = y_pred
    df.reset_index(inplace=True)

    # The following lines of code will create a line graph that will contain the Closing Price, Selected Moving
    # Average, and our Predicted Close Values.
    # These will be our first visual type: Line Graphs
    st.subheader("Closing Price and " + moving_avg_title + " of " + stock)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df[moving_avg_title], name=moving_avg_title))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted Next Day Close'], name="Predicted Next Day Close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    fig.layout.update(width=900, height=800)
    st.plotly_chart(fig)

    # The following lines of code will create a line graph that will show our RSI values (trend of a stock)
    st.subheader("Relative Strength Index of " + stock)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
    fig2.add_hline(y=70, name='Overbought', line_color='Red')
    fig2.add_hline(y=30, name='Oversold', line_color='Red')
    fig2.layout.update(width=900, height=400)
    st.plotly_chart(fig2)

    # In this section, we aim to show the heat map/correlation between the main indicators and our stock price
    # This is our second visual type: Heat Map
    h1 = df.tail(moving_avg_period)[['Date', 'Close']]
    h2 = df.tail(moving_avg_period)[['RSI', moving_avg_title, 'Open']]
    var = pd.concat([h1, h2], axis=1, keys=['h1', 'h2']).corr().loc['h2', 'h1']
    st.subheader("Correlation Between Indicators/Open Price and Closing Price")
    fig3, ax = plt.subplots()
    heat_map = sns.heatmap(var, ax=ax, cmap="YlGnBu", annot=True)
    heat_map.set_xlabel("Closing Price")
    heat_map.set_ylabel("Indicators")
    st.write(fig3)

    # Print a table with the information for the last 2 weeks. We will make a new table and reset its index so that
    # the date in the left-most column
    # This is our third visual type: Table
    st.subheader('Past 14 Business Days')
    table = df[['Date', 'Open', 'Close', 'RSI', moving_avg_title, 'Predicted Next Day Close']]
    table.set_index('Date', inplace=True)
    st.table(table.tail(14))

    # Here we will display next business day's predicted closing price
    pred_close = df['Predicted Next Day Close'][int(len(df) - 1)]
    st.subheader("Next Business Day\'s Predicted Close: $" + str(round(pred_close, 2)))

    # This is our basic formula to tell the user to buy/sell a stock after getting all this information
    current_day = int(len(df) - 1)
    two_days_ago = int(len(df) - 3)
    today_rsi = float(df['RSI'][current_day])
    two_days_ago_rsi = float(df['RSI'][two_days_ago])
    next_predicted_close = float(df['Predicted Next Day Close'][current_day])
    last_close = float(df['Close'][current_day])

    # If today's RSI is greater than 2 days ago RSI, this shows a positive/upward trend. Our assumption is stronger if
    # our predicted close is greater than the last close. Finally, we call this a buy if RSI is LESS THAN 70.
    # When RSI is greater than 70, traders consider the stock to be OVERBOUGHT
    if (today_rsi > two_days_ago_rsi) and (next_predicted_close > last_close) and (today_rsi < 70):
        st.title('BUY STOCK')

    # If today's RSI is less than 2 days ago RSI, this shows a negative/downward trend. Our assumption is stronger if
    # our predicted value is less than the last close. Finally, we call this a sell if RSI is GREATER THAN 30. When
    # RSI is less than 30, traders consider the stock to be OVERSOLD.
    elif (today_rsi < two_days_ago_rsi) and (next_predicted_close < last_close) and (today_rsi > 30):
        st.title('SELL STOCK')
    # In all other conditions, tell the user to maintain a neutral position.
    else:
        st.title("Maintain Neutral Position")

    # These are the functions we use to determine the accuracy of our product
    st.subheader("Accuracy Metrics:")
    # This is our coefficient value between the closing price and selected moving average.
    st.write("Model Coefficients:", str(model.coef_[0][0]))
    # Ideally we want the Mean Absolute Error to be a low value.
    st.write("Mean Absolute Error:", str(mean_absolute_error(y_test, y_pred)))
    # Ideally we want the Coefficient of Determination to be close to 1.
    st.write("Coefficient of Determination:", str(r2_score(y_test, y_pred)))

except:
    # If our data cannot be loaded for any reason, we will print out the following message.
    data_load_state.text("Please Try Another Stock/Ticker")
