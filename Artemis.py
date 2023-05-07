import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

#user input tickers of stocks to predict in an array
symbols = input("Enter the stock tickers you want to predict separated by a space: ").split()


start_date = datetime.today() - timedelta(days=365)
end_date = datetime.today()

for symbol in symbols:
    #fetch data from yfinance
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    #next day's closing price
    stock_data['Next Day Close'] = stock_data['Close'].shift(-1)

    #next week's closing price
    stock_data['Next Week Close'] = stock_data['Close'].shift(-5)
    stock_data.dropna(inplace=True)


    
    features = stock_data.drop(['Next Day Close', 'Next Week Close'], axis=1)
    next_day_target = stock_data['Next Day Close']
    next_week_target = stock_data['Next Week Close']

    #linear regression model on the data
    model = LinearRegression()
    model.fit(features, next_day_target)

    #printing next day's predictions
    next_day_features = features.tail(1)
    next_day_date = end_date + timedelta(days=1)
    next_day_prediction = model.predict(next_day_features)[0]
    print(f"The predicted closing price of {symbol} for {next_day_date.strftime('%Y-%m-%d')} is ${next_day_prediction:.2f}")

    #printing next week's predictions
    model.fit(features, next_week_target)
    next_week_features = features.tail(5)
    next_week_date = end_date + timedelta(days=7)
    next_week_prediction = model.predict(next_week_features)[0]
    print(f"The predicted closing price of {symbol} for {next_week_date.strftime('%Y-%m-%d')} is ${next_week_prediction:.2f}")

    print()
