import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
print(df.head())
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig = model.plot(forecast)
# model.plot(df)
plt.show()

fig2 = model.plot_components(forecast)
plt.show()