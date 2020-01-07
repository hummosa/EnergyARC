#%%
# Import yfinance
import yfinance as yf  
 
# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
data = yf.download('DJIA','2010-01-01','2019-12-20')
 
# Plot the close prices
import matplotlib.pyplot as plt
# data.plot()


# %%

data.Close.plot()
plt.show()

# %%
