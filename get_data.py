import pytrends # https://github.com/GeneralMills/pytrends PYPI?: YES
import time
import numpy as np
import itertools
import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
import quandl # https://github.com/quandl/quandl-python PYPI?: YES
# NOTE: quandl depends on SSL headers.
# On Ubuntu: sudo apt-get install libssl-dev
import matplotlib


# Some utility functions
def MAPE(x1,x2):
  """Compute MAPE for test data x1 and estimator x2"""
  return 1/len(x1)*np.sum(np.abs((x1-x2)/x1))*100
def MSE(x1,x2):
  """Compute MSE for test data x1 and estimator x2"""
  return 1/len(x1)*np.sum((x1-x2)**2)
def ARV(x1,x2):
  """Compute ARV for test data x1 and estimator x2"""
  return np.sum((x1-x2)**2)/np.sum((x2.mean()-x2)**2)

# Enter Google account credentials here
# NOTE: Disable 2FA to login, otherwise HTTP 400
USER = "XXX@gmail.com"
PASS = "*****"
print("HEY! DON'T FORGET TO TURN 2FA BACK ON!")
time.sleep(0.1)
print("-----------------------------------------------------")

# Login to account with the script
conn = pytrends.pyGTrends.pyGTrends(USER,PASS)

# Years with Bitcoin search activity
years=[i for i in range(2009,2017)]

# In order to get daily data must limit ourselves to 90 day requests
months = [1,4,7,10]

# Get US, CN (China), DE (Germany), IN (India), BR (Brazil)
countries = ["US","CN","DE","IN","BR"]

# Save raw CSV data, parsing required later
for c in countries:
  dates = []
  SVI = []
  for y in years:
    for m in months:
      if y == 2016 and (m == 7 or m == 10): # No data for Jul-Dec of this year
        continue
      else:
        # Make request for specific daily trend data
        #req = conn.request_report('bitcoin',geo=c,date="%02d/%i 3m" %(m,y))
        # Short break to prevent spamming
        time.sleep(0.5)
        # Save quarter trend to CSV file
        conn.save_csv('./','%s_%i_%i' %(c,m,y))
        # Short break to prevent spamming
        time.sleep(2)
        # Load file for parsing
        if m == 1:
          mr = 90
        elif m == 4:
          mr = 91
        else:
          mr = 92
        try:
          # Horrible csv parsing incoming, take cover
          dates.append(list(np.genfromtxt('%s_%i_%i.csv' %(c,m,y),
            skip_header=5,max_rows=mr,delimiter=',',dtype=str,usecols=(0,))))
          SVI.append(list(np.genfromtxt('%s_%i_%i.csv' %(c,m,y),
            skip_header=5,max_rows=mr,delimiter=',',dtype=str,usecols=(1,))))
        except:
          # No trends for that time period
          print("No trends available for %s during %02d-%i" %(c,m,y))
  dates = list(itertools.chain(*dates))
  SVI = list(itertools.chain(*SVI))
  with open('%s_trends.txt' %c, 'w') as f:
    for i in range(len(dates)):
      f.write(dates[i]+'   '+SVI[i]+'\n')

print("GO TURN 2FA BACK ON!")
time.sleep(5)
print("NOT KIDDING. DO IT NOW.")

# Get blockchain data using Quandl
# NOTE: THEIR API IS NICE!!!!! (AND FREE)
# Going to start analysis on Jun 1st 2010 since 
# trends from all countries are available

quandl.ApiConfig.api_key = "******"
# This means we have 2192 samples
# And of course, Blockchain is missing some days,
# so this needs to be handled.
# I will just interpolate to get the missing value

tend = datetime.datetime(2016,6,30)
tstart = datetime.datetime(2010,7,1)
delta_time = tend - tstart
expected_times = np.empty((2192),dtype=object)

for i in range(delta_time.days + 1):
  expected_times[i] = tstart + i*datetime.timedelta(days=1)

# I'm going to use Blockchain.info's data set

# This provides 33 variables of which we'll use 15 as inputs
# plus the 5 Google Trends sets

dvars = ['MKPRU','DIFF','TOTBC','MKTCP','TRFEE','TRFUS','NETDF','NTRAN',
  'NADDU','TOUTV','ETRAV','ETRVU','TRVOU','CPTRA','HRATE','MIREV']
  
X = np.zeros((2192,21))
xcol = 0

for dset in dvars:
  quandl_data = quandl.get("BCHAIN/%s" %dset,returns="numpy",
  start_date="2010-07-01",end_date="2016-06-30",collapse="daily")
  date_array = quandl_data['Date']
  val_array = quandl_data['Value']
  # Must deal with missing days
  if len(date_array) != 2192:
    # Go through list, and interpolate missing values
    for i in range(2192):
      if expected_times[i] not in date_array:
        # Found the bastard
        date_array = np.insert(date_array,i,expected_times[i])
        avg_val = (val_array[i+1] + val_array[i]) / 2
        val_array = np.insert(val_array,i,avg_val)
      else:
        continue
  X[:,xcol] = val_array
  xcol += 1
  time.sleep(2)
  
# Load up Google Trends data, do the same cleaning
for c in countries:
  date_str= np.genfromtxt('%s_trends.txt' %c,usecols=(0,),dtype=str)
  Nd = len(date_str)
  date_array = np.empty(Nd,dtype=object)
  val_array = np.loadtxt('%s_trends.txt' %c,usecols=(1,))
  for i in range(Nd):
    date_array[i] = datetime.datetime.strptime(date_str[i], '%Y-%m-%d')
  val_array = val_array[date_array>=tstart]
  date_array = date_array[date_array>=tstart]
  # Must deal with missing days
  if len(date_array) != 2192:
    # Go through list, and interpolate missing values
    for i in range(2192):
      if expected_times[i] not in date_array:
        # Found the bastard
        date_array = np.insert(date_array,i,expected_times[i])
        avg_val = (val_array[i+1] + val_array[i]) / 2
        val_array = np.insert(val_array,i,avg_val)
      else:
        continue
  X[:,xcol] = val_array
  xcol += 1

# Exploratory
plt.matshow(np.corrcoef(X,rowvar=0))
# Degenerate correlations: 3,6,15
# These will be removed
degen = np.ones(21,dtype=bool)
degen[[3,6,15]] = False

X1 = X[:,degen]

# Separate inputs

X2 = X1[:,1:]
Y = X1[:,0]

# Standardize inputs
for i in range(len(X2[0,:])):
  X2[:,i] = (X2[:,i]-X2[:,i].mean())/X2[:,i].std()

# In the exploratory plot it seemed that BR and IN were more
# correlated with the price than expected countries like US and CN

# I'd like to know if any lag periods are also correlated, which
# might be helpful in forecasting

lag_times = np.arange(1,201)
corr_coeff_country_lag = np.zeros((200,5))
for i in lag_times:
  X_lag = np.empty((2192-i,len(X1[0,:])))
  X_lag[:,0] = X1[i:,0]
  X_lag[:,1:] = X1[:-i,1:]
  cc = np.corrcoef(X_lag,rowvar=0)[0][-5:]
  corr_coeff_country_lag[i-1,:] = cc

for i in range(5):
  plt.plot(lag_times,corr_coeff_country_lag[:,i],label=countries[i])
  
plt.xlabel('Lag time (days)')
plt.ylabel(r'$\rho_{\mathrm{MKPRU,GT}}$')
plt.legend()
plt.show()

X2_lag = X2[:-10,:]
Y_lag = Y[10:]
Y_lag = Y_lag.reshape(2192-10,1)

# Can we use a ANN model here?

# Nice tutorial by http://iamtrask.github.io/2015/07/12/basic-python-network/
# Following code is taken from that link
def nonlin(x,deriv=False):
  if(deriv==True):
    return x*(1-x)
  return 1/(1+np.exp(-x))
np.random.seed(4334)

# randomly initialize our weights with mean 0
syn0 = 20*np.random.random((17,2192-10)) - 1
syn1 = 10*np.random.random((2192-10,1)) - 1

for j in range(60000):
# Feed forward through layers 0, 1, and 2
  l0 = X2_lag
  l1 = nonlin(np.dot(l0,syn0))
  l2 = nonlin(np.dot(l1,syn1))

  # how much did we miss the target value?
  l2_error = Y_lag - l2

  if (j% 50) == 0:
    print("Error:" + str(np.mean(np.abs(l2_error))))

  # in what direction is the target value?
  # were we really sure? if so, don't change too much.
  l2_delta = l2_error*nonlin(l2,deriv=True)

  # how much did each l1 value contribute to the l2 error (according to the weights)?
  l1_error = l2_delta.dot(syn1.T)

  # in what direction is the target l1?
  # were we really sure? if so, don't change too much.
  l1_delta = l1_error * nonlin(l1,deriv=True)

  syn1 += l1.T.dot(l2_delta)
  syn0 += l0.T.dot(l1_delta)

# Unfortunately this converges too slowly to be useful

# Next let's try a lagged linear model
# Start with ridge regression function from sklearn

clf = linear_model.Ridge(alpha=0.25)
clf.fit(X2_lag,Y_lag)

# Let's see how we did
# Used a 10 day lag time, so last 10 values from inputs
# should be able to predict price starting in July
quandl_data = quandl.get("BCHAIN/MKPRU",returns="numpy",
  start_date="2016-07-01",end_date="2016-07-18",collapse="daily")
val_array = quandl_data['Value']
price_est = (clf.coef_.dot(X2_lag[-10:,:].T)+clf.intercept_[0])[0]

plt.plot(val_array[:10],label='Actual price')
plt.plot(price_est,label='Price estimator')
plt.show()

# Not too.. great, but not horrible, how about the performance with lag time?

lag_times = np.arange(1,18)
err_stats = np.zeros((18,3))
for i in lag_times:
  X2_lag = X2[:-i,:]
  Y_lag = Y[i:]
  Y_lag = Y_lag.reshape(len(Y_lag),1)
  clf = linear_model.Ridge(alpha=0.25)
  clf.fit(X2_lag,Y_lag)
  price_est = (clf.coef_.dot(X2_lag[-i:,:].T)+clf.intercept_[0])[0]
  real_price = val_array[:i]
  err_stats[i,0] = MSE(price_est,real_price)
  err_stats[i,1] = MAPE(price_est,real_price)
  err_stats[i,2] = ARV(price_est,real_price)

labels=['RMSE','RMAPE','RARV']
for i in range(3):
  plt.plot(lag_times,np.sqrt(err_stats[1:,i]),label=labels[i])
  plt.xlabel('Lag time (days)')
  plt.ylabel('Error')
  plt.legend()
  plt.show()

# Prediction of July data
ta=quandl_data['Date']
dts=matplotlib.dates.date2num(ta)
plt.plot_date(dts[1:],real_price,label='Actual price',ls='solid')
plt.plot_date(dts[1:],price_est,label='Price estimator',ls='dashed')
plt.legend()
plt.ylabel('Price of 1 BTC in US Dollars')
plt.gcf().autofmt_xdate()

