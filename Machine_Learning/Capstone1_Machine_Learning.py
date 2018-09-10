
# coding: utf-8

# # Capstone Project 1 Machine Learning/Forecasting

# In[2]:


import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# ## Data Extraction
# 
# For this analysis, I want to take the mean housing price by zip code for each county and create a separate data set for each county. I then want to use the first 70% of each set and use it to train an ARIMA model. Instead of starting my model from January 2010, I'm going to start at January 2012 because the ARIMA model requires the data to be stationary. Starting from January 2010 makes it difficult to get the data to be stationary.

# In[3]:


# Read in dataset
df = pd.read_csv('../data/by_zip.csv', index_col = [0,1], header=[0])
df = df.loc[:,'2012-01-01':]
df.head()


# In[4]:


# Group by CountyName
df_grouped = df.groupby('CountyName').mean()
df_grouped.head()


# In[5]:


# Assign each CountyName to its own dataset
alameda = df_grouped.loc['Alameda',]
contra_costa = df_grouped.loc['Contra Costa']
marin = df_grouped.loc['Marin',]
napa = df_grouped.loc['Napa',]
santa_clara = df_grouped.loc['Santa Clara',]
san_fran = df_grouped.loc['San Francisco',]
san_mateo = df_grouped.loc['San Mateo',]
sonoma = df_grouped.loc['Sonoma',]
solano = df_grouped.loc['Solano',]


# In[6]:


# Convert indices of each county data set to datetime
county_list = [alameda, contra_costa, marin, napa, santa_clara, san_fran, san_mateo, sonoma, solano]
for county in county_list:
    county.index = pd.to_datetime(county.index)


# In[7]:


# Split each CountyName to train and test sets
alameda_train, alameda_test = alameda[:int(len(alameda)*0.7)], alameda[int(len(alameda)*0.7):]
cc_train, cc_test = contra_costa[:int(len(contra_costa)*0.7)], contra_costa[int(len(contra_costa)*0.7):]
marin_train, marin_test = marin[:int(len(marin)*0.7)], marin[int(len(marin)*0.7):]
napa_train, napa_test = napa[:int(len(napa)*0.7)], napa[int(len(napa)*0.7):]
sc_train, sc_test = santa_clara[:int(len(santa_clara)*0.7)], santa_clara[int(len(santa_clara)*0.7):]
sf_train, sf_test = san_fran[:int(len(san_fran)*0.7)], san_fran[int(len(san_fran)*0.7):]
sm_train, sm_test = san_mateo[:int(len(san_mateo)*0.7)], san_mateo[int(len(san_mateo)*0.7):]
sonoma_train, sonoma_test = sonoma[:int(len(sonoma)*0.7)], sonoma[int(len(sonoma)*0.7):]
solano_train, solano_test = solano[:int(len(solano)*0.7)], solano[int(len(solano)*0.7):]


# In[8]:


# Create a dictionary with county train/test sets indexed by county name
county_train_test = {'Alameda':[alameda_train, alameda_test, alameda],
                    'Contra Costa':[cc_train, cc_test, contra_costa],
                    'Marin':[marin_train, marin_test, marin],
                    'Napa':[napa_train, napa_test, napa],
                    'Santa Clara':[sc_train, sc_test, santa_clara],
                    'San Francisco':[sf_train, sf_test, san_fran],
                    'San Mateo':[sm_train, sm_test, san_mateo],
                    'Solano':[solano_train, solano_test, solano],
                    'Sonoma':[sonoma_train, sonoma_test, sonoma]}


# ## Data Visualization
# 
# Below are plots that show the overall trend of the housing price by Bay Area county. Note that the y-axis scales are all different, but they all show an upward trend with no apparent seasonal trend.

# In[9]:


# visualize the time series data
for county in county_train_test:
    plt.plot(county_train_test[county][0])
    plt.title(county)
    plt.xlabel('Time')
    plt.ylabel('Mean Housing Price')
    plt.show()


# ## Stationary Test
# 
# A condition necessary for the ARIMA model is that the data be stationary. A data set is said to be stationary if the following is true over time:
# 1. constant mean
# 2. constant variance
# 3. an autocovariance that doesn't depend on time
# 
# There are two methods to validate if a data set is considered stationary, visual inspection and the Dickey-Fuller test. The Dickey-Fuller test is a numerical method to validate a stationary dataset. It assumes a null hypothesis that the data set is not stationary and only with a predetermined significance level, the returning p-value can justify rejecting the null hypothesis. The two functions below can be used to confirm if a dataset is stationary.

# In[10]:


# Make the time series data stationary
from statsmodels.tsa.stattools import adfuller

def TestStationaryAdfuller(ts, cutoff = 0.05):
    ts_test = adfuller(ts, autolag = 'AIC')
    ts_test_output = pd.Series(ts_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in ts_test[4].items():
        ts_test_output['Critical Value (%s)'%key] = value
    print(ts_test_output)
    
    if ts_test[1] <= cutoff:
        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root, hence it is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[11]:


def TestStationaryPlot(ts, rolling_window = 12):
    rol_mean = ts.rolling(window = rolling_window, center = False).mean()
    rol_std = ts.rolling(window = rolling_window, center = False).std()
    
    plt.plot(ts, color = 'blue',label = 'Original Data')
    plt.plot(rol_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rol_std, color ='black', label = 'Rolling Std')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    
    plt.xlabel('Time', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.legend(loc='best', fontsize = 12)
    plt.show()


# ## Inducing a Stationary Dataset
# 
# The following methods can be used to imposing a data set to be stationary:
# 1. Moving Average
# 2. Exponentially weighted moving average
# 3. Differencing
# 4. Decomposing
# 
# Each method requires transforming the data to become stationary and each of the following functions takes a data set and transforms it according to one of the methods mentioned. It's also valid to transform the scale of the data such as taking the $log$, $sqrt$, etc.

# In[12]:


def moving_avg(dataset, window=12):
    rolling_avg = dataset.rolling(window).mean()
    diff = dataset - rolling_avg
    diff.dropna(inplace=True)
    TestStationaryPlot(diff)
    TestStationaryAdfuller(diff)
    return diff


# In[13]:


def weighted_moving_avg(dataset, hl=12):
    exp_weighted_avg = dataset.ewm(halflife=hl).mean()
    diff = dataset - exp_weighted_avg
    diff.dropna(inplace=True)
    TestStationaryPlot(diff)
    TestStationaryAdfuller(diff)
    return diff


# In[14]:


def differencing(dataset, s):
    diff = dataset - dataset.shift(s)
    diff.dropna(inplace=True)
    TestStationaryPlot(diff)
    TestStationaryAdfuller(diff)
    return diff


# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose

def decomposing(dataset):
    decomp = seasonal_decompose(dataset)
    
    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.resid
    
    plt.subplot(411)
    plt.plot(dataset, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    residual.dropna(inplace=True)
    TestStationaryPlot(residual)
    TestStationaryAdfuller(residual)
    
    return residual


# ## Plot ACF & PACF
# 
# The ARIMA model needs the following three parameters to fine-tune the model:
# + $p$: The number of auto-regressive (AR) terms
# + $d$: The number of non-seasonal differences
# + $q$: The number of moving-average (MA) terms.
# 
# $d$ can be found by determining how many differences are needed to take from the data to make it stationary. $p$ and $q$ could be determined by the autocorrelation function and partial autocorrelation function. The value at which these functions fall below a given significance level is their corresponding values for the ARIMA model. The user-defined function below plots both plots.

# In[16]:


# Plot the correlation and autocorrelation charts
from statsmodels.tsa.stattools import acf, pacf

def acf_pacf(data):
    lag_acf = acf(data, nlags=20)
    lag_pacf = pacf(data, nlags=20, method='ols')

    # Plot ACF
    plt.subplot(1, 2, 1)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y = -1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF
    plt.subplot(1,2,2)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y = -1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()


# ## Build ARIMA Model
# 
# After determining your parameters for you model, it's time to fit it. The following function takes a list all arguments needed to build an ARIMA model and fits it. It'll then return a plot of fitted values to the stationary dataset and the fitted model to perform forecasting.

# In[17]:


# Construct the ARIMA model function
def arima_summary(data, stationary, p, d, q):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary())
    
    residuals = model_fit.resid
    residuals_norm = (residuals - residuals.mean())/residuals.std()
    
    plt.subplot(2, 1, 1)
    plt.plot(stationary, color = 'blue', label='Stationary')
    plt.plot(model_fit.fittedvalues, color='red', label='Fitted Values')
    plt.legend(loc='best')
    plt.title('RSS: %.4f' % sum(model_fit.fittedvalues - stationary[0])**2)
    
    plt.subplot(2, 1, 2)
    residuals_norm.plot(kind='kde')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.show()
    
    print(residuals_norm.describe())
    
    return model_fit


# ## Forecasting

# When a model is created, it's important to see how well the model fits the training data and the testing data. The first function below calculated and plots the fitted values of the model and the original training data. The second plot the original testing data along with forecasted data predicted by the model. The model also calculates a 95% confidence interal of the forecasted data.

# In[18]:


# Create a function to make predictions on the model
def ARIMA_predictions(county, model):
    county_train = county_train_test[county][0]
    
    fitted_values = pd.DataFrame(np.exp(model.predict(typ='levels')), index = county_train.index, columns=['Predictions'])
    fitted_values = fitted_values.fillna(county_train[0])
    
    plt.plot(county_train, color='blue', label='Original')
    plt.plot(fitted_values, color = 'red', label='Predicted')
    plt.legend(loc='best')

    plt.title('RMSE: %.4f' %(np.sqrt(sum(fitted_values['Predictions'] - county_train)**2)/len(alameda_train)))
    plt.show()


# In[19]:


def forecast_score(county_name, predict_model, ts_steps=24):
    test_df = pd.DataFrame(county_train_test[county_name][1])
    
    model_forecast = predict_model.forecast(steps = ts_steps)
    model_forecast_values = np.exp(model_forecast[0])
    model_forecast_confid = np.exp(model_forecast[2])
    model_forecast_df = pd.DataFrame(model_forecast_values, index=test_df.index, columns=['Predictions'])
    model_forecast_ci_df = pd.DataFrame(model_forecast_confid, index=test_df.index, columns=['Lower', 'Upper'])
    
    fig, ax = plt.subplots()
    plt.plot(county_train_test[county_name][2], color='blue', label='Test Data')
    plt.plot(model_forecast_df, color='red', label='Predictions')
    ax.fill_between(model_forecast_ci_df.index,
                    model_forecast_ci_df.iloc[:, 0],
                    model_forecast_ci_df.iloc[:, 1], color='r', alpha=.5)
    plt.legend(loc='best')
    plt.title('%s Test vs. Predictions' %(county_name))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
    forecast_rsme = np.sqrt(mean_squared_error(test_df, model_forecast_df))
    print('Test RSME: %.4f' %(forecast_rsme))


# ## Bay Area Housing Analysis

# Now that all the functions necessary for this analysis have been created, it's time to apply them to the Bay Area housing data. I first wanted to see how stationary each of the county data sets were (see below).

# In[20]:


for county in county_train_test:
    plt.title("%s Rolling Mean & Standard Deviation" %(county), fontsize = 12)
    TestStationaryPlot(county_train_test[county][0], rolling_window=12)


# In[21]:


for county in county_train_test:
    print(county)
    TestStationaryAdfuller(county_train_test[county][0])


# According to the Dickey-Fuller tests p-values above, none of the county data sets could be considered stationary with an alpha of $0.05$. Each will have to be transformed to impose the data set to be stationary before they could be analyzed.

# ### Alameda County

# In[22]:


# Take log of data set
alameda_train_log = np.log(alameda_train)


# In[23]:


alameda_ma = moving_avg(alameda_train_log, window=12)


# In[24]:


alameda_wma = weighted_moving_avg(alameda_train_log, hl=6)


# In[25]:


alameda_train_log_diff = differencing(alameda_train_log, 1)


# In[26]:


alameda_train_resid = decomposing(alameda_train_log)


# In[27]:


acf_pacf(alameda_train_log_diff)


# In[28]:


alameda_train_p = 3
alameda_train_q = 2


# In[29]:


alameda_model_fit = arima_summary(alameda_train_log, alameda_train_log_diff, alameda_train_p, 1, alameda_train_q)


# In[30]:


ARIMA_predictions('Alameda', alameda_model_fit)


# In[31]:


forecast_score('Alameda', alameda_model_fit)


# ### Contra Costa County

# In[32]:


cc_train_log = np.log(cc_train)


# In[33]:


cc_ma = moving_avg(cc_train_log, window=12)


# In[34]:


cc_wma = weighted_moving_avg(cc_train_log, hl=12)


# In[35]:


cc_train_log_diff = differencing(cc_train_log, 1)


# In[36]:


cc_train_resid = decomposing(cc_train_log)


# In[37]:


acf_pacf(cc_wma)


# In[38]:


cc_train_p=9
cc_train_q =2


# In[39]:


cc_model_fit = arima_summary(cc_train_log, cc_wma, cc_train_p, 1, cc_train_q)


# In[40]:


ARIMA_predictions('Contra Costa', cc_model_fit)


# In[41]:


forecast_score('Contra Costa', cc_model_fit)


# ### Marin County

# In[42]:


marin_train_log = np.log(marin_train)


# In[43]:


marin_ma = moving_avg(marin_train_log, window=12)


# In[44]:


marin_wma = weighted_moving_avg(marin_train_log, hl=4)


# In[45]:


marin_train_log_diff = differencing(marin_train_log, 1)


# In[46]:


marin_train_resid = decomposing(marin_train_log)


# In[47]:


acf_pacf(marin_train_resid)


# In[48]:


marin_train_p = 4
marin_train_q = 2


# In[118]:


marin_model_fit = arima_summary(marin_train_log, marin_train_resid, marin_train_p, 2, marin_train_q)


# In[119]:


ARIMA_predictions('Marin', marin_model_fit)


# In[120]:


forecast_score('Marin', marin_model_fit)


# ### Napa County

# In[52]:


napa_train_log = np.log(napa_train)


# In[53]:


napa_ma = moving_avg(napa_train_log, window=2)


# In[54]:


napa_wma = weighted_moving_avg(napa_train_log, hl=12)


# In[55]:


napa_train_log_diff = differencing(napa_train_log, 1)


# In[56]:


napa_train_resid = decomposing(napa_train_log)


# In[57]:


acf_pacf(napa_wma)


# In[58]:


napa_train_p = 7
napa_train_q = 2


# In[121]:


napa_model_fit = arima_summary(napa_train_log, napa_wma, napa_train_p, 1, napa_train_q)


# In[122]:


ARIMA_predictions('Napa', napa_model_fit)


# In[123]:


forecast_score('Napa', napa_model_fit)


# ### San Francisco County

# In[62]:


sf_train_log = np.log(sf_train)


# In[63]:


sf_ma = moving_avg(sf_train_log, window=12)


# In[64]:


sf_wma = weighted_moving_avg(sf_train_log, hl=12)


# In[65]:


sf_train_log_diff = differencing(sf_train_log, 12)


# In[66]:


sf_train_resid = decomposing(sf_train_log)


# In[67]:


acf_pacf(sf_train_resid)


# In[68]:


sf_train_p = 2
sf_train_q = 2


# In[69]:


sf_model_fit = arima_summary(sf_train_log, sf_train_resid, sf_train_p, 2, sf_train_q)


# In[70]:


ARIMA_predictions('San Francisco', sf_model_fit)


# In[71]:


forecast_score('San Francisco', sf_model_fit)


# ### San Mateo County

# In[72]:


sm_train_log = np.log(sm_train)


# In[73]:


sm_ma = moving_avg(sm_train_log, window=2)


# In[74]:


sm_wma = weighted_moving_avg(sm_train_log, hl=6)


# In[75]:


sm_train_log_diff = differencing(sm_train_log, 1)


# In[76]:


sm_train_resid = decomposing(sm_train_log)


# In[77]:


acf_pacf(sm_train_resid)


# In[78]:


sm_train_p=3
sm_train_q=2


# In[124]:


sm_model_fit = arima_summary(sm_train_log, sm_train_resid, sm_train_p, 2, sm_train_q)


# In[80]:


ARIMA_predictions('San Mateo', sm_model_fit)


# In[81]:


forecast_score('San Mateo', sm_model_fit)


# ### Santa Clara County

# In[82]:


sc_train_log = np.log(sc_train)


# In[83]:


sc_ma = moving_avg(sc_train_log, window=12)


# In[84]:


sc_wma = weighted_moving_avg(sc_train_log, hl=12)


# In[85]:


sc_train_log_diff = differencing(sc_train_log, 12)


# In[86]:


sc_train_resid = decomposing(sc_train_log)


# In[87]:


sc_double_diff = differencing(sc_train_log_diff, 1)


# In[88]:


acf_pacf(sc_double_diff)


# In[89]:


sc_train_p=3
sc_train_q=2


# In[90]:


sc_model_fit = arima_summary(sc_train_log, sc_double_diff, sc_train_p, 2, sc_train_q)


# In[91]:


ARIMA_predictions('Santa Clara', sc_model_fit)


# In[92]:


forecast_score('Santa Clara', sc_model_fit)


# ### Solano County

# In[93]:


solano_train_log = np.log(solano_train)


# In[94]:


solano_ma = moving_avg(solano_train_log, window=12)


# In[95]:


solano_wma = weighted_moving_avg(solano_train_log, hl=12)


# In[96]:


solano_train_log_diff = differencing(solano_train_log, 12)


# In[97]:


solano_train_resid = decomposing(solano_train_log)


# In[98]:


solano_double_diff = differencing(solano_train_log_diff, 1)


# In[99]:


acf_pacf(solano_double_diff)


# In[104]:


solano_train_p = 5
solano_train_q = 2


# In[105]:


solano_model_fit = arima_summary(solano_train_log, solano_double_diff, solano_train_p, 2, solano_train_q)


# In[106]:


ARIMA_predictions('Solano', solano_model_fit)


# In[107]:


forecast_score('Solano', solano_model_fit)


# ### Sonoma County

# In[108]:


sonoma_train_log = np.log(sonoma_train)


# In[109]:


sonoma_ma = moving_avg(sonoma_train_log, window=12)


# In[110]:


sonoma_wma = weighted_moving_avg(sonoma_train_log, hl=12)


# In[111]:


sonoma_train_log_diff = differencing(sonoma_train_log, 12)


# In[112]:


sonoma_train_resid = decomposing(sonoma_train_log)


# In[113]:


acf_pacf(sonoma_train_resid)


# In[114]:


sonoma_train_p=3
sonoma_train_q=2


# In[115]:


sonoma_model_fit = arima_summary(sonoma_train_log, sonoma_train_resid, sonoma_train_p, 2, sonoma_train_q)


# In[116]:


ARIMA_predictions('Sonoma', sonoma_model_fit)


# In[117]:


forecast_score('Sonoma', sonoma_model_fit)

