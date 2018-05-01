+++
date = "2018-02-09T17:00:00+09:00"
title = "Time Series Analysis"
thumbnail = "images/thumbnail.jpg" # Optional, referenced at `$HUGO_ROOT/static/images/thumbnail.jpg`
categories = [
  "Time series",  
  "Data Science"
]
tags  = ['ML', 'AR', 'ARMA', 'ARIMA']
toc = true # Optional
+++

### Time Series Analysis and Predictive Data Modeling 
-------------------------------

This post is about time series and using [AR](http://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AR.html) and [ARMA](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARMA.html?highlight=arma) models Im going to try and predict the next few days price for... Bitcoin! Woohoo!<br/> Would be interesting to see how these models can predict the price of this rather unpredictable entity :)        
So first things first, we need to get the data. You can download daily closing price of all major crypto-currencies from [here](https://www.coindesk.com/price/). Click on the 'Export' button on the top right hand side of the chart and you can download the data as a CSV file.

All the following code and data is also available on my [github](https://github.com/jessgess/Time_Series_Analysis_ARIMA).

Now let's load the data and have a look at it:

	data = pd.read_csv('data/bitcoin_USD_close_data-2017-02-09_2018-02-09.csv',
						parse_dates=['Date'])
	print(data.shape)
	data.tail() 

(366, 2)

 | Date |	Price |
--- | --- | --- 
361	| 2018-02-05 00:00:00 | 6914.26
362	| 2018-02-06 00:00:00 | 7700.39
363	| 2018-02-07 00:00:00 | 7581.8
364	| 2018-02-08 00:00:00 | 8237.24
365	| 2018-02-09 00:00:00 | 8689.84

<br/>	
We can format the `Date` column and use its unique values as index.

	data['Date'] = data['Date'].dt.date
	data.set_index(pd.to_datetime(data['Date']), inplace=True)
	data.drop('Date', axis=1, inplace=True)
	data.head()

 | Price |
--- | ---
2017-02-09 | 988.95
2017-02-10 | 993.08
2017-02-11 | 1010
2017-02-12 | 999.51
2017-02-13 | 996.86
<br/>
Next thing plotting the data to get the general feel of our series.

	data.plot(y='Price', color='teal', figsize=(16,3))
	
![bitcoin price](/images/output_1.png)
	
Its clear from the plot that there was a constant upward trend all the way to mid December and then a sharp drop in the price followed by a spark earlier in January, but in general price was going down in 2018.<br/> 
To better capture the trend of data you can always resample the data by taking biweekly, weekly, monthly or yearly average.  

	price_weekly_avg = data.resample('W').apply(np.mean)
	price_monthly_avg = data.resample('M').apply(np.mean)
	plt.figure(figsize=(15,3))
	plt.subplot(121)
	plt.title('Weekly Avg')
	plt.plot(price_weekly_avg.Price, "-o", markersize=3, color='teal')
	plt.subplot(122)
	plt.title('Monthly Avg')
	plt.plot(price_monthly_avg.Price, "-o", markersize=3, color='teal')
![price average](/images/output_2.png)

Checking the above graph for monthly average, we can see that it confirms our observation earlier on about the trend of the data. Another way of capturing the trend is through rolling average which we will do below. Basically we take a window of consecutive data points, in our case this is a daily frequency, calculate the average of them, and replace the whole window by that average, either at the extreme right or at the center of the window. So window of 7 is the same as weekly average: 

	rolling_mean = data.Price.rolling(window=7, center=False).mean() 
	plt.plot(data.Price, color='teal')
	plt.plot(rolling_mean, 'red') 
![rolling average](/images/output_3.png)

<br/>
### Making the data stationary and autocorrelation 
After some initial explorations its now time to create our model. For that we first need to make the series stationary as this is a prerequisite for most of the models.
Making the data stationary means taking the trend out of it (de-trending the data) in order to have its statistical properties (i.e. mean, variance) constant over time. This will help the sample to be more predictable by the model since it can be assumed that the statistical properties of the data will be the same in the future as they were in the past. 
One way of stationarising a time series is through differencing, that is taking the difference of two data points within a specified period, this period is called lag.  
In order to find the optimal lag we use [autocorerlation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.autocorr.html) method. 

Autocorrelation measures the correlation (similarity) between the time series and a lagged version of itself. Below we define varies lag values to see which one causes the highest correlation.

	lags = [7, 8, 10, 14, 17, 28, 30, 60, 90, 180] 
	autocorrs = [data.Price.autocorr(lag=lag) for lag in lags]
	plt.stem(lags, autocorrs)
	plt.xlabel("Lag")
	plt.ylabel("Autocorrelation")
![autocorr](/images/output_4.png)

Based on the above graph the highest correlation occurs at lag of 7, that means the data repeats a pattern on a weekly basis. This is somewhat true of Bitcoin price as by experience I have seen the price going up throughout the week, normally hitting the highest point on the weekend then coming back down on Monday and repeating more or less the same cycle again...

Another way of checking the autocorrelation is through Pandas tools library. Running that we can see positive correlation for the first 100 lags with the most significant ones being the first 10 ones.  
	
	from pandas.tools.plotting import autocorrelation_plot

	autocorrelation_plot(data) 	
![autocorr2](/images/output_5.png)

We can conclude that lag of 7 is a good candidate for the optimal lag. Let's plot the original series vs the lagged version of itself with difference of 7. This means each data point (starting from the 7th one) will be deducted from the data 7 days before that and plotted on the chart (orange line below).	

	plt.plot(data.Price, '-o', color='teal', label="original ts")
	plt.plot(data.Price.diff(7), '-o', color='orange', label="differenced ts (lag=7)") 
![autocorr2](/images/output_6.png)

Now that we know what optimal lag to use to make a relatively stationary version of our original time series we can move on to next step which is training and fitting the models.   
<br/>
### Developing the models 

To start, lets save the stationary data into a new variable and call it `time_series`:

	prices = data.Price
	price_diff = prices.diff(7).dropna()
	time_series = price_diff


<br/>**AR Model**<br/>

AR stands for Auto-Regressive time series model where values are modeled as a linear combination of the _p_ past values therefore it's indicated as **AR(p)**. <br/>
_p_ defines the number of past observations that we would want the model to use to predict the next data and is also known as the lag order.
 
We can use `select_order` method to find the optimal value for _p_ as below, have in mind you will need to define `maxlag` which is the highest lag length to be tried out. **ic** is the criterion used for selecting the optimal lag, you can use different values for it defined in the [docs] (http://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AR.fit.html#statsmodels.tsa.ar_model.AR.fit) and see how those perform. I found 'aic' better for this task. 
	
	maxlag=91	
	ar = tsa.AR(time_series)
	optlag = ar.select_order(axlag, ic='aic') 
	print("Optimal lag for AR model: {}".format(optlag))
Optimal lag for AR model: 47<br/>

Optimal _p_ value seems to be 47 days, having that we can train our model and see how it performs using mean absolute error (MAE) or mean of the residuals which is calculated as below, where _y_ and _x_ are predicted versus actual values per observation:

<center><img src="https://latex.codecogs.com/svg.latex?\Large&space;MAE=\frac{\sum_{i=1}^{n}|y_i \_ x_i|}{n}"/></center>


I am also printing out MAE_2 which is the mean absolute error only for the last 90 predictions as a fixed set to compare the results of this model with ARMA model later on.  

	lag = 47 
	ar = tsa.AR(time_series)
	ar_fit = ar.fit(maxlag=lag, ic='aic')
	ar_prediction = ar_fit.predict(start=lag)
	
	plt.plot(time_series, '-o', label='true', color='teal')
	plt.plot(ar_prediction, '-o', label='model', color='orange')
	
	print('MAE = {0:.3f}'.format(mean_absolute_error(time_series[lag:], ar_prediction)))
	print('MAE2 = {0:.3f}'.format(mean_absolute_error(time_series[-90:], ar_prediction[-90:]))) 
	
MAE = 258.940<br/>
MAE2 = 509.166<br/>
 
![AR_model](/images/output_9.png)

<br/>
**ARMA Model**<br/>	

ARMA stands for Autoregressive Moving Average. The difference between ARMA and AR is that the latter models a point in the time series as a linear model of the previous values, and treats the error between the actual data and the predictions as noise. But what if there could still be information in the series of errors? ARMA on the other hand includes the past errors as additional features to the model and is defined as **ARMA(p,q)**.

Just as the name suggests ARMA is comprised of two models, AR(p) (just like above) and MA(q) which is the moving average part and _q_ defines the number of moving average terms.<br/> Let's now run the model with p=7 and q=7 and see the results. You should also play around with these two variables and see the different results that they produce. 

	arma = tsa.ARMA(time_series, order=(7, 7)) 
	arma_fit = arma.fit()
	start=100
	arma_prediction = arma_fit.predict(start=start)
	
	plt.plot(time_series, '-o', label='true', color='teal')
	plt.plot(arma_prediction, '-o', label='model', color='orange')
	plt.legend();
	
	print('MAE = {0:.3f}'.format(mean_absolute_error(time_series[start:], arma_prediction)))
	print('MAE2 = {0:.3f}'.format(mean_absolute_error(time_series[-90:], arma_prediction[-90:]))) 
	
MAE = 319.051<br/>
MAE2 = 647.323

![ARMA_model](/images/output_10.png)

Comparing the results for AR and ARMA, AR is producing a slightly lower error on the last 90 predictions (our MAE2 measure). 

<br/>
### Out of sample predictions
So far we trained the models on the existing data and evaluated their performance by checking the mean absolute error of the predictions vs the actual values. Now that we have two trained models lets use them and predict the future (out of sample) values. 

Given that our AR model produced lower error, I will use that to predict the future. But you can follow the same steps below for ARMA model and see how that predicts the future values. I have published the complete code for both models as well as ARIMA on [github](https://github.com/jessgess/Time_Series_Analysis_ARIMA). 

Using the same optimal lag of 47, we predict 7 out of sample values. `end` argument of `predict` function, is where we define index of the last prediction we want to predict. This for 7 out of sample predictions will be the length of the series + 6.  

	lag = 47
	ar = tsa.AR(time_series)
	ar_fit = ar.fit(maxlag=lag)
	ar_forecast = ar_fit.predict(end=len(time_series)+6)[-7:]
	print(ar_forecast)

2018-02-10 &nbsp; -654.683766<br/>
2018-02-11 &nbsp; -116.566214<br/>
2018-02-12 &nbsp; 1158.861627<br/>
2018-02-13 &nbsp; 1209.549713<br/>
2018-02-14 &nbsp; 2028.570897<br/>
2018-02-15 &nbsp; 1947.398843<br/>
2018-02-16 &nbsp; 2259.390295

<br/>
Let's plot the above predictions on the price chart:

	plt.plot(time_series, '-o', label="original data", color='teal')
	plt.plot(ar_forecast, '--o', label='prediction', color='orange')
![AR_future](/images/output_11.png)

What we have predicted is not the actual future prices but the differenced values. Remember initially we made the data stationary by differencing with lag of 7? Now in order to find the actual future prices we need to reverse that process and change the series from stationary to the original form. For that we add each predicted value to the actual value 7 days before that. The last 7 days values in the original series were: 

	print(prices[-7:])

Date |  |
--- | --- | 
2018-02-03 | 9224.39
2018-02-04 | 8186.65
2018-02-05 | 6914.26
2018-02-06 | 7700.39
2018-02-07 | 7581.80
2018-02-08 | 8237.24
2018-02-09 | 8689.84

Having that, we calculate the actual values for the 7 out of sample predictions and plot them with the rest of the series:

	idx = ar_forecast.index.values
	
	forecast_prices = []
	lag = 7
	for i, diff in enumerate(ar_forecast):  
	    prev_value = prices[-(lag)+i:][0]
	    forecast_prices.append(prev_value+diff)
	    
	bitcoin_forecast = pd.Series(forecast_prices, index=idx)
	print(bitcoin_forecast)
	
	hist_values = data['Price'].append(bitcoin_forecast)
	plt.plot(hist_values, '-o', color='teal', alpha=0.5)
	plt.show() 

2018-02-10 &nbsp; 8569.706234<br/>
2018-02-11 &nbsp; 8070.083786<br/>
2018-02-12 &nbsp; 8073.121627<br/>
2018-02-13 &nbsp; 8909.939713<br/>
2018-02-14 &nbsp; 9610.370897<br/>
2018-02-15 &nbsp; 10184.638843<br/>
2018-02-16 &nbsp; 10949.230295   
![pred_vs_real](/images/output_14.png)

<br/>
Here we go, our predictions for the future! Though Im not sure if Im going to trust those numbers %100 and start buying or selling Bitcoin purely based on the above :) since we know how unpredictable Bitcoin actually is. But still I will check the actual prices in a week time to see how close or far from reality our predictor worked... 

<br/>
___
___
___

						
**FUTURE Reality Check**

It's a week since I ran the model and below are the actual Bitcoin closing prices for the dates we made predictions for. These are again extracted form [Coindesk](https://www.coindesk.com/price/): 

2/10/2018 0:00	&nbsp; 8556.61<br/>
2/11/2018 0:00	&nbsp; 8070.8<br/>
2/12/2018 0:00	&nbsp; 8891.21<br/>
2/13/2018 0:00	&nbsp; 8516.24<br/>
2/14/2018 0:00	&nbsp; 9477.84<br/>
2/15/2018 0:00	&nbsp; 10016.49<br/>
2/16/2018 0:00	&nbsp; 10178.71

Comparing the actual values with our predictions, the model hasn't performed that far off from the reality!  Below plot shows the predicted vs actual closing prices:

<br/>![pred_vs_real](/images/output_15.png)


Looking at the plot, the first two values, for the 10th and 11th, are spot on, and the rest except for the third and the last are also pretty close.<br/> 
One thing you need to have in mind is that the further in the future you predict the more uncertainty there is and therefore the more difficult the prediction task becomes. This is not just because of the unknown factors but also because you include more errors (residuals) in the predictions, and we can see this has contributed to the value of the very last prediction.<br/> 
To conclude, predicting the future is not an easy task and there is always something that you could have not seen coming. On top of that in the case of Bitcoin, price changes on hourly, minutely or even secondly basis. So there is always the question of which trade price you want to predict. 

In another post I will use Neural Nets to do the same task and will use the results in this post as a baseline for comparison. 