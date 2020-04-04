import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt



url = "https://raw.githubusercontent.com/datasets/covid-19/master/time-series-19-covid-combined.csv"

df=pd.read_csv(url)

df = df.loc[df['Province/State']=='Hubei',['Date','Confirmed']]
FMT = '%Y-%m-%d'
date = df['Date']
df['Date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01", FMT)).days  )

print(df.head())

#LOGISTICAL MODEL 


def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

x = list(df.loc[:,'Date'])
y = list(df.loc[:,'Confirmed'])
popt, pcov = curve_fit(logistic_model,x,y,p0=[2,10,200])

a=popt[0]
b=popt[1]
c=popt[2]
#print ('POPT=\n', popt)
#print ('PCOV=\n', pcov)

fit=  curve_fit(logistic_model,x,y,p0=[2,100,20000])

errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]

sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c), b))

specific_date = datetime(2020, 1, 1)
new_b = specific_date + timedelta(int(b))
new_sol= specific_date + timedelta(sol)

print('Expected end of infection with logistic model:\n', new_sol)
print('Expected infection peak:\n' , new_b)
print('Expected number of infected people at infection end: \n', int(c))

#EXPONENTIAL MODEL

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))
exp_fit = curve_fit(exponential_model,x,y,p0=[2,10,100])

#RESIDUALS

y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1], exp_fit[0][2]) for i in x]
log_mse=mean_squared_error(y,y_pred_logistic)
exp_mse=mean_squared_error(y,y_pred_exp)

if log_mse < exp_mse :
	print('Logistical model has a minor Mean Squared Error, so it is the best choice')
else :
	print('Exponential model has a minor Mean Squared Error, so it is the best choice')

#PLOT

pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
# Predicted exponential curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()



