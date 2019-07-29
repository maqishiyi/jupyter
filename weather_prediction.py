
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('end-part2_df.csv').set_index('date')


# In[2]:


df.corr()[['meantempm']].sort_values('meantempm')


# In[3]:


predictors = ['meantempm_1',  'meantempm_2',  'meantempm_3',  
              'mintempm_1',   'mintempm_2',   'mintempm_3',
              'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
              'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3',
              'mindewptm_1',  'mindewptm_2',  'mindewptm_3',
              'maxtempm_1',   'maxtempm_2',   'maxtempm_3']
df2 = df[['meantempm'] + predictors]


# In[4]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [16, 22]

fig, axes =  plt.subplots(nrows=6, ncols=3, sharey=True)

arr = np.array(predictors).reshape(6, 3)

for row, col_arr in enumerate(arr):
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['meantempm'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='meantempm')
        else:
            axes[row, col].set(xlabel=feature)
            
plt.show()


# In[8]:


import statsmodels.api as sm

X = df2[predictors]
y = df2['meantempm']

X = sm.add_constant(X)
X.ix[:5, :5]


# In[9]:


alpha = 0.05

model = sm.OLS(y, X).fit()

model.summary()


# In[10]:


X = X.drop('meandewptm_3', axis=1)

model = sm.OLS(y, X).fit()

model.summary()


# In[11]:


X = X[['const','mintempm_1', 'mintempm_2', 'mintempm_3', 'maxdewptm_1', 'maxdewptm_3', 'mindewptm_1', 'maxtempm_1']]
model = sm.OLS(y, X).fit()
model.summary()


# In[12]:


from sklearn.model_selection import train_test_split

X = X.drop('const', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# In[13]:


# http://ddrv.cn/a/126672

from sklearn.linear_model import LinearRegression

# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The expalined Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

