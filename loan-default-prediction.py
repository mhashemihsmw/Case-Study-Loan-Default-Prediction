#!/usr/bin/env python
# coding: utf-8

# !pip install mlflow
# import numpy as np 
# import scipy.stats as ss
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math
# import scipy as spy
# # mlflow
# import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature
# # sklearn
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
# from sklearn.pipeline import make_pipeline, Pipeline
# import sklearn.preprocessing as sp
# import sklearn.compose as sc
# # XGB
# import xgboost as xgb
# 
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# 
# pd.option_context('mode.use_inf_as_na', True)

# In[2]:


data = pd.read_csv('/kaggle/input/loan-default-dataset/Loan_Default.csv')
data.head()


# In[3]:


data.dtypes


# Before we proceed with outlier detection and in order to make the investigation simpler, as a rule of thumb we will throw out those samples for which 15% of features have a missing values. This not only makes the analysis easier, it is also rational because a sample which is "contaminated" by so many missing values is not rather worthy of imputation. 

# In[4]:


l = math.floor(len(data.columns)/5)
print(l)
print(' # Samples that will be deleted: ' + str((data.isna().sum(axis = 1) >= l).sum()))
df = data[data.isna().sum(axis = 1) < l]
df = df.reset_index(drop = True)
df.head()


# In[5]:


# count plots of categorical variables 
df_obj = df.select_dtypes(include = 'object')
df_obj_status = df_obj.join(df['Status'])

plt.figure(figsize=(20,25))
for i, col in enumerate(df_obj):
    plt.tight_layout()
    plt.subplot(7,4,i+1)
    sns.countplot(data = df_obj_status, x = col, hue='Status')


# In[6]:


# log transformation
df_log_transformed = df.copy()
df_log_transformed = df_log_transformed[(df_log_transformed['income'] != 0) & (df_log_transformed['income'].isna() == False)]
df_log_transformed.loc[:,'income'] = np.log10(df_log_transformed.loc[:,'income'])
df_log_transformed['loan_amount'] = df_log_transformed['loan_amount'].astype(np.float64)
df_log_transformed.loc[:,'loan_amount'] = np.log10(df_log_transformed.loc[:,'loan_amount'] + 1)
df_log_transformed.loc[:,'property_value'] = np.log10(df_log_transformed.loc[:,'property_value'] + 1)
df_log_transformed.loc[:,'Upfront_charges'] = np.log10(df_log_transformed.loc[:,'Upfront_charges'] + 1)
df_log_transformed = df_log_transformed[df_log_transformed['LTV'] <= 100]
df_log_transformed.replace([np.inf, -np.inf], 0, inplace=True)


# In[7]:


#  histograms of numerical variables
pd.option_context('mode.use_inf_as_na', True)
df_num = df_log_transformed.select_dtypes(exclude = 'object')
df_sub_num = df_num.drop(columns = ['ID', 'year', 'term', 'Status'])
#'LTV', 'income', 'property_value', 'Upfront_charges', 'rate_of_interest', 'Interest_rate_spread'
plt.figure(figsize=(20,25))
for i, col in enumerate(df_sub_num):
    if col != 'Status':
        plt.tight_layout()
        plt.subplot(7,4,i+1)
        sns.histplot(data = df_sub_num, x = col, bins = 30)


# # Outliers
# Here is how an outlier is defined according to Barnett and Lewis (19844: s "an observation (or subset of observations) which appears to be inconsistent with the remainder of that set of data". We can view outliers from two point of views. First, by looking at each feature individually. In this case, an outlier is identified as a value which is not consistent with the rest of the observations for a specific feature. We can use methods like "modified zscore" and "IQR method" to identify these outliers. Second, by looking at the dependency between features, but in particular between the features and the target variable. A good approach would be to use with grouping of the data using pandas "crosstab". Then apply the methods above or calculate leverage or other distances such as Mahalanobis distance to identify outliers. 
# 
# In general, droping entries because they have been identified outliers by these methods should be done with cautious. It is always better to identify inconsitancy in data by applying EDA. 

# ### Loan to Value ratio (LTVr)
# 
# LTVr is the amount of loan divided by the value of the asset which is being purchased. LTVr cannot  be more than 100 %. 

# In[8]:


print(df['LTV'].describe())
sns.scatterplot(df, y = 'LTV', x = 'loan_amount')
plt.show()


# Calculating asset values by loan amounts multiplied by LTVr clearly shows that these values are most probably entered by mistake. Additionally, we find a lot of entries with missing "rate of interest" among the entries with LTVr greater than 100, which is suspecious! We can assume that these values are entered by mistake and correct them by dividing by 100. However, this is too risky and would be more wise to delete these entries. Just in case:
# 
# https://en.wikipedia.org/wiki/Loan-to-value_ratio#:~:text=Higher%20LTV%20ratios%20are%20primarily,100%25%20are%20called%20underwater%20mortgages.

# In[9]:


df_out = df.copy()
df_out = df_out[df_out['LTV'] <= 100]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(df_out, y = 'LTV', x = 'loan_amount', hue = 'Status', palette = 'colorblind', ax = ax[0])
sns.boxplot(data = df_out, x = 'loan_limit', y = 'LTV', hue = 'Status', palette = 'colorblind', ax = ax[1])
plt.tight_layout()
plt.show()


# ### Rate of interest
# There is also an unusal sample with zero rate of interest. Since interest rate spread is also an outlier for this sample we will assume that the sample is contaminated and it is better to drop it. 

# In[10]:


df_out = df_out[df_out['rate_of_interest'] != 0]
df_out


# ### Income
# 
# Dept to income ratio is non-zero for all samples which do not have missing values. However, income is zero in many cases. This cannot be true! Instead considering these as outliers, we think they can be considered as missing values. We will discuss this case in the next chapter. 
# 
# Additionally, some incomes are really low for those who are getting a loan for secondery or investment residency (occypancy type of sr and ir). 

# In[11]:


df_tmp = df_out[df_out['income'] > 0] 
pd.crosstab(df_tmp['occupancy_type'], df_tmp['Status'], values = df_tmp['income'], aggfunc = 'median').round(2)


# In[12]:


df_tmp = df_out[df_out['income'] > 1500] 
pd.crosstab(df_tmp['occupancy_type'], df_tmp['Status'], values = df_tmp['income'], aggfunc = 'median').round(2)


# However, by comparing two cases where one only considers incomes above 1500 (a salary which one earns almost with minimum wage per hour), there is no concrete evidance that there is systematic sampling error behind these observations.

# ### Modified Zscore
# 
# 

# We will derive the modified zscores for the numeric features just to get a sense of the extent of outliers recognized by this method. Later during modeling we will remove such entries to see if removing such entries it has "reasonable" impact on the performance of the models. By reasonable we mean if test accuracy is not impacted or overfitting does not occur. 

# In[13]:


df_num_num = df.copy()
df_num_num = df_num_num.select_dtypes(exclude = 'object')
df_num_num = df_num_num.drop(['ID', 'year', 'Status', 'Upfront_charges'], axis = 1)
df_num_num = df_num_num[df_num_num['income'] > 0]


# In[14]:


def modified_zscore (data):
    med = data.apply(lambda x: abs(x - x.median()))
    mad = med.apply(lambda x: x.median())
    mzscore = pd.DataFrame()
    for i in range(1,len(mad)):
        if mad.iloc[i] != 0:
            mz = abs(0.6745 *(data.iloc[:,i] - data.iloc[:,i].median())/mad.iloc[i]) > 3.5
            mzscore = pd.concat([mzscore.reset_index(drop=True), mz], axis = 1)
    return mzscore

mzscores = modified_zscore(df_num_num)
mzscores.sum()


# ### Mahalanobis Distance

# We will investigate this distance after dealing with missing data.

# In[15]:


def MahalanobisDistance(data): 
    y_mu = data - np.mean(data) 
    cov = np.cov(data.values.T) 
    #inv_covmat = np.linalg.inv(cov) 
    #left = np.dot(y_mu, inv_covmat) 
    #mahal = np.dot(left, y_mu.T) 
    return cov#mahal.diagonal()
#df_num_log = np.log10(df_num_num + 1)
#MahalD = MahalanobisDistance(df_num_log)
#MahalD


# # Treating Missing Data

# ## Estimation
# 
# In this approach we estimate missing information by using each feature individually.

# In[16]:


print("#Samples: " + str(len(df)))
df_out.isna().sum()


# In[17]:


df_out = df_out.reset_index(drop = True)
df_missing = df_out.copy()
income_est = pd.crosstab(df_out['Status'], df_out['loan_purpose'], values = df_out['income'], aggfunc = 'median')
income_est


# **Income** is log-normally distributed (histrogram suffices for us as evidance; however tests of normality can also be applied). To handle missing data, we can either impute using the mean of the log-transformed values or the median of the original income values. Additionally, we can improve the imputation by incorporating loan status and loan purpose as grouping variables. This is because income seems to differ noticably between these groups. In particular, we replace missing income values with the corresponding estimator (either the mean of the log-transformed values or the median of the original values) derived from the subset of data within each loan status group. This approach ensures that the imputation reflects differences across loan statuses. Note that incomes with zero values are also considered as missing. 

# In[18]:


def numerical_missing (dff, est, col, vals = None):
    dat = dff.copy()
    where_income_missing = np.where((dat[col].isna()) | (dat[col] == 0))[0]
    y = 0
    for row in where_income_missing:
        x = dat.iloc[row,:]
        if vals is None:
            y = est
        elif len(vals) == 1:
            y = est.loc[x[vals[0]]]
        elif len(vals) == 2:
            y = est.loc[x[vals[0]],x[vals[1]]]
        dat.loc[row,col] = y
    return dat


# In[19]:


df_cleaned_0 = numerical_missing(df_missing, income_est, 'income', np.array(['Status', 'loan_purpose']))
df_cleaned_0.head()


# In[20]:


# income
print(df_cleaned_0.select_dtypes(exclude = 'object').drop(['ID', 'year', 'Status', 'Upfront_charges'], axis = 1).corr()['income'])
sns.scatterplot(df_cleaned_0, y = 'income', x = 'loan_amount')
plt.xscale('log')
plt.yscale('log')
plt.show()


# For a series of categorical features for which missing data is present, we can use either knn or binomial/multinomial distributions to impute the missing data. Among those are:
# - age
# - submission of application
# - approved in advance (approv_in_adv)
# - loan purpose
# - Negative Ammortization
# - loan limit

# In[21]:


print(df_cleaned_0['age'].value_counts()) #multinomial
print('  ')
print(df_cleaned_0['submission_of_application'].value_counts()) #binomial
print('  ')
print(df_cleaned_0['approv_in_adv'].value_counts()) #binomial
print('  ')
print(df_cleaned_0['loan_purpose'].value_counts()) #multinomial
print('  ')
print(df_cleaned_0['Neg_ammortization'].value_counts()) #binomial
print('  ')
df['loan_limit'].value_counts() # binomial


# In[22]:


def categorical_missing(aldata, col, p = None, vals = None):
    dff = aldata.copy()
    if vals is None:
        vals = dff[col].dropna().unique()
    if p is None:
        p = dff[col].value_counts()/dff[col].value_counts().sum()
    where_missing = np.where(dff[col].isna())[0]
    n = len(where_missing)
    mult = np.random.multinomial(1, p, n)
    k = 0
    t = []
    for i in mult:
        k = k + 1
        t.append(vals[i == 1][0])
    dff.loc[where_missing, col] = t
    return dff


# Similar to "age", "**loan term**" can be placed in brackets and treated as a categorical variable. There is not enough evidance of high correlation with other features. Hence, we will replace missing terms by using multinomial distribution where probabilities are estimated as the weights of each term interval.

# In[23]:


df_cleaned_1 = categorical_missing(df_cleaned_0, 'age')
df_cleaned_1 = categorical_missing(df_cleaned_1, 'submission_of_application')
df_cleaned_1 = categorical_missing(df_cleaned_1, 'approv_in_adv')
df_cleaned_1 = categorical_missing(df_cleaned_1, 'loan_purpose')
df_cleaned_1 = categorical_missing(df_cleaned_1, 'loan_limit')
df_cleaned_1 = categorical_missing(df_cleaned_1, 'Neg_ammortization')


# **Term** of a loan can be also an important factor in default. Term has few missing values. It can be seen that more than 80% of the loans are issued for 360 months or 30 years, and more than 95% for 15 years or more. The 30 years term is so dominant in the data that makes replacing the missing values a chance. Later in feature engineering we would bin this feature in few intervals, namely, <180, 180-240, 240-360. 

# In[24]:


df_tmp = df_cleaned_1[df_cleaned_1['term'].isna() == False]
print(pd.crosstab(df_tmp['Status'], df_tmp['loan_purpose'], values = df_tmp['term'], aggfunc = 'mean'))
print(df_tmp['term'].value_counts())


# In[25]:


N = (df_tmp['term'].isna() == False).sum()
p1 = (df_tmp['term'] <= 180 ).sum()/N
p2 = ((df_tmp['term'] <=240 ) & (df_tmp['term'] > 180 )).sum()/N
p3 = (df_tmp['term'] > 240).sum()/N
pp = np.array([p1,p2,p3])
v = np.array([180,240,360])
df_cleaned_2 = categorical_missing(df_cleaned_1, 'term', p = pp, vals = v)


# **Debt to Income Ratio (DTIR)** is another important feature which contains relatively large amount missing data. It incorporates debt, an important variable which can be the leading cause of default. Therefore, it is important to impute the missing data for this random variable. DTIR has a skewed distribution. Additionally, it does not indicate difference in distribution and first moments when marginal distributions conditioned on different categorical variables are taken into account. However, there is slight difference if we consider default and non-default cases. Hence, the median of DTIR for each default and non-default cases can be a good estimator for imputing missing DTIR. 

# In[26]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_out, x = 'dtir1', bins = 20, ax = ax[0])
sns.boxplot(data = df_log_transformed, x = 'Status', y = 'dtir1', ax = ax[1])
plt.tight_layout()
plt.show()


# In[27]:


# dtir1
ditr_est = df_cleaned_2.groupby(['Status'])['dtir1'].median()
df_cleaned_3 = numerical_missing(df_cleaned_2, ditr_est, 'dtir1', np.array(['Status']))
df_cleaned_3.head()


# **Upfront Charges** seems to be dependent on loan type and loan purpose.  However, there are a lot of loans issued with zero upfront charges. For these cases, the fee is probably included in the loan amount itself. Banks usually do that, however, then the interest rates wil be calculated higher. We also keep in mind that the upfront charges are usually calculated as a percentage of loan amount. Hence, we do not know two things about missing data and we would like to estimate those. First, if the upfront charges were included in the loan. For these loans, the upfront charges would be zero. Second, what percentage of loan is considered as an upfront fee if this fee is not zero. We will do as follows: 1) we flip a coin for the fee to be included in the loan amount, i.e., fee being zero or not, 2) if not zero, then we calculated the fee as the loan amount multiplied by the mean percentage of non-zero upfront charges relative to loan amount. 

# In[28]:


df_tmp = df_cleaned_3.copy()
where_missing = np.where(df_tmp['Upfront_charges'].isna())[0]
lul = len(where_missing)
pul = len(df_tmp[df_tmp['Upfront_charges'] == 0])/len(df_tmp[df_tmp['Upfront_charges'].isna() == False])
pul = np.array([pul, 1- pul])
u = (df_tmp[df_tmp['Upfront_charges'] > 0]['Upfront_charges'])
l = (df_tmp[df_tmp['Upfront_charges'] > 0]['loan_amount'])
ul = u*100/l
mul = ul.mean()
vul = np.array([0,mul])
mull = np.random.multinomial(1, pul, lul)
k = 0
t = []
for i in mull:
    k = k + 1
    t.append(vul[i == 1][0])
df_tmp.loc[where_missing, 'Upfront_charges'] = t
df_cleaned_4 = df_tmp.copy().reset_index()


# **Rate of Interest** and **Interest rate spread** look pretty much normal. We can impute the missing values by using the average of each column. 

# In[29]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(df_cleaned_4['rate_of_interest'], bins = 30)
ax[1].hist(df_cleaned_4['Interest_rate_spread'], bins = 30)
plt.tight_layout()
plt.show()
df_cleaned_4[['rate_of_interest', 'Interest_rate_spread']].describe()


# In[30]:


df_cleaned_5 = df_cleaned_4.copy()
df_cleaned_5['rate_of_interest'] = df_cleaned_5['rate_of_interest'].fillna(df_cleaned_5['rate_of_interest'].mean())
df_cleaned_5['Interest_rate_spread'] = df_cleaned_5['Interest_rate_spread'].fillna(df_cleaned_5['Interest_rate_spread'].mean())


# ## Regression/Classification

# Alternatively, we can estimate missing values by treating the feature as a dependent variable and using othe features where missing data is absent as independent variable. 
# 
# For example, given the linear relationship between loan amount and income, simple linear regression can be applied to predict and impute the missing values of income. This approach leverages the predictive power of related variables to generate more accurate estimates for the missing data.

# # Feature Engineering
# 

# **Terms** of mortgages are usually 15, 20, or 30 years. This can be easily seen from the data that terms are dominated by these few values. Therefore, binning terms and converting it to categorical variable seems to be a reasonable idea rather than using it as numerical variable. While there are different appraoches to binning (either simple statistical appraoches as equal width/frequency binning, or ML appraoches like k-means clustering), based on our knowledge of the data it seems to be reasonable to bin the the terms in the intervals named above. 

# In[31]:


df_engineered_0 = df_cleaned_5.copy()
df_engineered_0['term'] = pd.cut(df_engineered_0['term'], bins = [0, 180, 240, 360], labels = ['short', 'medium', 'long'])


# The **upfront charges** feature is a complicated one. There are many 0 values and many fees that are realy small relative to loan amount. There are also many values as large as 20000 or above (actually the max is around 60000). As we discussed above, the upfront fees are usually calculated as a percentage of loan amount. Additionally, in many cases the fee can be included in the loan amount. Considering these facts we proceed as follows: 1) we create another feature which encodes the upfront charges to be zero or not (in fact we treat really small values like 300 or less also as zero). 2) we transform the upfront charges to percentages of loan amount. These transformations have the advantage that they add information to the data where it is actually missing, i.e., the zero fees.

# In[32]:


df_engineered_1 = df_engineered_0.copy()
df_engineered_1['Upfront_charges_y/n'] = np.where(df_engineered_1['Upfront_charges'] < 300, 0, 1)
df_engineered_1['Upfront_charges'] = df_engineered_1['Upfront_charges']*100/df_engineered_1['loan_amount']


# We will also log Transformation to **income, property value, loan amount, and Credit Score**. The reason is that their distributions are right-skewed. Transformation helps with fairly representing the difference between values, in particular, reducing the impact of outliers. 

# In[33]:


df_engineered_2 = df_engineered_1.copy()
df_engineered_2['income'] = np.log10(df_engineered_2['income'])
df_engineered_2['property_value'] = np.log10(df_engineered_2['property_value'])
df_engineered_2['loan_amount'] = np.log10(df_engineered_2['loan_amount'])


# Of note, we could still apply scaling, however, we find it not necessary at this point as most of the numerical do not differ significantly in range of values. Later we can also investigate the model performances if numerical featrues are scaled. 
# 
# Before we apply encoding to categorical variables, we will drop the following columns: **ID, Year, LTV**. LTV will be dropped because its information is already contained in loan amount and property value. Credit type and secured by will be dropped because of the imbalanced representaions of their values in the dataset.

# In[34]:


df_engineered_3 = df_engineered_2.drop(['ID', 'year', 'index', 'LTV', 'credit_type', 'Secured_by'], axis = 1)


# We can apply traget encoding to **age and Region** because of their high cardinality. The most suitable encoding for **term and total units** is ordinal labling. This is because higher terms and higher number of units is highly correlated with the property value and the loan amount, and consequently with the default ratio. The rest of categorical variables can be encoded using dummy method, i.e., using N-1 features to represent N labels/categories.

# In[35]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(df_engineered_3, x = 'total_units', y = 'loan_amount', hue = 'Status', ax = ax[0])
sns.scatterplot(df_engineered_3, x = 'loan_amount', y = 'property_value', hue = 'Status', ax = ax[1])
plt.tight_layout()
plt.show()


# In[36]:


df_engineered_4 = df_engineered_3.copy()
# target
target_mean_age = df_engineered_4.groupby('age')['Status'].mean()
df_engineered_4['age'] = df_engineered_4['age'].map(target_mean_age)
target_mean_region = df_engineered_4.groupby('Region')['Status'].mean()
df_engineered_4['Region'] = df_engineered_4['Region'].map(target_mean_region)
#ordinal
Oencoder = sp.OrdinalEncoder()
df_engineered_4[['term', 'total_units']] = Oencoder.fit_transform(df_engineered_4[['term', 'total_units']]) + 1
# dummy

## # Specify the categorical columns to be one-hot encoded
cardinal_cols = df_engineered_4.select_dtypes(include = 'object').columns

## # Define the ColumnTransformer with OneHotEncoder
## # drop='first' will drop the first category to avoid multicollinearity
column_transforms = sc.ColumnTransformer(transformers = [('onehot', sp.OneHotEncoder(drop='first'), cardinal_cols)], remainder='passthrough')

## # Apply the transformation
df_tmp = column_transforms.fit_transform(df_engineered_4[cardinal_cols])

## # Convert the result to a DataFrame for easier readability
## # Extract feature names after encoding
onehot_feature_names = column_transforms.named_transformers_['onehot'].get_feature_names_out(cardinal_cols)
df_tmp = pd.DataFrame(df_tmp, columns = onehot_feature_names)

## # drop original columns
df_engineered_4 = df_engineered_4.drop(cardinal_cols, axis = 1)

## # concat dataframes
df_engineered_4 = pd.concat([df_engineered_4, df_tmp], axis=1)


df_engineered_4[['Credit_Score', 'Status', 'Upfront_charges_y/n']] = df_engineered_4[['Credit_Score', 'Status', 'Upfront_charges_y/n']].astype('float64')


# In[37]:


def normality_test (feature):
    a = spy.stats.normaltest(feature)
    print(a)
    b = spy.stats.anderson(feature)
    print(b)
    n = np.random.normal(feature.mean(), feature.std(), len(feature))
    c = spy.stats.kstest(feature, n)
    print(c)


# # Modeling

# In the following we build a machine learning pipeline to evaluate the performance of different models on a dataset using cross-validation and hyperparameter tuning. Below is a summary of the main components:
# 
# 1. **Train-Test Split**:  
#    The function `train_test_splits` splits the dataset into training and testing sets based on a `test_size` of 50% and a random seed (`rn_state`) for reproducibility. It separates the target variable (`Status`) from the feature set.
# 
# 2. **Hyperparameter Configuration**:  
#    A list `hparameters` contains:
#    - Names of models (Logistic Regression, KNN, Random Forests, and Boosted Trees).
#    - Corresponding model objects or placeholders.
#    - Binary flags indicating whether dummy variables are included.
#    - Dictionaries with hyperparameter options for each model to facilitate grid search.
# 
# 3. **Model Performance Logging**:  
#    The function `mlflow_metrics` calculates and logs key performance metrics (accuracy, precision, recall, ROC/AUC) for both training and testing datasets using the `mlflow` library.
# 
# 4. **Pipeline and Scaling**:  
#    - Scales numeric features using `MinMaxScaler` through a `Pipeline` and `ColumnTransformer`.
#    - Uses `GridSearchCV` to optimize the hyperparameters for each model.
#    - Logs the best hyperparameters, fits the model, and records the predictions.
# 
# 5. **Automated Model Training**:  
#    The `mlflow_models` function:
#    - Iterates through the list of models and their respective hyperparameters.
#    - Chooses the appropriate dataset (with or without dummy variables).
#    - Logs experiment details to `mlflow`, trains the models, and evaluates performance metrics.

# In[38]:


# Train-Test Split
test_size = 0.5
rn_state = 1818
df_model_with_dummies = df_engineered_4.copy()
df_model_without_dummies = df_engineered_3.copy()
def train_test_splits(df_model, rn_state):
    df_model_split = df_model.copy()
    y = df_model_split.pop('Status')
    X = df_model_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = rn_state)
    return X_train, X_test, y_train, y_test

# Hyperparameter Values
hparameters = []

hparameters.append(['LogisticRegression', 'KNN', 'RandomForests', 'BoostedTrees']) # tags
hparameters.append([LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(), HistGradientBoostingClassifier()]) # models
hparameters.append([1, 1, 0, 0])
hparameters.append([{'penalty':['l2'], 'C':[0.5,1], 'solver': ['newton-cholesky'], 'max_iter':[200]}, #Logistic Regression,
                    {'n_neighbors':[3,5,10]}, #KNN
                    {'n_estimators': [50, 100, 200], 'criterion': ['log_loss'], 'max_depth': [5, 10], 'min_samples_split': [10,100]}, # Random Forests
                    {'l2_regularization': [0.1,0.5,0.9], 'min_samples_leaf': [10, 100], 'max_depth': [5, 10], 'learning_rate': [0.1,0.5]} #XGB
                   ]) 


# Model Performance 
def mlflow_metrics(y_train, y_train_pred, y_test, y_test_pred):
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    
    mlflow.log_metric('Train Accuracy', accuracy_train)
    mlflow.log_metric('Train Precision', precision_train)
    mlflow.log_metric('Train Recall', recall_train)
    mlflow.log_metric('Train ROC/AUC', roc_auc_train)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred)
    
    mlflow.log_metric('Test Accuracy', accuracy)
    mlflow.log_metric('Test Precision', precision)
    mlflow.log_metric('Test Recall', recall)
    mlflow.log_metric('Test ROC/AUC', roc_auc)

scoring = ['accuracy', 'precision', 'recall', 'roc_auc']

# Mlflow
def mlflow_models (flows, data_with_dummies, data_without_dummies):
    idx = 0
    for f in range(0,len(flows[0])):
        model_name = flows[0][f]
        print(model_name)
        mlflow.set_experiment(model_name)
        with mlflow.start_run():
            # Experiment Setup
            mlflow.set_tag('Model', model_name)
            mlflow.log_param('Test/Train Percentage', test_size*100)
            idx += 1
            experiment_tag = f"{model_name}.run_{idx}"
            mlflow.set_experiment_tag('ExpID', experiment_tag)
            model = flows[1][f]
            # Dataset
            if flows[2][f] == 1:
                dat_model = data_with_dummies.copy()
            else:
                dat_model = data_without_dummies.copy()
            
            # Parameters
            params = flows[3][f]
            params_keys = list(params.keys())
            for k in range(0,len(params)):
                params[model_name+'__'+params_keys[k]] = params.pop(params_keys[k]) 

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_splits(dat_model, rn_state)

            # Pipeline
            mmxscaler = sp.MinMaxScaler()
            numeric_features = ['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 
                                'property_value', 'income', 'dtir1', 'Credit_Score']
            numeric_transformer = Pipeline(steps=[("Scaler", mmxscaler)])
            preprocessor = sc.ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])
            pipe = Pipeline(steps = [('preprocessor', preprocessor), (model_name, model)])

            # Cross-Validation
            grid = GridSearchCV(pipe, params, scoring = scoring, refit = 'roc_auc', return_train_score=True, cv = 7, verbose = 4)

            # Fitting
            grid.fit(X_train, y_train) 
            best_params = grid.best_params_
            mlflow.log_params(best_params)
            best_model = grid.best_estimator_
            
            # Predictions
            y_train_pred = grid.predict(X_train) 
            y_test_pred = grid.predict(X_test)
            
            # Metrics
            mlflow_metrics(y_train, y_train_pred, y_test, y_test_pred)

            # Logging
            signature = infer_signature(model_input = X_train, model_output = y_test_pred)
            
            


# In[39]:


mlflow_models(hparameters, df_model_with_dummies, df_model_without_dummies)


# In[40]:


mldf =  mlflow.search_runs(experiment_names = hparameters[0])
mldf[['tags.Model', 'params.Test/Train Percentage', 
        'metrics.Train Recall', 'metrics.Test Recall',
        'metrics.Train Accuracy', 'metrics.Test Accuracy', 
        'metrics.Train Precision', 'metrics.Test Precision',
        'metrics.Train ROC/AUC', 'metrics.Test ROC/AUC']]


# # Results
# 
# Here’s a quick summary of how the models performed on the classification task:
# 
# 1. **Boosted Trees**: Absolutely crushed it! Nearly perfect scores across the board, with recall, precision, accuracy, and ROC/AUC all above 99.9%. Super reliable and consistent.
# 
# 2. **Random Forests**: Also performed really well, with recall above 99.3% and ROC/AUC around 99.7%. It’s not quite as good as Boosted Trees but still a strong contender.
# 
# 3. **K-Nearest Neighbors (KNN)**: Did a decent job, with a test recall of 96.7% and ROC/AUC in the same range. However, its precision dropped to 84.2%, suggesting it might struggle with misclassifications.
# 
# 4. **Logistic Regression**: Didn’t fare so well here, with a test recall of just 44.9% and ROC/AUC around 70.9%. It seems this model isn’t the best fit for this problem.
# 
# **Takeaway**: Boosted Trees is the clear winner, with Random Forests coming in as a solid second choice. KNN is okay but could use some tuning, while Logistic Regression probably isn’t worth pursuing further for this dataset.
# 
# ![Screenshot 2024-11-25 203127.png](attachment:258f0b29-0bb9-4bb3-bd1c-68a807b3a5f8.png)

# # Conclusion
# 
# In conclusion, this pipeline demonstrates a robust approach to evaluating and tuning various machine learning models on the loan default dataset. However, the possibilities for exploration and improvement are vast. For instance, we could incorporate additional models to expand our analysis, experimenting with advanced algorithms like Support Vector Machines or Neural Networks. When it comes to data preprocessing, while we used MinMaxScaler here, other scaling techniques, or even skipping scaling altogether, might lead to better outcomes depending on the model and dataset characteristics. 
# 
# Currently, we utilized all available features for training, but we could enhance this process by identifying the most impactful ones through methods like Principal Component Analysis (PCA). This could streamline our model and potentially improve its performance. Regarding handling missing values, instead of the exhaustive imputation we applied, we might simply exclude rows with missing data if the dataset is sufficiently large or use sophisticated imputation strategies such as regression analysis to estimate missing values more accurately.What this hthis highlights is the near-limitless scope of innovatiML pipeline designcience. While the tools and methods for buildMLarning pipeline are well-d and limitedefined, the creative combinations, adaptations, and insights we can derive from them are infinite. It reminds me of the movie *The Legend of 1900*, where the protagonist plays on a finite set of piano keys but creates music that feels boundtiior.
