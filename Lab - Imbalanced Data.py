#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


get_ipython().system(' pip install sqlalchemy')
get_ipython().system(' pip install PyMySQL')


# In[2]:


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sb

import pymysql
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

import getpass
password = getpass.getpass()


# In[3]:


# Creating a connection to use Sakila database from MySQL

connection_string = 'mysql+pymysql://root:' + password + '@localhost/sakila'
engine = create_engine(connection_string)


# ### In order to optimize our inventory, we would like to know which films will be rented next month and we are asked to create a model to predict it.

# In[4]:


# Step 1: Create a query or queries to extract the information you think may be relevant for building the prediction model.

query = '''

WITH COUNT AS (SELECT DISTINCT A.FILM_ID,
                               SUBSTRING(C.RENTAL_DATE, 1, 7)    AS RENTAL_MONTH,
                               COUNT(DISTINCT C.RENTAL_ID)       AS RENTAL_COUNT
               FROM SAKILA.FILM A
                        INNER JOIN
                    SAKILA.INVENTORY B ON A.FILM_ID = B.FILM_ID
                        INNER JOIN
                    SAKILA.RENTAL C ON B.INVENTORY_ID = C.INVENTORY_ID
               GROUP BY 1, 2)

SELECT DISTINCT
    A.FILM_ID,
    A.TITLE,
    A.RELEASE_YEAR,
    A.RENTAL_DURATION,
    A.RENTAL_RATE,
    A.REPLACEMENT_COST,
    A.RATING,
    E.NAME,
    SUM(CASE
        WHEN RENTAL_MONTH IN ('2005-05' , '2005-06', '2005-07', '2005-08') THEN RENTAL_COUNT
        ELSE 0
    END) AS TOTAL_RENTAL,
    SUM(CASE
        WHEN RENTAL_MONTH = '2005-05' THEN RENTAL_COUNT
        ELSE 0
    END) AS RENTAL_MAY05,
    SUM(CASE
        WHEN RENTAL_MONTH = '2005-06' THEN RENTAL_COUNT
        ELSE 0
    END) AS RENTAL_JUN05,
    SUM(CASE
        WHEN RENTAL_MONTH = '2005-07' THEN RENTAL_COUNT
        ELSE 0
    END) AS RENTAL_JUL05,
    SUM(CASE
        WHEN RENTAL_MONTH = '2005-08' THEN RENTAL_COUNT
        ELSE 0
    END) AS RENTAL_AUG05,
    SUM(CASE
        WHEN RENTAL_MONTH = '2006-02' THEN RENTAL_COUNT
        ELSE 0
    END) AS RENTAL_FEB06
FROM
    SAKILA.FILM A
        INNER JOIN
    COUNT B ON A.FILM_ID = B.FILM_ID
        INNER JOIN
    SAKILA.FILM_CATEGORY D ON A.FILM_ID = D.FILM_ID
        INNER JOIN
    SAKILA.CATEGORY E ON D.CATEGORY_ID = E.CATEGORY_ID
GROUP BY 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8

'''

# Step 2: Read the data into a Pandas dataframe.

data = pd.read_sql_query(query, engine)
data


# In[5]:


# Step 3: Analyze extracted features and transform them.


# In[6]:


data = data.set_index('FILM_ID')
data


# In[7]:


data.isna().sum()


# In[8]:


data.dtypes


# In[9]:


data.describe()


# We see how mean value for the number of times each was rented increase significantly from May 2005 to August 2005. Also, all films we rented at least 1 time in July 2005 and August 2005. We will drop Feb 2006 as we don't have timeline trends to compare against. On average, each movie has been rented 17 times.

# In[ ]:





# In[33]:


data['TOTAL_RENTAL'].value_counts().sort_values()


# Between May 2005 and August 2005, all movies hve been rented, the minimum was 4 times. This is an issue with imbalanced data, as there is no movie without rental, so we can't predict if it will be rented.

# In[10]:


data['RELEASE_YEAR'].value_counts()


# All films are released in 2006, hence we can drop this column for our analysis. However it's interesting to understand what "release year" means as we have rentals hapenning in 2005.

# In[11]:


data['RENTAL_MAY05'].value_counts()


# In[12]:


data['RENTAL_JUN05'].value_counts()


# In[13]:


data['RENTAL_JUL05'].value_counts()


# In[14]:


data['RENTAL_AUG05'].value_counts()


# In[24]:


for col in ['TOTAL_RENTAL', 'RENTAL_MAY05',
       'RENTAL_JUN05', 'RENTAL_JUL05', 'RENTAL_AUG05']:
    sb.displot(data[col])
    plt.show()


# All films have been rented during July and August, hence our prediction would always be 1. We do see a positive linear trend over time, meaning that month after month more movies get rented and each movie gets rented more times. 
