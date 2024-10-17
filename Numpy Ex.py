#!/usr/bin/env python
# coding: utf-8

# In[1]:


#HW1 - Bhavya Parmar


# In[31]:


#Q1. Convert the list of 26 letters of the alphabet to the index of the Dataframe, and assign 26 random integer numbers and print out the first 5 rows. 

import pandas as pd

import numpy as np

letters = list('abcdefghijklmnopqrstuvwxyz')

rand_int = np.random.randint(1,100,26)

datafr = pd.DataFrame(rand_int, index = letters,columns = ['Random Integers'])

print(datafr.head(5))


# In[32]:


#Q2. Construct the following DataFrame and print it out. Use "iloc" to convert the first column of this DataFrame as a Series.  

import pandas as pd

import numpy as np

data = {
    'class1' : [1,2,3,4,7,11],
    'class2' : [4,5,6,9,5,0],
    'class3' : [7,5,8,12,1,11]
}

datafr = pd.DataFrame(data)

print(datafr)

first_col_as_series = datafr.iloc[:, 0]

print("\n The First Column As Series Is: ")

print(first_col_as_series)


# In[36]:


#Q3. Work on the above data (question 2), implement the following tasks using indexers loc and/or iloc:   

#(1) Select all columns, except one given column 'class3', and print out the result.

#(2) Remove first 3 rows of the DataFrame and print it out.

#(3) Remove last 3 rows of the DataFrame and print it out.

#---------------------------------------------------------------------------------------

#1. 
print("Q1) Result after selecting all columns, except one given column 'class3', and printing out the result':\n")

print(datafr.loc[: , datafr.columns != 'class3'])

print()

#2.
print("Q2) Result after removing first 3 rows of the DataFrame and printing it out:\n")

print(datafr.iloc[3:])

print()

#3.
print("Q3) Result after removing last 3 rows:\n")

print(datafr.iloc[:-3])

print()


# In[83]:


#Q4.Use Pandas MultiIndex method(s) to create a multi-indexing DataFrame as below [4 points]:
#Note: you should use random number generators to mock some reasonable data such as HR (Heart Rate) and Temp (Body Temperature).

index = pd.MultiIndex.from_tuples([(2018,1),(2018,2),(2019,1),(2019,2)])

columns = pd.MultiIndex.from_tuples([('Bob','HR'),('Bob','Temp'),('Julia','HR'),('Julia','Temp'),('Sue','HR'),('Sue','Temp')])

dt = np.round(np.random.uniform(30, 99,size=(4,6)),2)

dt = pd.DataFrame(data=data,index=index,columns = columns)

dt

#Q4.1 (1) Get all the information of Julia.
dt.Julia

#Q4.2(2) Get Juila's heart rate data.
dt.Julia.HR

#Q4.3 (3) Get Bob’s all information in 2018.   
dt.Bob.loc[2018]


# In[125]:


#Q5.
#1: Create a DataFrame df: the name of 4 columns is pqrs respectively, the name of 10 indexes is abcdefghij respectively. The values are randomly distributed in [1,100) and the seed is set as 42.

np.random.seed(42)

index = ['a','b','c','d','e','f','g','h','i','j']
columns = ['p','q','r','s']
        
datafr = pd.DataFrame(np.random.randint(1,100, size = (10,4)), columns = columns, index = index)
        
print(datafr)

print()
        
#Q5.
#2: Create a new column such that, each row of this df contains the row number of nearest row-record by Euclidean distance.

np.random.seed(42)

index = ['a','b','c','d','e','f','g','h','i','j']

columns = ['p','q','r','s']
        
datafr = pd.DataFrame(np.random.randint(1,100, size = (10,4)), columns = columns, index = index)

def euclidean_dis(row_1, row_2):
    return np.sqrt(np.sum((row_1 - row_2) ** 2))

#nearest_row = nr

df['nr'] = None

for i, row in datafr.iterrows():
    dist = datafr.drop(i).apply(lambda x: euclidean_dis(row, x), axis=1)
    
    nearestrowindex = dist.idxmin()
    
    datafr.at[i, 'nr'] = nearestrowindex

print("The Answer 2 : ")

print(datafr)      


# In[90]:


#Q6. Create a Dataframe with rows as strides from a given series: L = pd.Series(range(15))

import pandas as pd

L = pd.Series(range(15))

rws = [L[i:i+4].values for i in range(0, len(L)-3, 2)]

datafr = pd.DataFrame(rws).to_numpy()

datafr


# In[124]:


#Q7. 
#(1) Create a DataFrame, the name of columns is abcde respectively, the values ranges [0,20) [2 points].

#(2) Interchange columns 'a' and 'c' [2 points].

#(3) Create a generic function to interchange arbitrary two columns [6 points].

#(4) Sort the columns in reverse alphabetical order, that is column 'e' first through column 'a' last. [2 points]

datafr = pd.DataFrame(np.random.randint(0, 20, size=(5, 5)), columns=['a', 'b', 'c', 'd', 'e'])

#1: Display original DataFrame

print(datafr)

print("\n")

#2: Interchanging the columns 'a' and 'c'
datafr[['a', 'c']] = datafr[['c', 'a']]

print("After Interchanging the columns 'a' and 'c':")

print(datafr)

print("\n")

#3: Create a generic function to interchange arbitrary two columns
def interchange_columns(df, col1, col2):
    
    df[[col1, col2]] = df[[col2, col1]]

interchange_columns(datafr, 'b', 'd')

print("DataFrame after interchanging 'b' and 'd' and using generic function:")

print(datafr)

print("\n")

#4: Sort the columns in reverse alphabetical order
datafr = datafr[sorted(datafr.columns, reverse=True)]

print("DataFrame after sorting columns in reverse alphabetical order:")

print(datafr)


# In[98]:


#Q8 Create a TimeSeries starting ‘2021-01-01’ and 10 weekends (Saturdays) after that having random numbers as values.  

np.random.seed(10)

date_range = pd.date_range(start='2021-01-01', periods=10, freq='W-SAT')

saturdays = date_range[date_range.day_name() == 'Saturday']

timeseries = pd.Series(np.random.rand(len(saturdays)), index=saturdays)

timeseries


# In[102]:


#Q9.
#1: Create a Pandas time series with missing dates (from 2021-01-01 to 2021-01-08) and values, shown as below

dt = {'2021-01-01': 1.0,
        '2021-01-03': 10.0,
        '2021-01-06': 3.0,
        '2021-01-08': None}

data = pd.Series(dt).astype(float)

data.index = pd.to_datetime(data.index)

print(data)

#2: Make all missing dates appear and fill up with value from previous date, shown as below.

all_dates = pd.date_range(start='2021-01-01', end='2021-01-08')

time_series_final = data.reindex(all_dates).ffill()

print(time_series_final)


# In[111]:


#Q10.
#1: Create a DataFrame ‘df’: the indexes are integer from 0 to 8 (9 rows); the columns are ‘fruit’, ‘taste’, ‘price’; the items of ‘fruit’ are: apple, banana, orange, respectively and repeating 3 times. The ‘taste’ ranges [0,1). The price is integer and ranges [1, 15). The seed of random values is set as 100. [4 points]

#2:  Find the second largest value of 'taste' for 'banana' (Hint: you can use groupby) [6 points]

#3: Compute the mean price of every fruit, while keeping the fruit as another column instead of an index. [6 points]

np.random.seed(100)

dt = {
    'Fruit': ['Apple', 'Banana', 'Orange'] * 3,
    
    'Taste': np.random.rand(9),
    
    'Price': np.random.randint(1, 15, size=9)
    
}

datafr = pd.DataFrame(dt, index=range(9))

print(datafr)

print("\n")

x = datafr[datafr['Fruit'] == 'Banana']['Taste'].nlargest(2).iloc[-1]

print("2nd largest taste value for 'banana':", x)

print('\n')

meanprice = datafr.groupby('Fruit')['Price'].mean().reset_index()

print("Mean price of every fruit:")

print(meanprice.groupby(['Fruit']).agg(Mean_Price=('Price', 'mean')).reset_index())


# In[114]:


#Q11.
#(1) Create a df, for the values [0,100) with a seed 1, its shape is (8,10). [2 points]

#(2) Create a new column with the name 'largest2nd', which has the second largest value of each row of df. [6 points]

np.random.seed(1)

datafr = pd.DataFrame(np.random.randint(0, 100, size=(8, 10)))

print(datafr)

print("\n")

datafr['Largest2nd'] = datafr.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)

print("Answer 2: The DataFrame with 'Largest2nd' Column Is:\n")

print(datafr)


# In[117]:


#Q12 For the same df created in above question, use Pandas ufuncs to normalize all columns of df by subtracting the column mean and divide by standard deviation and keep the result within two decimal places.

np.random.seed(1)

datafr = pd.DataFrame(np.random.randint(0, 100, size=(8, 10)))

#normalized ndf
nrmndf = (datafr - datafr.mean()) / datafr.std()

#rounding rndf
rndndf = nrmndf.round(2)

print(rndndf)


# In[123]:


#Q13 Use the Planets dataset (taught in Pandas 4 lecture regarding groupby), count discovered planets by method and by decade

import seaborn as sns

planets = sns.load_dataset('planets')

print("Original Are: ")

print(planets.head(10))

print("\n")

planets['decade'] = (planets['year'] // 10) * 10

op = planets.groupby(['method', 'decade'])['number'].sum().unstack().fillna(0)

op.columns = [f"{int(col)}s" for col in op.columns]

print(op)


# In[ ]:




