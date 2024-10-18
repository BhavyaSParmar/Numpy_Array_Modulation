# Numpy_Array_Modulation
The provided Python script is a comprehensive set of exercises and tasks demonstrating data manipulation and analysis using the Pandas library, with some additional functionality from NumPy and Seaborn. Below is a detailed description of each section of the script:

1. Generating a DataFrame with Random Integers and Alphabet Index (Q1)
The script creates a DataFrame with the 26 letters of the alphabet as the index and 26 random integers between 1 and 100 as the data. The first five rows are printed to showcase the results.

2. Constructing a DataFrame and Extracting Series Data (Q2)
A DataFrame is constructed with three columns, class1, class2, and class3, populated with numerical values.
The first column is extracted as a Series using the iloc method.

3. Indexing and Slicing the DataFrame (Q3)
Various tasks are performed using iloc and loc:
Selecting all columns except class3.
Removing the first three rows.
Removing the last three rows.

4. Creating a MultiIndex DataFrame (Q4)
A multi-index DataFrame is created, representing a data structure for tracking Heart Rate (HR) and Temperature (Temp) data for three individuals (Bob, Julia, Sue) over different years (2018, 2019).
The script performs operations to filter data for Julia and Bob, specifically retrieving heart rate data and filtering by year.

5. DataFrame with Nearest Row Calculations (Q5)
A DataFrame is created with random values, and a new column is added to indicate the nearest row for each entry, calculated using the Euclidean distance.

6. Generating a DataFrame with Rows as Strides from a Series (Q6)
Creates a DataFrame by taking strides from a given Series, effectively converting a continuous range into row-based segments.

7. Manipulating DataFrame Columns (Q7)
The script demonstrates column manipulation techniques:
Creating a DataFrame with specified column names.
Interchanging columns a and c.
Implementing a generic function to interchange arbitrary columns.
Sorting columns in reverse alphabetical order.

8. Creating a Time Series with Specific Dates (Q8)
A Time Series is generated for 10 consecutive Saturdays starting from '2021-01-01', with random numbers as values.

9. Handling Missing Dates in a Time Series (Q9)
The script creates a time series with missing dates, then fills in the missing dates by forward-filling values from previous dates.

10. Working with a DataFrame of Fruits (Q10)
A DataFrame is created with columns for fruit, taste, and price. Tasks include:
Finding the second-largest taste value for bananas.
Calculating the mean price of each fruit.

11. Calculating the Second-Largest Value per Row (Q11)
For a given DataFrame, a new column is added that contains the second-largest value for each row.

12. Normalizing a DataFrame (Q12)
The DataFrame is normalized by subtracting the column mean and dividing by the standard deviation, with results rounded to two decimal places.

13. Analyzing the Planets Dataset (Q13)
Uses the Seaborn planets dataset to count the number of discovered planets by the discovery method and decade.

Summary:
This script provides a thorough exercise in data manipulation, cleaning, transformation, and analysis using Pandas. It covers essential operations like indexing, slicing, aggregating, creating multi-index DataFrames, handling missing data, normalizing datasets, & performing exploratory data analysis on built-in datasets like planets. These tasks are valuable for anyone learning data science or data analysis using Python.
