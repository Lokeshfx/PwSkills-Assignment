#1. Demonstrate three different methods for creating identical 2D arrays in NumPy. Provide the code for each 
# method and the final output after each method.
import numpy as np
arr1 = np.array([[1,2,3],[3,4,5]])
arr1.ndim

np.ones((3,3))
np.zeros((3,3))
np.full((3,3),4)


#2.  Using the Numpy function, generate an array of 100 evenly spaced numbers between 1 and 10 and 
# Reshape that 1D array into a 2D array.

arr = np.linspace(1,10,100)
arr1 = arr.reshape(10,10)
arr1.ndim


#  Explain the following terms
#  The difference in np.array, np.asarray and np.asanyarray


#4.  Generate a 3x3 array with random floating-point numbers between 5 and 20. Then, round each number in 
# the array to 2 decimal places.
import pandas as pd
arr = np.arange(5,20)
arr = np.random.random_sample((5,20,3))
np.random.uniform(5,20,(3,3))

arr.round(2)



#5. Create a NumPy array with random integers between 1 and 10 of shape (5, 6). After creating the array.
arr = np.random.randint(1,10,(5,6))
arr

# perform the following operations:
#  a)Extract all even integers from array.

even_arr = arr[arr%2==0 ]



#6. Create a 3D NumPy array of shape (3, 3, 3) containing random integers between 1 and 10. Perform the 
# following operations:

ar =np.random.randint(1,11,(3,3,3))

arr = np.argmax(ar, axis=2)
arr

ar *arr
#  Find the indices of the maximum values along each depth level (third axis).
arr = np.argmax(arr,axis=2)
import numpy as np

# Create a 3D NumPy array of shape (3, 3, 3) with random integers between 1 and 10
array = np.random.randint(1, 11, size=(3, 3, 3))

# a) Find the indices of the maximum values along each depth level (third axis)
max_indices = np.argmax(array, axis=2)

# b) Perform element-wise multiplication of the array with itself
multiplied_array = array * array

# Print the original array, max indices, and multiplied array
print("Original 3D array:\n", array)
print("\nIndices of maximum values along each depth level:\n", max_indices)
print("\nElement-wise multiplied array:\n", multiplied_array)




#8.  Clean and transform the 'Phone' column in the sample dataset to remove non-numeric characters and 
# convert it to a numeric data type. Also display the table attributes and data types of each column
import numpy as np
# import pandas as pd
# np.random.sample()
# sample_data = {
#     'Phone' : [ 12345,'Name',77484,49392,29290]
# }
# df = pd.DataFrame(sample_data)
# df[df.dtypes[df.dtypes=='object'].index].describe()

# df.describe(include='object')
# df.describe()
# df.astype('object').describe()
# df['New_col'] = 'Name_of_Stu'
# df.info()
# pd.Categorical(df['Phone'])
# df.New_col.value_counts()

# df[df['Phone'] < 33333]

# df[df['Phone'] == 0]['Phone'] = 1
# df
# import pandas as pd
df = pd.read_csv('D:\\Laptop data\\pw_skills_Data_Analysis\\python\\python assignment\\8th assignment\\People Data.csv')

# df.head()
# df.describe()
# df.dtypes
# df['Phone'].value_counts(normalize=True)*100

# df.columns # Gender, Email, Phone, User id Job Title, Salary

# len(df[df['Gender']== 'Female'])/len(df)*100
# df['Gender']
# import numpy as np
# len(df[df['Phone'] == np.nonzero])/len(df)

# df.info()
# df.shape
# df.isna().sum()

# df.dtypes[df.dtypes == 'object'].index
# df[df.dtypes[df.dtypes == 'object'].index].describe()

# df.describe(include= 'object')
# df.describe(include='all')
# df.describe()
# df.astype('object').describe()
# pd.Categorical(df['Phone'])

# df[df['Phone'].nunique].empty()

# df['Phone'] 

# len(df[df['Salary'] < 60000])
# df[df['Salary']>60000]['First Name']
# df.columns 
# df['Salary'].mean()
# df.describe()
# df.Salary.median()

# len(df[df['Salary'] <df['Salary'].mean()])

# len(df[df['Salary']==50000])
# len(df[df['Gender'] =='Male'])

# df['Gender'].value_counts(normalize=True)*100

# df['Gender'].value_counts()/10
# df[df['Salary']] > (df[df['Gender'] =='Female']['Salary'].mean())

# len(df[(df['Gender'] == 'Female') & (df['Salary'] > df['Salary'].mean())])

# np.mean(df['Salary'])

# df[df['Salary'] ==max(df.Salary)]['First Name']


# import warnings
# warnings.filterwarnings('ignore')

# df['Salary'].apply(len)
# len('Rama')
# df['First Name'].apply(len)
# df['Salary'].apply()
# df['Salary'].max()
# df['Salary'].describe()
# df['Phone'].describe()
# df.rolling(window=2).mean()

df.dtypes
df.dtypes
df['Phone'] = df['Phone'].fillna('0000000000')
len(df[df['Phone'].isnull()])

df['Phone'] =  df['Phone'].str.replace(r'\D','',regex=True)
df['Phone'].dtypes
df.dtypes
df['Phone'] = pd.to_numeric(df['Phone'])




import pandas as pd

df = pd.read_csv('People Data.csv')

# Function to clean and convert 'Phone' column, including handling null values
def clean_phone_column(df):
    # Fill null or empty values with a default phone number (e.g., '0000000000')
    df['Phone'] = df['Phone'].fillna('0000000000').replace('', '0000000000')
    
    # Remove non-numeric characters using regex (including dots, parentheses, dashes)
    df['Phone'] = df['Phone'].str.replace(r'\D', '', regex=True)
    
    # Convert the 'Phone' column to numeric (integer type)
    df['Phone'] = pd.to_numeric(df['Phone'], errors='coerce')  # 'coerce' will handle any remaining issues by setting invalid parsing to NaN
    
    return df

# Clean the 'Phone' column
df = clean_phone_column(df)

# Display the cleaned DataFrame
print("Cleaned DataFrame:\n", df)

# Display table attributes and data types
print("\nTable Attributes and Data Types:")
print(df.dtypes)


#  Perform the following tasks using people dataset:

#  a) Read the 'data.csv' file using pandas, skipping the first 50 rows.

#  b) Only read the columns: 'Last Name', ‘Gender’,’Email’,‘Phone’ and ‘Salary’ from the file.

#  c) Display the first 10 rows of the filtered dataset.

#  d) Extract the ‘Salary’' column as a Series and display its last 5 values.

df = pd.read_csv('D:\\Laptop data\\pw_skills_Data_Analysis\\python\\python assignment\\8th assignment\\People Data.csv',skiprows=50)
df

df.head(50)
df[['Last Name','Gender','Email','Phone','Salary']].head.(10)
pd.Series(df['Salary']).tail(5)
df['Phone'].tail()



# 9. Filter and select rows from the People_Dataset, where the “Last Name' column contains the name
# 'Duke',  'Gender' column contains the word Female and ‘Salary’ should be less than 85000.
import pandas as pd

df[(df['Last Name'] == 'Duke') & (df['Gender']=='Female') & (df['Salary']<85000)]



# 10. Create a 7*5 Dataframe in Pandas using a series generated from 35 random integers between 
# 1 to 6?.
import pandas as pd
import numpy as np

random_no = np.random.randint(1,7,35)
random_no
reshape_no = random_no.reshape(7,5)
reshape_no
data_frame = pd.DataFrame(reshape_no)
data_frame.dtypes



# 11. Create two different Series, each of length 50, with the following criteria: 
# a) The first Series should contain random numbers ranging from 10 to 50.
#  b) The second Series should contain random numbers ranging from 100 to 1000.
#  c) Create a DataFrame by joining these Series by column, and, change the names of the columns to 'col1', 'col2', etc

first_sr = pd.Series(np.random.randint(10,51,50))

second_sr = pd.Series(np.random.randint(100,1001,50))
df = pd.concat([first_sr,second_sr],axis=1)
df.columns = ('col1','col2')
df


# 12. Perform the following operations using people data set:
#  a) Delete the 'Email', 'Phone', and 'Date of birth' columns from the dataset. 
# b) Delete the rows containing any missing values. 
# d) Print the final output also.
df = pd.read_csv('D:\\Laptop data\\pw_skills_Data_Analysis\\python\\python assignment\\8th assignment\\People Data.csv')
df.columns

df1 = df.drop(['Email','Phone','Date of birth'],axis=1,)
print(df1.columns)



# 13. Create two NumPy arrays, x and y, each containing 100 random float values between 0 and 1.
# Perform the following tasks using Matplotlib and NumPy:
#  a) Create a scatter plot using x and y, setting the color of the points to red and the marker style to 'o'. 
# b) Add a horizontal line at y = 0.5 using a dashed line style and label it as 'y = 0.5'. 
# c) Add a vertical line at x = 0.5 using a dotted line style and label it as 'x = 0.5'. 
# d) Label the x-axis as 'X-axis' and the y-axis as 'Y-axis'.
#  e) Set the title of the plot as 'Advanced Scatter Plot of Random Values'. f) Display a legend for the scatter plot, the horizontal line, and the vertical line.
import numpy as np
import matplotlib.pyplot as plt

# Create two NumPy arrays with 100 random float values between 0 and 1
x = np.random.rand(100)
y = np.random.rand(100)

# Create a scatter plot
plt.scatter(x, y, color='red', marker='o', label='Random Points')

# Add horizontal and vertical lines
plt.axhline(y=0.5, color='green', linestyle='--', label='y = 0.5')
plt.axvline(x=0.5, color='blue', linestyle=':', label='x = 0.5')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Advanced Scatter Plot of Random Values')

# Display legend
plt.legend()

# Show the plot
plt.show()


# 14. Create a time-series dataset in a Pandas DataFrame with columns: 
# 'Date', 'Temperature', 'Humidity' and Perform the following tasks using Matplotlib:
# a) Plot the 'Temperature' and 'Humidity' on the same plot with different
# y-axes (left y-axis for 'Temperature' and right y-axis for 'Humidity'). 
# b) Label the x-axis as 'Date'. c) Set the title of the plot as 'Temperature
# and Humidity Over Time.


