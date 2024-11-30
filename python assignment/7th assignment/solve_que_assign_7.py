#  1. Create a scatter plot using Matplotlib to visualize the relationship between two arrays, x and y for the given 
# data.
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.scatter(x,y,color ='r')
plt.title('relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')

plt.show()



#  2. Generate a line plot to visualize the trend of values for the given data.
data = np.array([3, 7, 9, 15, 22, 29, 35])

import matplotlib.pyplot as plt
plt.plot(data)
plt.title('trend of value')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()


#  3. Display a bar chart to represent the frequency of each item in the given array categories.
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]
import matplotlib.pyplot as plt
plt.bar(categories,values,color='r',align='center')
plt.title('category wise value')
plt.xlabel('categoris')
plt.ylabel('values')
plt.show()


# 4. Create a histogram to visualize the distribution of values in the array data.
data = np.random.normal(0, 1, 1000)
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.normal(0, 1, 1000)

# Create the histogram
plt.hist(data, bins=30, color='blue')

# Add labels and a title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Random Data')

# Show the plot
plt.show()

#  5. Show a pie chart to represent the percentage distribution of different sections in the array `sections`.
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]
explode =[0.01,0.1,0.01,0.02]
plt.pie(sizes,labels=sections,autopct= '%1.1f%%',shadow=True,explode=explode,)
plt.title('distribution of sections')
plt.show()


#  1. Create a scatter plot to visualize the relationship between two variables, by generating a synthetic 
# dataset.

import seaborn as sns
sns.get_dataset_names()
health = sns.load_dataset('healthexp')
health.head()
sns.scatterplot(x=health['Year'], y=health['Life_Expectancy'],hue=health['Country'])
plt.title('Year wise Life Expectancy')
plt.show()

# 2. Generate a dataset of random numbers. Visualize the distribution of a numerical variable.
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
data = np.random.normal(1,100,1000)

sns.histplot(data, kde=True, bins=30)
plt.show()


# 3. Create a dataset representing categories and their corresponding values. Compare different categories 
# based on numerical values.
import pandas as pd
data ={
'categories' : ['Computer','Laptop','Mobile','Smartwatch'],
'Sales' : [45,93,100,123] }
data = pd.DataFrame(data)
data
sns.barplot(x=data['categories'],y=data['Sales'],color='y')
plt.title('Sales by categories')
plt.xlabel('Categories')
plt.ylabel('Sales Value')
plt.show()


#  4. Generate a dataset with categories and numerical values. Visualize the distribution of a numerical 
# variable across different categories.

import numpy as np
import pandas as pd
import seaborn as sns 

data = {
    'Categories' : ['DSA','ML','Python','ML','DSA','ML','Python'],
    'Values' : np.random.randint(1,100,7)
}
df = pd.DataFrame(data)
sns.catplot(x=df['Categories'],y=df['Values'],kind='violin')
plt.title('Categories wise value')
plt.show()
df


#  5. Generate a synthetic dataset with correlated features. Visualize the correlation matrix of a dataset using a 
# heatmap.

import seaborn as sns 
sns.get_dataset_names()
 df = sns.load_dataset('titanic')
df.head()
df =df[['survived','age','fare','pclass']]
df.corr()
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
plt.show()



#  1. Using the given dataset, to generate a 3D scatter plot to visualize the distribution of data points in a three
# dimensional space.
 np.random.seed(30)
 data = {
 'X': np.random.uniform(-10, 10, 300),
 'Y': np.random.uniform(-10, 10, 300),
 'Z': np.random.uniform(-10, 10, 300)
 }
 
 df = pd.DataFrame(data)
 
 
#  1. Using the given dataset, to generate a 3D scatter plot to visualize the distribution of data points in a three
# dimensional space.
 np.random.seed(30)
 data = {
 'X': np.random.uniform(-10, 10, 300),
 'Y': np.random.uniform(-10, 10, 300),
 'Z': np.random.uniform(-10, 10, 300)
 }
 
 df = pd.DataFrame(data)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px 

fig = go.Figure()
fig =px.scatter_3d(x=df['X'],y=df['Y'],z=df['Z'])
fig.show()


 2. Using the Student Grades, create a violin plot to display the distribution of scores across different grade 
categories.
 np.random.seed(15)
 data = {
 'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
 'Score': np.random.randint(50, 100, 200)
 }
 
 df = pd.DataFrame(data
       
                   
# 2. Using the Student Grades, create a violin plot to display the distribution of scores across different grade 
# categories.


import numpy as np
import pandas as pd
import plotly.express as px

np.random.seed(15)

# Create a DataFrame with random grades and scores
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}

df = pd.DataFrame(data)

# Create a violin plot using Plotly Express
fig = px.violin(df, x='Grade', y='Score', title='Distribution of Scores by Grade')

# Show the plot
fig.show()



# 3. Using the sales data, generate a heatmap to visualize the variation in sales across different months and 
# days.
 np.random.seed(20)
 data = {
 'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
 'Day': np.random.choice(range(1, 31), 100),
 'Sales': np.random.randint(1000, 5000, 100)
 }
df = pd.DataFrame(data)
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
fig = px.imshow(df)
fig.show()



 np.random.seed(20)
 data = {
 'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
 'Day': np.random.choice(range(1, 31), 100),
 'Sales': np.random.randint(1000, 5000, 100)
 }
df = pd.DataFrame(data)
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
fig = px.density_heatmap(df,x='Month',y='Day',z='Sales',title='Heat map char of data',)
fig.show()



# 4. Using the given x and y data, generate a 3D surface plot to visualize the function 
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))
data = {
 'X': x.flatten(),
 'Y': y.flatten(),
 'Z': z.flatten()
 }
df = pd.DataFrame(data)
import numpy as np
import plotly.graph_objects as go

# Generate the data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create a 3D surface plot
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

# Set the title and axis labels
fig.update_layout(title='3D Surface Plot', xaxis_title='X', yaxis_title='Y', zaxis_title='Z')

# Show the plot
fig.show()



import numpy as np
impor
import bokeh.plotting as bp

# Generate x and y values for the sine wave
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# Create a Bokeh figure
fig = bp.figure(title="Sine Wave")

# Add a line plot to the figure
fig.line(x, y, line_width=2)

# Show the plot in a browser window
bp.show()



