import pandas as pd
import numpy as np

import numpy as np
x = np.random.rand(100)
type(x)
y = np.random.rand(100)
import matplotlib.pyplot as plt

# b) Add a horizontal line at y = 0.5 using a dashed line style and label it as 'y = 0.5'. 
# c) Add a vertical line at x = 0.5 using a dotted line style and label it as 'x = 0.5'. 
# d) Label the x-axis as 'X-axis' and the y-axis as 'Y-axis'.
#  e) Set the title of the plot as 'Advanced Scatter Plot of Random Values'. f) Display a legend for the scatter plot, the horizontal line, and the vertical line.
plt.scatter(x,y,color = 'red',marker='o',label = 'Random Points')
plt.axhline(y=0.5,linestyle = '--',label ='y= 0.5')
plt.axvline(x=0.5,linestyle =':',color='Green',label = 'x=0.5')
plt.show()
plt.scatter(x,y,linewidths=0.5)
plt.xlabel('ylable')
plt.ylabel('xlable')
plt.title('Advanced Scatter plot of random values')
plt.show()

# Create a time-series dataset in a Pandas DataFrame with columns: 'Date', 'Temperature', 'Humidity'
# and Perform the following tasks using Matplotlib:
# a) Plot the 'Temperature' and 'Humidity' on the same plot with different y-axes (left y-axis for 
# 'Temperature' and right y-axis for 'Humidity'). b) Label the x-axis as 'Date'. c) Set the title 
# of the plot as 'Temperature and Humidity Over Time.
data = pd.DataFrame({
    'Date' : pd.date_range(start='12-02-2024',periods=365),
    'Temp' : np.random.randint(10,45,365),
    'Humidity' : np.random.randint(30,95,365)
})
data
plt.plot(data['Date'])
plt.plot


import pandas as pd
import matplotlib.pyplot as plt

# Sample time-series dataset
data = {
    'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'Temperature': [30, 32, 31, 29, 28, 35, 34, 33, 31, 30],
    'Humidity': [65, 70, 72, 68, 67, 66, 64, 63, 62, 60]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting with two y-axes
fig, ax1 = plt.subplots()

# Plot Temperature on the left y-axis
ax1.plot(df['Date'], df['Temperature'], color='tab:red', label='Temperature')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for Humidity
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Humidity'], color='tab:blue', label='Humidity')
ax2.set_ylabel('Humidity (%)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Set the title
plt.title('Temperature and Humidity Over Time')

# Show plot
plt.show()



# 15. Create a NumPy array data containing 1000 samples from a normal distribution. Perform the following tasks using Matplotlib:
# a) Plot a histogram of the data with 30 bins. 
# b) Overlay a line plot representing the normal distribution's probability density function (PDF). 
# c) Label the x-axis as 'Value' and the y-axis as 'Frequency/Probability'.
#  d) Set the title of the plot as 'Histogram with PDF Overlay'.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
arr = np.random.normal(1000)
plt.plot(arr,type='line')
plt.hist(arr,30)
plt.show()


# 17. Create a Seaborn scatter plot of two random arrays, color points based on their position relative to the 
# origin (quadrants), add a legend, label the axes, and set the title as 'Quadrant-wise Scatter Plot'.
import seaborn as sns
arr1 = np.random.rand(50)
arr2 = np.random.rand(50)
sns.scatterplot(x=arr1,y=arr2)
plt.show()



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate two random arrays for x and y coordinates
np.random.seed(0)  # For reproducibility
x = np.random.uniform(-10, 10, 100)
y = np.random.uniform(-10, 10, 100)

# Determine quadrant for each point
def get_quadrant(x, y):
    if x > 0 and y > 0:
        return 'Quadrant I'
    elif x < 0 and y > 0:
        return 'Quadrant II'
    elif x < 0 and y < 0:
        return 'Quadrant III'
    elif x > 0 and y < 0:
        return 'Quadrant IV'
    else:
        return 'On Axis'

# Create DataFrame for plotting
df = pd.DataFrame({'x': x, 'y': y})
df['Quadrant'] = [get_quadrant(xi, yi) for xi, yi in zip(df['x'], df['y'])]

# Create Seaborn scatter plot with color based on quadrant
plt.figure(figsize=(8, 6))
scatter_plot = sns.scatterplot(data=df, x='x', y='y', hue='Quadrant', palette='tab10', s=70, edgecolor='k')

# Label axes and set the title
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Quadrant-wise Scatter Plot')
plt.axhline(0, color='black',linewidth=0.5)  # Add origin lines
plt.axvline(0, color='black',linewidth=0.5)
plt.legend(title='Position Relative to Origin')

# Show plot
plt.show()



# 18. With Bokeh, plot a line chart of a sine wave function, add grid lines, label the axes, and 
# set the title as 'Sine Wave Function'.

from bokeh.plotting import figure, show, output_notebook
import numpy as np

# Display plot inline in notebook
output_notebook()

# Generate data for the sine wave
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

# Create a Bokeh figure
p = figure(title="Sine Wave Function", width=700, height=400)

# Plot the sine wave
p.line(x, y, line_width=2, color="blue", legend_label="y = sin(x)")

# Add grid lines (enabled by default)
p.grid.grid_line_color = "gray"

# Label the axes
p.xaxis.axis_label = "X-Axis"
p.yaxis.axis_label = "Y-Axis"

# Show the plot
show(p)




from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
import pandas as pd
import numpy as np

# Display plot inline in notebook
output_notebook()

# Generate random categorical data
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 100, size=len(categories))

# Create a DataFrame
df = pd.DataFrame({'Category': categories, 'Value': values})

# Create a ColumnDataSource
source = ColumnDataSource(df)

# Create a Bokeh figure for the bar chart
p = figure(x_range=categories, title="Random Categorical Bar Chart", width=700, height=400)

# Color map based on values
color_mapper = factor_cmap('Category', palette="Viridis256", factors=categories)

# Plot the bars
p.vbar(x='Category', top='Value', width=0.5, source=source, color=color_mapper)

# Add hover tool to show exact values
hover = HoverTool()
hover.tooltips = [("Category", "@Category"), ("Value", "@Value")]
p.add_tools(hover)

# Label the axes
p.xaxis.axis_label = "Category"
p.yaxis.axis_label = "Value"

# Show the plot
show(p)


20. Using Plotly, create a basic line plot of a randomly generated dataset, label the axes, and set the title as 'Simple Line Plot'.

import plotly.graph_objects as go
import numpy as np

# Generate random data for the line plot
x = np.arange(0, 50)
y = np.random.randint(0, 100, size=50)

# Create a Plotly line plot
fig = go.Figure()

# Add the line trace
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Random Data'))

# Label the axes and set the title
fig.update_layout(
    title='Simple Line Plot',
    xaxis_title='X-Axis',
    yaxis_title='Y-Axis'
)

# Show the plot
fig.show()


21. Using Plotly, create an interactive pie chart of randomly generated data, add labels and percentages, set the title as 'Interactive Pie Chart'.

import plotly.graph_objects as go
import numpy as np

# Generate random data for the pie chart
labels = ['Category A', 'Category B', 'Category C', 'Category D']
values = np.random.randint(10, 100, size=len(labels))

# Create a Plotly pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0)])

# Add title and enable display of percentages
fig.update_traces(textinfo='percent+label')

# Set the title
fig.update_layout(title='Interactive Pie Chart')

# Show the plot
fig.show()
