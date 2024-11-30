# 5. Using the given dataset, create a bubble chart to represent each country's population (y-axis), GDP (x
# axis), and bubble size proportional to the population.
np.random.seed(25)
data = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
    'Population': np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000, 5)
    }
 
df = pd.DataFrame(data)
df
import plotly.graph_objects as go
import plotly.express as px 
fig = go.Figure()
fig.add_trace(go.sca)
# fig.add_trace(go.Scatter(x=df.GDP,y=df.Country ,z=df.Population, mode='markers',marker_size =12*df.Population))
# fig = px.scatter_3d(df,x='GDP',y='Population',size='Population',color='Country')
ig = px.scatter(df, x='GDP', y='Population', size='Population', color='Country',
                 title='Bubble Chart of Population and GDP')
fig.show()




import numpy as np
import pandas as pd
import plotly.express as px

np.random.seed(25)

# Create a DataFrame with random data
data = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
    'Population': np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000, 5)
}

df = pd.DataFrame(data)

# Create a bubble chart using Plotly
fig = px.scatter(df, x='GDP', y='Population', size='Population', color='Country',
                 title='Bubble Chart of Population and GDP')

# Show the bubble chart
fig.show()



#  1.Create a Bokeh plot displaying a sine wave. Set x-values from 0 to 10 and y-values as the sine of x.
 
 
 import bokeh.io
import bokeh.plotting
bokeh.io.output_notebook()

from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure, output_file, show
 
 
 import numpy as np
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

import numpy as np
import bokeh.plotting as bp

# Generate x and y values for the sine wave
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# Create a Bokeh figure
fig = bp.figure(title="Sine Wave")

# Add a line plot to the figure
fig.line(x, y, line_width=2)

# Show the plot in a browser window with the created figure as the argument
bp.show(fig)




#  2.Create a Bokeh scatter plot using randomly generated x and y values. Use different sizes and colors for the 
# markers based on the 'sizes' and 'colors' columns.
import numpy as np
import bokeh.plotting as bp

# Generate random x and y values
x = np.random.rand(100)
y = np.random.rand(100)

# Generate random sizes and colors
sizes = np.random.randint(10, 50, 100)
colors = np.random.choice(['red', 'blue', 'green'], 100)

# Create a Bokeh figure
fig = bp.figure(title="Scatter Plot with Different Sizes and Colors")

# Add a scatter plot to the figure
fig.circle(x, y, size=sizes, color=colors, alpha=0.7)

# Show the plot with the created figure as the argument
bp.show(fig)




from bokeh.plotting import figure, output_file, show
import numpy as np
import random
import webbrowser

# Generate random data for x and y values
x = np.random.rand(50) * 100
y = np.random.rand(50) * 100

# Generate random sizes and colors
sizes = np.random.rand(50) * 50 + 10  # sizes between 10 and 60
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(50)]

# Create a scatter plot
p = figure(title="Bokeh Scatter Plot with Random Data", x_axis_label='X Axis', y_axis_label='Y Axis')

# Add scatter renderer
p.scatter(x, y, size=sizes, color=colors, fill_alpha=0.6)

# Save the plot to an HTML file
output_file("scatter_plot.html")

# Show the plot in the browser
show(p)

# (Optional) Open the saved file explicitly with webbrowser module
webbrowser.open("scatter_plot.html")
