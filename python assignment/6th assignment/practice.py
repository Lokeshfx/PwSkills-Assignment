# 1. Create a NumPy array 'arr' of integers from 0 to 5 and print its data type.
import numpy as np
arr = np.arange(6) #Create an array from 0 to 5 inclusive
print(arr)
print(arr.dtype)


# 2. Given a NumPy array 'arr', check if its data type is float64. 
arr = np.array([1.5, 2.6, 3.7])
if arr.dtype == np.float64:
    print("This data type of arr is float64")
else:
    print("This data type of arr is not float64")
    
#  3. Create a NumPy array 'arr' with a data type of complex128prin containing three complex numbers.

arr = np.array([3+5j,9+8j,2+1j], dtype= np.complex128)
print(arr)
print(arr.dtype)
    

# 4. Convert an existing NumPy array 'arr' of integers to float32 data type

arr = np.array([2,3,5,6,8]) #create integers array
print(arr.dtype) #output int64

arr_float = arr.astype(np.float32) #convert integers to float32
print(arr_float)
print(arr_float.dtype) #output float32



# 5. Given a NumPy array 'arr' with float64 data type, convert it to float32 to reduce decimal precision.

arr = np.array([1.3,3.5,32.2],dtype=np.float64)
print(arr.dtype)

arr_float32 = arr.astype(np.float32)
print(arr_float32)
print(arr_float32.dtype)


import numpy as np

arr = np.array([1.23456789, 2.34567890, 3.45678901], dtype=np.float64)
print(arr.dtype)  # Output: float64

# Convert the array to float32
arr_float32 = arr.astype(np.float32)
print(arr_float32.dtype)  # Output: float32


#  6. Write a function array_attributes that takes a NumPy array as input and returns its shape, size, and data 
def array_attributes(arr):
    shape = arr.shape
    size = arr.size
    data = arr.tolist()
    
    return shape, size, data
arr = np.array([[1,2,3,4],[33,4,2,8]])

shape, size, data = array_attributes(arr)
print('Shape:', shape)
print('Size :',size)
print('data: ',data)

import numpy as np

def array_attributes(arr):
  """Returns the shape, size, and data of a NumPy array.

  Args:
    arr: The NumPy array.

  Returns:
    A tuple containing the shape, size, and data of the array.
  """

  shape = arr.shape
  size = arr.size
  data = arr.tolist()

  return shape, size, data

# Example usage
arr = np.array([[1, 2, 3], [4, 5, 6]])
shape, size, data = array_attributes(arr)
print("Shape:", shape)
print("Size:", size)
print("Data:", data)



# 7. Create a function array_dimension that takes a NumPy array as input and returns its dimensionality.

def array_dimension(arr):
    return arr.ndim

arr = np.array([33,4,5,6])

arr_dimension = array_dimension(arr)
arr_dimension

import numpy as np

def array_dimension(arr):
  """Returns the dimensionality of a NumPy array.

  Args:
    arr: The NumPy array.

  Returns:
    The dimensionality of the array.
  """

  return arr.ndim

# Example usage
arr = np.array([[1, 2, 3], [4, 5, 6]])
dimension = array_dimension(arr)
print("Dimensionality:", dimension)


#  8. Design a function item_size_info that takes a NumPy array as input and returns the item size and the total 
# size in bytes

def item_size_info(arr):
    item_size = arr.itemsize
    tot_size = arr.nbytes
    return item_size, tot_size
arr = np.array([1,2,3,4,5])
item_size, tot_size = item_size_info(arr)
arr
print('item size:' ,item_size ,"bytes")
print('tot size :',tot_size,'bytes')


import numpy as np

def item_size_info(arr):
  """Returns the item size and total size in bytes of a NumPy array.

  Args:
    arr: The NumPy array.

  Returns:
    A tuple containing the item size and total size in bytes.
  """

  item_size = arr.itemsize
  total_size = arr.nbytes

  return item_size, total_size

# Example usage
arr = np.array([[1, 2, 3], [4, 5, 6]])
item_size, total_size = item_size_info(arr)
print("Item size:", item_size, "bytes")
print("Total size:", total_size, "bytes")

# 9. Create a function array_strides that takes a NumPy array as input and returns the strides of the array.

def array_strides(arr):
    strides = arr.strides
    return strides
arr = np.array([3,4,5,6])
strides = array_strides(arr)
print('stridess:' ,strides)

# 10. Design a function shape_stride_relationship that takes a NumPy array as input and returns the shape 
# and strides of the array

def shape_stride_relationship(arr):
    shape = arr.shape
    stridess = arr.strides
    return shape, stridess
arr = np.array([1,2,3,4,5])
shape, stridess = shape_stride_relationship(arr)
print('strides :',stridess)
print('shape :', shape)


# 11. Create a function `create_zeros_array` that takes an integer `n` as input and returns a NumPy array of 
# zeros with `n` elements

def crate_zeros_array(n):
    return np.zeros(n)
   
n = 10 
zeros_array = crate_zeros_array(n)
print(zeros_array)


# 12. Write a function `create_ones_matrix` that takes integers `rows` and `cols` as inputs and generates a 2D 
# NumPy array filled with ones of size `rows x cols`.

def create_ones_matrix(rows,cols):
    return np.ones((rows,cols))

rows = 4
cols = 5
nump_arr = create_ones_matrix(rows,cols)
print(nump_arr)


# 13. Write a function `generate_range_array` that takes three integers start, stop, and step as arguments and 
# creates a NumPy array with a range starting from `start`, ending at stop (exclusive), and with the specified 
# `step`.

import numpy as np  
def generate_range_array(start,stop,step):
  return np.arange(start,stop,step)
start = 1
stop = 10
step = 2

arr = generate_range_array(start,stop,step)
print(arr)


#  14. Design a function `generate_linear_space` that takes two floats `start`, `stop`, and an integer `num` as 
# arguments and generates a NumPy array with num equally spaced values between `start` and `stop` 
# (inclusive).

def generate_linear_space(start,stop,num):
  return np.linspace(start,stop,num)
start = 1
stop = 20
num = 8
arr = np.linspace(1,19,5)
print(arr)

#  15. Create a function `create_identity_matrix` that takes an integer `n` as input and generates a square 
# identity matrix of size `n x n` using `numpy.eye`.

import numpy.matlib as nm


def crate_identity_matrix(n):
  return np.eye(n)
n = 2
matrix = crate_identity_matrix(n)
print(matrix)


#  16. Write a function that takes a Python list and converts it into a NumPy array

def take_list(list):
  # list = []
  return np.array(list)
lis = [1,2,3,4,5]

np_arr = take_list(lis)
print(np_arr)
type(np_arr)


# 17. Create a NumPy array and demonstrate the use of `numpy.view` to create a new array object with the 
# same data.

arr = np.array([1,2,3,4,5])
arr_new = arr.view()
print('original array :',arr)
print('new array :',arr_new)

arr[0] = 100
print('modify original array :',arr)
print('array :',arr_new)


# 18. Write a function that takes two NumPy arrays and concatenates them along a specified axis

def concat_two_arr(arr1, arr2,axis = 0):
  return np.concatenate((arr1, arr2) , axis=axis)

arr1 = np.array([1,2,3,4])
arr2 = np.array([3,4,5,6,])
ar = concat_two_arr(arr1,arr2)
print(ar)

#  19.  Create two NumPy arrays with different shapes and concatenate them horizontally using `numpy.
#  concatenate`.

arr1 = np.array([[1,2,10],[3,4,5]])
arr2 = np.array([[5,6,3],[8,9,3]])
concat = np.concatenate((arr1,arr2),axis=1)
print(concat)

#  20. Write a function that vertically stacks multiple NumPy arrays given as a list

import numpy as np

def vertical_stack(arrays):
  """Vertically stacks multiple NumPy arrays.

  Args:
    arrays: A list of NumPy arrays.

  Returns:
    A new NumPy array containing the vertically stacked arrays.
  """

  return np.vstack(arrays)

# Example usage
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8], [9, 10]])
arr3 = np.array([[11, 12]])

stacked_arr = vertical_stack([arr1, arr2, arr3])
print(stacked_arr)

def vertically_stacks(arrays):
  return np.vstack(arrays)

arrr = vertically_stacks([arr1, arr2, arr3])
print(arrr)



# 21. Write a Python function using NumPy to create an array of integers within a specified range (inclusive) 
# with a given step size.

def create_int_arr(start,stop,step=1):
  """ Create a numpy array of integers within a specified range
  Args:
      start : The starting value of the range
      stop : The stop value of the range (inclusive).
      step : The step size between elements 

  Returns:
      A numpy array of integers within the specified range.
  """
  
  return np.arange(start, stop+1,step)

# Example usage
start = 2
stop = 10
step = 2
integer_array = create_integer_array(start, stop, step)
print(integer_array)


#  22. Write a Python function using NumPy to generate an array of 10 equally spaced values between 0 and 1 
# (inclusive)

def equally_space_arr(start,step,num_space):
  return np.linspace(start,step,num_space)

num = equally_space_arr(0,1,10)
num


# 23. Write a Python function using NumPy to create an array of 5 logarithmically spaced values between 1 and 
# 1000 (inclusive).

def log_space_array(start,stop, num):
  return np.logspace(np.log10(start),np.log10(stop),num)

start = 1
stop = 1000
num = 5
arrr = log_space_array(start , stop, num)
print(arrr)



# 24. Create a Pandas DataFrame using a NumPy array that contains 5 rows and 3 columns, where the values 
# are random integers between 1 and 100.
import numpy as np
import pandas as pd
pdd = np.random.randint(1,100,(3,4))
pdd
pdd = pd.DataFrame(pdd)
pdd
type(pdd)


#  25. Write a function that takes a Pandas DataFrame and replaces all negative values in a specific column 
# with zeros. Use NumPy operations within the Pandas DataFrame.

def func(df,Coloumn_name):
  negative_mask = df[Coloumn_name] <0
  df[Coloumn_name] = np.where(negative_mask,0,df[Coloumn_name])
  return df
    
func(pdd,2)


#  26. Access the 3rd element from the given NumPy array.
  arr = np.array([10, 20, 30, 40, 50])
  arr[2]


#  27. Retrieve the element at index (1, 2) from the 2D NumPy array.
 arr_2d = np.array([[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]])

arr_2d[0:1,:2]

# 28. Using boolean indexing, extract elements greater than 5 from the given NumPy array.
 arr = np.array([3, 8, 2, 10, 5, 7])
 mask = arr<5
 print(mask)
 
 
#  29. Perform basic slicing to extract elements from index 2 to 5 (inclusive) from the given NumPy array.
 arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
 arr[2:6]
 
 
#  30. Slice the 2D NumPy array to extract the sub-array `[[2, 3], [5, 6]]` from the given array.
 arr_2d = np.array([[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]])

arr_2d[0:2,1:3]


# 31.Write a NumPy function to extract elements in specific order from a given 2D array based on indices 
# provided in another array.
import numpy as np

def extract_elements(array, indices):
  """Extracts elements from a 2D NumPy array based on indices.

  Args:
    array: The 2D NumPy array.
    indices: A 2D NumPy array containing the row and column indices of the elements to extract.

  Returns:
    A 1D NumPy array containing the extracted elements.
  """

  return array[indices[:, 0], indices[:, 1]]

# Example usage
array = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

indices = np.array([[0, 1],
                   [1, 2]])

extracted_elements = extract_elements(array, indices)
print(extracted_elements)

# 32. Create a NumPy function that filters elements greater than a threshold from a given 1D array using 
# boolean indexing.

def filter_value(array, threshold):
  mask = array >threshold
  return mask

array = np.array([1,2,3,4,5])
threshold = 3
filter_value(array, threshold)

np.array(array) <np.array(threshold)

# 33. Develop a NumPy function that extracts specific elements from a 3D array using indices provided in three 
# separate arrays for each dimension.

def extract_element_3d(array, row_indices, column_indices,depth_indices):
  return array(row_indices,column_indices,depth_indices)

arr = np.random.randint(1,8,(3,3))
array
row_indices = np.array([0,1])
column_indices = np.array([1,2])
depth_indices = np.array([0,1])
extract_element_3d(array, row_indices,column_indices,depth_indices)


import numpy as np

def extract_elements_3d(array, row_indices, col_indices, depth_indices):
  """Extracts elements from a 3D NumPy array based on indices.

  Args:
    array: The 3D NumPy array.
    row_indices: A 1D NumPy array containing the row indices.
    col_indices: A 1D NumPy array containing the column indices.
    depth_indices: A 1D NumPy array containing the depth indices.

  Returns:
    A 1D NumPy array containing the extracted elements.
  """

  return array[row_indices, col_indices, depth_indices]

# Example usage
array = np.array([[[1, 2, 3],
                   [4, 5, 6]],
                  [[7, 8, 9],
                   [10, 11, 12]]])

row_indices = np.array([0, 1])
col_indices = np.array([1, 2])
depth_indices = np.array([0, 1])

extracted_elements = extract_elements_3d(array, row_indices, col_indices, depth_indices)
print(extracted_elements)


#  34. Write a NumPy function that returns elements from an array where both two conditions are satisfied 
# using boolean indexing.

def both_cond_satisfy(array,condition1, condition2):
  mask = condition1 & condition2:
    return array[array]
  


# 35. Create a NumPy function that extracts elements from a 2D array using row and column indices provided 
# in separate arrays
def extract_elements2d(array, row_indices, column_indices):
  mask = array(row_indices,column_indices)
  return mask

extract_elements2d(arr,[1:2],[0:2])


#  36. Given an array arr of shape (3, 3), add a scalar value of 5 to each element using 
# NumPy broadcasting.

#creat a 3x3 numpy array 
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
# adding 5 to each of element
result = arr+5
#result 
print(result) 



#  37. Consider two arrays arr1 of shape (1, 3) and arr2 of shape (3, 4). Multiply each row of arr2 by the 
# corresponding element in arr1 using NumPy broadcasting.

arr1 = np.random.randint(1,5,(1,3))
arr2 = np.random.randint(1,5,(3,4))
arr3 = arr1 * arr2   


arr1 = np.array([[1,2,3]])
arr2 = np.array([[1,2,3,4],[2,3,4,5],[5,6,7,8]])
print(arr3)

import numpy as np

# Create two NumPy arrays
arr1 = np.array([[1, 2, 2]])
arr2 = np.array([[4, 5, 5, 7],
                 [8, 9, 10, 11],
                 [12, 13, 14, 15]])

# Multiply each row of arr2 by the corresponding element in arr1 using broadcasting
result = arr1 * arr2.reshape(4,3)

arr2= arr2.reshape(4,3)
print(result)


#  38. Given a 1D array arr1 of shape (1, 4) and a 2D array arr2 of shape (4, 3), add arr1 to each row of arr2 using 
# NumPy broadcasting.

arr1 = np.array([[1,2,3,4]])
arr2 = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
arrr = arr2 + arr1
print(arrr)



import numpy as np

# Create NumPy arrays
arr1 = np.array([[1, 2, 3, 4]])
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])
arr2.reshape(3,4)


# Add arr1 to each row of arr2 using broadcasting

result = arr2 + arr1.reshape(4, 1)  # Reshape arr1 to (4, 1)
print(result)



#  39. Consider two arrays arr1 of shape (3, 1) and arr2 of shape (1, 3). Add these arrays using NumPy 
# broadcasting.
import numpy as np
arr1 = np.array([[1],[2],[3]])
arr2 = np.array([[1,2,3]])
arr = arr1+arr2
arr


# 40. Given arrays arr1 of shape (2, 3) and arr2 of shape (2, 2), perform multiplication using NumPy 
# broadcasting. Handle the shape incompatibility.

arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[2,3],[4,5]])
ar = arr2.reshape(2,3)
arr= arr1.reshape(1,2)*arr2


import numpy as np

# Create NumPy arrays
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr2 = np.array([[7, 8],
                 [9, 10]])

# Reshape arr2 to (2, 1)
arr2_reshaped = arr2.reshape(2, 1)

# Perform element-wise multiplication
result = arr1 * arr2_reshaped

print(result)



import numpy as np

# Create NumPy arrays
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr2 = np.array([[7, 8],
                 [9, 10]])

# Reshape arr2 to (2, 1) and add a new dimension
arr2_reshaped = np.expand_dims(arr2, axis=1)

# Perform element-wise multiplication
result = arr1 * arr2_reshaped

print(result)

import numpy as np

# Create NumPy arrays
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr2 = np.array([[7, 8],
                 [9, 10]])

# Reshape arr2 to (2, 1) and add a new dimension
arr2_reshaped = np.expand_dims(arr2, axis=1)

# Perform element-wise multiplication
result = arr1 * arr2_reshaped

print(result)

#  41. Calculate column-wise mean for the given array:
 arr = np.array([[1, 2, 3], [4, 5, 6]])
 arr.mean(axis =0)
 
 
 import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Calculate column-wise mean
column_means = np.mean(arr, axis=0)

print(column_means)


# 42. Find maximum value in each row of the given array:
 arr = np.array([[1, 2, 3], [4, 5, 6]])
 arr.max(axis=1)
 
 
# 43. For the given array, find indices of maximum value in each column.
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.max(axis=0)


# 44. For the given array, apply custom function to calculate moving sum along rows.
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])

#calculate the moving sum along rows
moving_sum = arr.cumsum(axis= 1)
print(moving_sum)


# 45. In the given array, check if all elements in each column are even.
arr = np.array([[2, 4, 6], [3, 5, 7]])

check_even = np.all(arr%2 ==0,axis= 0)
check_even 


# 46. Given a NumPy array arr, reshape it into a matrix of dimensions `m` rows and `n` columns. Return the 
# reshaped matrix.
original_array = np.array([1, 2, 3, 4, 5, 6])import numpy as np

def reshape_matrix(original_array, m, n):
    """
    Reshapes a NumPy array into a matrix of specified dimensions.

    Args:
        original_array: The original NumPy array.
        m: The desired number of rows in the reshaped matrix.
        n: The desired number of columns in the reshaped matrix.

    Returns:
        The reshaped matrix.

    Raises:
        ValueError: If the product of `m` and `n` is not equal to the length of the original array.
    """

    if m * n != len(original_array):
        raise ValueError("The product of m and n must equal the length of the original array.")

    return original_array.reshape(m, n)

# Example usage:
original_array = np.array([1, 2, 3, 4, 5, 6])
m = 2
n = 3
reshaped_matrix = reshape_matrix(original_array, m, n)
print(reshaped_matrix)


# 47. Create a function that takes a matrix as input and returns the flattened array.
input_matrix = np.array([[1, 2, 3], [4, 5, 6]])
def flatten_array(array):
  return array.flatten()


#  48. Write a function that concatenates two given arrays along a specified axis.
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

import numpy as np
def concat_two_arr((arr1, arr2), axiss):
  return np.concatenate((arr1, arr2), axis=axiss )
axiss = 0
concat_two_arr((array1,array2),axis=True)
np.concat(array1,array2)
np.concatenate

a =np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
np.concatenate((a,b),axis=0)

def concattt((ar1,ar2),axis= 0):
  return np.concatenate((ar1,ar2), axis = axis)



import numpy as np

def concatenate_arrays(array1, array2, axis=0):
  """
  Concatenates two given arrays along a specified axis.

  Args:
    array1: The first array.
    array2: The second array.
    axis: The axis along which to concatenate.

  Returns:
    The concatenated array.
  """

  return np.concatenate((array1, array2), axis=axis)

# Example usage:
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Concatenate along the first axis (row-wise)
concatenated_array1 = concatenate_arrays(array1, array2, axis=0)
print(concatenated_array1)

# Concatenate along the second axis (column-wise)
concatenated_array2 = concatenate_arrays(array1, array2, axis=1)
print(concatenated_array2)



# 49. Create a function that splits an array into multiple sub-arrays along a specified axis.
original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def split_arr(arr,axis=0, sections=None):
  if sections is None:
    sections = arr.shape[axis] // 2 # default to spliting two equal parts
  return np.split(arr,sections, axis=axis)

axis=0
split_arr(original_array,3,axis)
np.split(original_array,3)


import numpy as np

def split_array(arr, axis=0, sections=None):
  """
  Splits a NumPy array into multiple sub-arrays along a specified axis.

  Args:
    arr: The input array.
    axis: The axis along which to split.
    sections: The number of sections to split the array into (optional).

  Returns:
    A list of split arrays.
  """

  if sections is None:
    sections = arr.shape[axis] // 2  # Default to splitting into two equal parts

  return np.split(arr, sections, axis=axis)

# Example usage:
original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Split into two equal parts along the second axis (column-wise)
split_arrays = split_array(original_array)
print(split_arrays)

# Split into three sections along the first axis (row-wise)
split_arrays = split_array(original_array, axis=0, sections=3)
print(split_arrays)



# 50. Write a function that inserts and then deletes elements from a given array at specified indices.
 original_array = np.array([1, 2, 3, 4, 5])
 indices_to_insert = [2, 4]
 values_to_insert = [10, 11]
 indices_to_delete = [1, 3]
 
import numpy as np
def insert_and_delete_ar(arr,indices_to_insert,values_to_insert,indices_to_delete):
  """
  Inserts and then deletes elements from a given array at specified indices.

  Args:
    arr: The input array.
    indices_to_insert: A list of indices where to insert the new elements.
    values_to_insert: A list of values to insert.
    indices_to_delete: A list of indices where to delete elements.

  Returns:
    result
  """
  array = np.insert(arr, indices_to_insert,values_to_insert)  
  result = np.delete(array,indices_to_delete)
  return result
#usage
insert_and_delete_ar(original_array,indices_to_insert,values_to_insert,indices_to_delete)
 
 
 original_array[2,4] = np.append([10,11]) 
 new_arr = np.insert(original_array,indices_to_insert,values_to_insert)
 new_ar = np.delete(new_arr,indices_to_delete)
 new_ar
 
 
#  51. Create a NumPy array `arr1` with random integers and another array `arr2` with integers from 1 to 10. 
# Perform element-wise addition between `arr1` and `arr2`.
import numpy as np

# Create array arr1 with random integers
arr1 = np.random.randint(1,100,(1,10))


# Create array arr2 with integers from 1 to 10
arr2 = np.random.randint(1,11,(1,10))
# Perform element-wise addition
result = arr1 + arr2

print("arr1:", arr1)
print("arr2:", arr2)
print("Result:", result)


#  52. Generate a NumPy array `arr1` with sequential integers from 10 to 1 and another array `arr2` with integers 
# from 1 to 10. Subtract `arr2` from `arr1` element-wise.

arr1 = np.arange(10,0,-1)
arr1
arr2 = np.arange(1,11)
arr2
result = arr1- arr2
result


# 53. Create a NumPy array `arr1` with random integers and another array `arr2` with integers from 1 to 5. 
# Perform element-wise multiplication between `arr1` and `arr2`.
import numpy as np

# Create array arr1 with random integers
arr1 = np.random.randint(10,50,(1,5))
# Create array arr2 with integers from 1 to 5
arr2 = np.arange(1,6)
# Perform element-wise multiplication
result = arr1 * arr2

print("arr1:", arr1)
print("arr2:", arr2)
print("Result:", result)


# 54. Generate a NumPy array `arr1` with even integers from 2 to 10 and another array `arr2` with integers from 1 
# to 5. Perform element-wise division of `arr1` by `arr2`.

arr1 = np.arange(2,11,2)
arr2 = np.arange(1,6)
arr2
result = arr1/arr2
result

# 55. Create a NumPy array `arr1` with integers from 1 to 5 and another array `arr2` with the same numbers 
# reversed. Calculate the exponentiation of `arr1` raised to the power of `arr2` element-wise.

arr1 = np.arange(1,6)
arr1
arr2 = np.arange(5,0,-1)
arr2
arr1**arr2


# 56. Write a function that counts the occurrences of a specific substring within a NumPy array of strings.
arr = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
 
 np.unique_counts(arr)
 
 
 import numpy as np

def count_substring_occurrences(arr, substring):
  """Counts the occurrences of a specific substring within a NumPy array of strings.

  Args:
    arr: The input NumPy array of strings.
    substring: The substring to search for.

  Returns:
    The total number of occurrences of the substring in the array.
  """

  # Use np.char.count to count occurrences of the substring in each element
  counts = np.char.count(arr, substring)

  # Sum the counts to get the total number of occurrences
  total_count = np.sum(counts)

  return total_count

# Example usage:
arr = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
substring = 'hello'
total_count = count_substring_occurrences(arr, substring)
print(total_count)  # Output: 3

def char_countt(arr, substring):
  count = np.char.count(arr,substring)
  count = sum(count)
  return count
char_countt(arr, 'hello')


#  57. Write a function that extracts uppercase characters from a NumPy array of strings.
arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])
arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])

def extract_upprcase(arr):
  for i in arr:
    if np.char.isupper(i) = True
    return i
  
  
  
  import numpy as np

def extract_uppercase(arr):
  """Extracts uppercase characters from a NumPy array of strings.

  Args:
    arr: The input NumPy array of strings.

  Returns:
    A NumPy array containing only the uppercase characters from the original array.
  """
  # Use np.char.isalpha and np.char.isupper to filter only uppercase characters

  upper = arr[np.char.isalpha(arr) & np.char.isupper(arr)]

  return upper
extract_uppercase(arr)

# Example usage:
arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])
upper = arr[np.char.isupper(arr)]
print(upper)



arr[np.char.isalpha(arr) & np.char.isupper(arr)]



# 58. Write a function that replaces occurrences of a substring in a NumPy array of strings with a new string.
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
def replace_substring(arr, old_substring, new_substring):
  return new_ar = np.char.replace(arr, old_substring, new_substring)

replace_substring(arr, 'banana','Kiwi')
arr



# 59. Write a function that concatenates strings in a NumPy array element-wise.
arr1 = np.array(['Hello', 'World'])
arr2 = np.array(['Open', 'AI'])

arr1 + arr2
np.char.add(arr1,arr2)



import numpy as np

def concatenate_strings(arr1, arr2):
  """Concatenates two NumPy arrays of strings element-wise.

  Args:
    arr1: The first NumPy array of strings.
    arr2: The second NumPy array of strings.

  Returns:
    A new NumPy array with the concatenated strings.
  """

  # Use np.char.add to concatenate corresponding elements of arr1 and arr2
  result = np.char.add(arr1, arr2)

  return result

# Example usage:
arr1 = np.array(['Hello', 'World'])
arr2 = np.array(['Open', 'AI'])
concatenated_arr = concatenate_strings(arr1, arr2)
print(concatenated_arr)

np.char.add()


#  60. Write a function that finds the length of the longest string in a NumPy array.
 arr = np.array(['apple', 'banana', 'grape', 'pineapple', 'mr.loekshs dkdkd'])
 arr = np.array(['apple', 'banana', 'grape', 'pineapple'])




def longest_str(arr):
  lenth = np.char.str_len(arr)
  return np.max(lenth)
longest_str(arr)


#  61. Create a dataset of 100 random integers between 1 and 1000. Compute the mean, median, variance, and 
# standard deviation of the dataset using NumPy's functions.
arr = np.random.rand(1,1000, 100)
len(arr)
arr
arr = np.random.random_integers(1,1001,100)
np.mean(arr)
np.median(arr)
np.var(arr)
np.std(arr)


#  62. Generate an array of 50 random numbers between 1 and 100. Find the 25th and 75th percentiles of the 
# dataset.

arr = np.random.randint(1,101,50)
len(arr)
np.percentile(arr, 50)
np.mean(arr)
np.median(arr)


# 63. Create two arrays representing two sets of variables. Compute the correlation coefficient between these 
# arrays using NumPy's `corrcoef` function.

arr1_price = np.array([100,200,340,480,136,774])
arr2_profit = np.array([30,50,34,59,120,230])
np.corrcoef(arr1_price,arr2_profit)[0,1]
type(arr1_price)


# 64. Create two matrices and perform matrix multiplication using NumPy's `dot` function.

mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[4,5],[6,8]])
mat1 * mat2

np.dot(mat1,mat2)



import numpy as np

# Create two matrices
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[7, 8], [9, 10], [11, 12]])

# Perform matrix multiplication using np.dot
result = np.dot(matrix1, matrix2)

print("Matrix 1:")
print(matrix1)
print("Matrix 2:")
print(matrix2)
print("Result:")
print(result)


# 65. Create an array of 50 integers between 10 and 1000. Calculate the 10th, 50th (median), and 90th 
# percentiles along with the first and third quartiles.

arr = np.random.randint(10,1001,50)
len(arr)
np.percentile(arr,10)
np.percentile(arr, 50)
np.percentile(arr, 90)



# 66. Create a NumPy array of integers and find the index of a specific element.

arr = np.random.randint(1,10,20)
index = np.where(arr ==4)[0][0]
arr

print('index of the first occurance of the element 4:', index)
arr[4]


# 67. Generate a random NumPy array and sort it in ascending order.

arr = np.random.randint(1,10,14)
arr
np.sort(arr)

# 68. Filter elements >20  in the given NumPy array.
arr = np.array([12, 25, 6, 42, 8, 30])
ar_filter = arr[arr >20]
ar_filter 


# 69. Filter elements which are divisible by 3 from a given NumPy array.
arr = np.array([1, 5, 8, 12, 15])
arr_div_3 = arr[arr%3==0]
arr_div_3 


# 70. Filter elements which are ≥ 20 and ≤ 40 from a given NumPy array.
arr = np.array([10, 20, 30, 40, 50])
arr_filter = arr[(arr >=20) & (arr <=40) ]
arr_filter


# 71. For the given NumPy array, check its byte order using the `dtype` attribute byteorder.
arr = np.array([1, 2, 3])
np.dtype(arr)

arr = np.array([3,4,6,22,455])


byyy= arr.dtype.byteorder
byyy


import numpy as np

arr = np.array([1, 2, 3])

byte_order = arr.dtype.byteorder

print("Byte order:", byte_order)


# 72. For the given NumPy array, perform byte swapping in place using `byteswap()`.
arr = np.array([1, 2, 3], dtype=np.int32)

byteswapp =arr.byteswap(inplace= True)

byteswapp


import numpy as np

arr = np.array([1, 2, 3], dtype=np.int32)

# In-place byte swapping
arr.byteswap(inplace=True)

print(arr)



# 73. For the given NumPy array, swap its byte order without modifying the original array using newbyteorder()`.
arr = np.array([1, 2, 3], dtype=np.int32)

new_byte_order = arr.byteswap()
print(new_byte_order)
 
 
 
#  74. For the given NumPy array and swap its byte order conditionally based on system endianness using 
#  newbyteorder().
arr = np.array([1, 2, 3], dtype=np.int32)

import numpy as np

def swap_byte_order_conditionally(arr):
  """Swaps the byte order of a NumPy array conditionally based on system endianness.

  Args:
    arr: The input NumPy array.

  Returns:
    A new NumPy array with the swapped byte order if necessary.
  """

  # Determine the system's native byte order
  native_byte_order = np.dtype(arr.dtype).byteorder
  
  # Swap byte order if the array's byte order is different from the native byte order
  if native_byte_order == '<':
    swapped_arr = arr.newbyteorder('>')
  elif native_byte_order == '>':
    swapped_arr = arr.newbyteorder('<')
  else:
    swapped_arr = arr

  return swapped_arr

# Example usage:
arr = np.array([1, 2, 3], dtype=np.int32)
swapped_arr = swap_byte_order_conditionally(arr)

print("Original array:", arr)
print("Swapped array:", swapped_arr)


# 75. For the given NumPy array, check if byte swapping is necessary for the current system using `dtype` 
# attribute `byteorder`.
arr = np.array([1, 2, 3], dtype=np.int32)

import numpy as np

def is_byte_swap_necessary(arr):
  """Checks if byte swapping is necessary for the given NumPy array based on system endianness.

  Args:
    arr: The input NumPy array.

  Returns:
    True if byte swapping is necessary, False otherwise.
  """

  # Determine the system's native byte order
  native_byte_order = np.dtype(arr.dtype).byteorder

  # Check if the array's byte order is different from the native byte order
  if arr.dtype.byteorder != native_byte_order:
    return True
  else:
    return False

# Example usage:
arr = np.array([1, 2, 3], dtype=np.int32)
is_necessary = is_byte_swap_necessary(arr)

if is_necessary:
  print("Byte swapping is necessary.")
else:
  print("Byte swapping is not necessary.")
  
  
#    76. Create a NumPy array `arr1` with values from 1 to 10. Create a copy of `arr1` named `copy_arr` and modify 
# an element in `copy_arr`. Check if modifying `copy_arr` affects `arr1`.

arr1 = np.random.randint(1,10,10)
copy_arr = np.copy(arr1)
copy_arr[0] = 44

copy_arr
arr1
arr1 = np.arange(1,11)



# 77. Create a 2D NumPy array `matrix` of shape (3, 3) with random integers. Extract a slice `view_slice` from 
# the matrix. Modify an element in `view_slice` and observe if it changes the original `matrix`.

matrix = np.random.randint(1,10,(3,3))
view_slice = matrix[1:,2:]
view_slice[0,0] = 999
view_slice
matrix



# 78. Create a NumPy array `array_a` of shape (4, 3) with sequential integers from 1 to 12. Extract a slice 
# `view_b` from `array_a` and broadcast the addition of 5 to view_b. Check if it alters the original `array_a`.

array_a = np.random.randint(1,13,(4,3))
view_b = array_a[0:,2:]
view_b += 5
view_b
array_a 
 
 

#  79. Create a NumPy array `orig_array` of shape (2, 4) with values from 1 to 8. Create a reshaped view 
# `reshaped_view` of shape (4, 2) from orig_array. Modify an element in `reshaped_view` and check if it 
# reflects changes in the original `orig_array`.
import numpy as np
import pandas as pd

orig_array = np.random.randint(1,9,(2,4))
reshaped_view = orig_array.reshape(4,2)
reshaped_view
reshaped_view[0] =[9]
orig_array



# 80. Create a NumPy array `data` of shape (3, 4) with random integers. Extract a copy `data_copy` of 
# elements greater than 5. Modify an element in `data_copy` and verify if it affects the original `data`.

data = np.random.randint(1,20,(3,4))
data
data_copy = data[data >5].copy()
data_copy[3] =20
data_copy
data


# 81. Create two matrices A and B of identical shape containing integers and perform addition and subtraction 
# operations between them.

A = np.matrix([1,2,3])
B = np.matrix([5,6,7])
A+B


#  82. Generate two matrices `C` (3x2) and `D` (2x4) and perform matrix multiplication.
 
C = np.random.randint(1,10,(2,4))
D = np.random.randint(1,10,(2,4))
C*D



#  83. Create a matrix `E` and find its transpose.
E = np.random.randint(1,12,(3,2))
d= np.transpose(E)
d
E


#  84. Generate a square matrix `F` and compute its determinant.

F = np.random.randint(1,12,(2,2))
np.linalg.det(F)

import numpy as np

# Create a square matrix F
F = np.random.randint(1, 10, (4, 4))

# Compute the determinant of F
determinant = np.linalg.det(F)

print("Matrix F:")
print(F)
print("Determinant of F:", determinant)


# 85. Create a square matrix `G` and find its inverse.

G = np.random.randint(1,19,(4,4))
G
h = np.linalg.inv(G)
h


