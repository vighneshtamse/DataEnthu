---
title: "5 Basic Pytorch Functions"
date: 2020-06-01T12:39:50+05:30
draft: true
author: "Vighnesh Tamse"
---

# Basic Pytorch Tensor Functions

Pytorch is a python based scientific computing package which is replacement for Numpy to use the power of GPUs and also provides maximum flexibility and speed. We will now look at 5 of the many interesting pytorch functions in this section.

- `torch.from_numpy(ndarray)`
- `torch.argmax(input)`
- `torch.unsqueeze(input, dim)`
- `torch.mm(input, mat2, out=None)`
- `torch.pow(input, exponent, out=None)`


```python
# Import torch and other required modules
import torch
import jovian
import numpy as np
```


    <IPython.core.display.Javascript object>



## Function 1 - torch.from_numpy(ndarray)

This function is used to create tensors from numpy array. It expects the input as a numpy array (numpy.ndarray) and the output type is a tensor. The returned tensor and ndarray share the same memory.


```python
# Example 1 - working
a = np.array([1,2,3,4,5])
torch.from_numpy(a)
```


    tensor([1, 2, 3, 4, 5], dtype=torch.int32)

Here we created a 1D tensor from a 1D numpy arrray. 


```python
# Example 2 - working
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
torch.from_numpy(b)
```


    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=torch.int32)

Here we created a 3D tensor from a 3D numpy array.


```python
# Example 3 - breaking (to illustrate when it breaks)
c = np.array([1,2.,'c'])
print(c)
torch.from_numpy(c)
```

    ['1' '2.0' 'c']
    
    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-4-46829fbfa4c2> in <module>
          2 c = np.array([1,2.,'c'])
          3 print(c)
    ----> 4 torch.from_numpy(c)


    TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, int64, int32, int16, int8, uint8, and bool.


When we included a string in the numpy array and tried to convert it to torch tensors, we got an error. So, when converting the numpy array to tensors we need to nake sure that the data type of elements in the numpy array is integer, float or boolean.

**torch.from_numpy()** can be used when one wants to convert a numpy array to a tensor which can further be used to perform various other functions on the thus created tensor.



## Function 2 - torch.argmax(input)

torch.argmax() is a reduction operation function which returns the indices of the maximum value of all elements in the input tensor.


```python
# Example 1 - working
a = torch.randn(4,4)
print(a)
torch.argmax(a)
```

    tensor([[ 0.9441, -1.1231, -0.4217,  0.7111],
            [-0.3351, -0.2536,  0.2316, -2.9140],
            [-1.4188, -0.7284,  0.8831, -1.3805],
            [ 0.1804,  0.2054, -0.0078,  1.1865]])
    
    tensor(15)

Here first we generated a 4x4 matrix with random numbers and then applied the torch.argmax() function on it which returns the index of the maximum number out of all the elements in the tensor. As we can from the above example that the maximum value, out of all the values present, is at the 12th position i.e. 11th index (as indexing starts from 0) in the tensor b, hence we got the output as tensor(11).


```python
# Example 2 - working
b = torch.randn(3,3)
print(b)
print("Column wise maximum: ",torch.argmax(b,dim=0))
print("Row wise maximum: ",torch.argmax(b,dim=1))
```

    tensor([[ 0.0339, -0.6427, -0.1145],
            [-0.3618, -0.2167,  1.4323],
            [ 0.2412, -0.2032,  1.0173]])
    Column wise maximum:  tensor([2, 2, 1])
    Row wise maximum:  tensor([0, 2, 2])


Here we use a parameter called **dim** (dimension) in case we want the maximum number from rows and columns. dim=0 means columns and dim=1 means rows.


```python
# Example 3 - Inconsistent results between numpy.argmax() and torch.argmax() when there are duplicate values.
a = np.array([5, 2, 6, 11, 8, 11])
print(np.argmax(a))
b = torch.from_numpy(a)
#print(b)
print(torch.argmax(b))
```

    3
    tensor(5)


From the above example we can see that the argmax function of numpy and torch returns different index values of maximum numbers when there are duplicate values. The argmax function of numpy returns the first occurrence of the maximum value and the argmax function of torch returns the last occurrence of the maximum value.

**torch.argmax()** function is useful when you want to know the index of the maximum value of all the elements or index of the maximum value row wise or column wise. But when there are duplicate values remember that it returns the index of the last occurrence of the maximum value unlike numpy which returns the index of the first occurrence.



## Function 3 - torch.unsqueeze(input, dim)

This function returns a new tensor with a dimension of size one inserted at the specified position.
A dim value within the range [-input.dim() - 1, input.dim() + 1] can be used.


```python
# Example 1 - working
a = torch.tensor([2,8,6,7])
print(a)
torch.unsqueeze(a,1)
```

    tensor([2, 8, 6, 7])
    
    tensor([[2],
            [8],
            [6],
            [7]])

Using torch.unsqueeze() in the above example, we added a dimension at position 1.


```python
# Example 2 - working
a = torch.randn(3,3)
print(a)
torch.unsqueeze(a,2)
```

```
tensor([[ 1.1121, -0.9097,  0.3571],
        [ 0.7203,  1.3986, -0.0914],
        [ 0.0932, -0.5584,  2.3528]])
```

```
tensor([[[ 1.1121],
         [-0.9097],
         [ 0.3571]],

        [[ 0.7203],
         [ 1.3986],
         [-0.0914]],

        [[ 0.0932],
         [-0.5584],
         [ 2.3528]]])
```

In the above example, we added a dimension at position 2


```python
# Example 3 - breaking (to illustrate when it breaks)
a = torch.randn(3,3)
print(a)
torch.unsqueeze(a,3)
```

    tensor([[ 1.2530,  0.2198,  1.0602],
            [ 0.6132, -0.5068, -0.5944],
            [-0.4629,  0.7406,  0.9527]])
    
    ---------------------------------------------------------------------------
    
    IndexError                                Traceback (most recent call last)
    
    <ipython-input-10-b839220d5c74> in <module>
          2 a = torch.randn(3,3)
          3 print(a)
    ----> 4 torch.unsqueeze(a,3)


    IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)


In the above example, the dimension should be in the range between -3 and 2 according to the formula [-input.dim() - 1, input.dim() + 1] but we are providing a dimension of 3 which is out of range hence the error.

**torch.unsqueeze()** function should be used when the dimensionality of the input is unknown but we would still like to insert a singleton dimension. This function lets us insert the dimension without explicitly being aware of the other dimensions when writing the code.



## Function 4 - torch.mm(input, mat2, out=None)

This function is used to perform matrix multiplication of the matrices - input and mat2.
If input is a (n x m) tensor, mat2 is a (m x p) tensor, out will be a (n x p) tensor.


```python
# Example 1 - working
a = torch.tensor([[1,2,3],[6,5,4],[7,8,9]])
b = torch.tensor([[9,8,7],[4,5,6],[3,2,1]])
torch.mm(a,b)
```


    tensor([[ 26,  24,  22],
            [ 86,  81,  76],
            [122, 114, 106]])

In the above example, we created two 3x3 matrices and performed matrix multiplication on the same resulting in 3x3 matrix.


```python
# Example 2 - working
a = torch.randn(5,3)
b = torch.randn(3,5)
torch.mm(a,b)
```


    tensor([[-0.8871,  1.6875,  1.0668,  0.4330,  1.0075],
            [-2.5780,  0.9463,  0.0709, -1.9850,  2.8363],
            [ 1.7439, -2.2066, -2.2754, -1.2523, -1.8266],
            [ 1.1687, -2.0800, -1.2103, -0.3440, -1.3347],
            [-0.2073, -2.9300, -1.1681, -1.1860,  0.0179]])

In the above example, our matrix 1 is 5x3 and matrix 2 is 3x5. So the resulting matrix due to matrix multiplication will be 5x5.


```python
# Example 3 - breaking (to illustrate when it breaks)
a = torch.randn(5,3)
b = torch.randn(5,3)
torch.mm(a,b)
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-13-0474d6ccc1b5> in <module>
          2 a = torch.randn(5,3)
          3 b = torch.randn(5,3)
    ----> 4 torch.mm(a,b)


    RuntimeError: size mismatch, m1: [5 x 3], m2: [5 x 3] at C:\w\1\s\tmp_conda_3.7_100118\conda\conda-bld\pytorch_1579082551706\work\aten\src\TH/generic/THTensorMath.cpp:136


For matrix multiplication, the columns of first matrix and the rows of second matrix should be same (n x m and m x p). But in this example this is not the case, hence runtime error.

**torch.mm()** should be used when we need to multiply two matrices which can be broadcasted i.e. having the same dimensions.



## Function 5 - torch.pow(input, exponent, out=None)

This function takes the power of each element in input with exponent and returns a tensor with the result.


```python
# Example 1 - working
a = torch.tensor([2,4,6,8])
torch.pow(a,2)
```


    tensor([ 4, 16, 36, 64])

In the above example we apply a power of 2, which is a single scalar value, to each element of the tensor.


```python
# Example 2 - working
b = torch.randn(5)
print(b)
torch.pow(b,torch.tensor([2,3,4,5,6]))
```

    tensor([ 1.0490,  1.8707,  0.4406, -0.9861, -1.8026])
    
    tensor([ 1.1004,  6.5470,  0.0377, -0.9325, 34.3131])

In the above example, we created 5 random numbers using the torch.randn() function and applied an exponential value to each of these values using a tensor of [2,3,4,5,6]. The number of elements in the exponent should be equal to the number of elements in the input tensor.


```python
# Example 3 - breaking (to illustrate when it breaks)
b = torch.randn(5)
print(b)
torch.pow(b,torch.tensor([1,2,3,4,5,6]))
```

    tensor([ 1.6209, -0.1225,  1.2093, -1.5934,  0.3392])
    
    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-16-6c01d5166f67> in <module>
          2 b = torch.randn(5)
          3 print(b)
    ----> 4 torch.pow(b,torch.tensor([1,2,3,4,5,6]))


    RuntimeError: The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 0


Here as we see that the number of elements in the exponent tensor is greater that the input tensor and hence the runtime error. The shapes of input and exponent must be same i.e. broadcastable.

**torch.pow()** can be used when we need to calculate the power of each element in the input tensor either by using a single exponent value or multiple values. But keep in mind that when using multiple exponent values, the shape of the expoenent tensor should be same as that of input tensor.



## Conclusion

In this notebook we learnt some of the basic yet interesting pytorch functions for faster prototyping and development of a Deep Learning Project. These functions are just the icing on the cake. There are numerous such functions which you can find in the official documentation of pytorch - https://pytorch.org/docs/stable/tensors.html



## Reference Links

Provide links to your references and other interesting articles about tensors

* Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html