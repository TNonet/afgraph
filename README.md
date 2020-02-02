# afgraph
Arbitrary Function Graphs

Afgraph is a programmatic mathematical function interface written in Python and capable of running through numpy or tensorflow. It was developed with a focus on enabling activation functions of neural networks to be evolved using genetic algorthims. 

# Example:

The standard activation function is ReLU
```python
def relu(x):
  return max(0, x)
```

However, this function has could be "evolved" or changed randomly by chance to a set of neigbors:

```python
def relu_n1(x):
  return max(0, x-a)
  
def relu_n2(x):
  return max(b, x)
  
def relu_n3(x):
  return max(b, x**2)

...

def relu_ni(x):
  return max(b, sin(x))
```

Normally this would require a user to code these functions but with afgraph a base function can be written, and then it can be modified programmatically, recomplied in numpy or tensorflow and used in any setting with no coding.

```python
from afgraph.function.tree import FunctionTree
from afgraph.function.node import Max, Constant

def ReLU():
    test_graph = FunctionTree('ReLU')
    max_node = Max('max')
    const_node = Constant('a', 0)
    test_graph.insert_node(max_node, 'Output', 'x')
    test_graph.insert_node(const_node, 'max')
    return test_graph
```

ReLU can then be:
+ compiled into tensorflow function;
  ```python
  relu = ReLU()
  f = test.get_tensorflow()
  print(f(np.random.rand(3,3)))
  ```
  ```python
  array([[0.61421275, 0.77908075, 0.36841294],
         [0.15692288, 0.765053  , 0.324683  ],
         [0.46453375, 0.53577733, 0.69865566]], dtype=float32)
  ```
+ seen as a graph;
  ```python
  relu.get_graph()
  ```
  [Insert Image] (Must be added to git for image)
+ plotted;
  ```python
  relu.get_plot(np.arange(-10, 10, .001)
  ```
  [Insert Image] (Must be added to git for image)
+ made into numpy code;
  ```python
  print(test.get_numpy_str())
  ```
  ```python
  def f(x):
	  return np.max([0*np.ones_like(x), x], axis = 0)
  ```
+ made into latex code;
  ```python
  print(test.get_latex_str())
  ```
  ```python
  f(x) = max(0, x)
  ```
+ and much more

## Evolution.
**Note** This is still being worked on but afgraph offers several programmatic ways to change a function graph. All through a standard API interface that can be used to define how one graph changes to the next.
1. Insert Node between two feasible nodes
2. Insert Node as a child of a feasible node
3. Insert another function graph between two feasible nodes
4. Insert another function graph as a child of a feasible node
5. Replace node with another acceptable node
6. Connect two feasible nodes
7. Remove node and reconnect parent and child nodes:
8. And more other operations.

All of t
