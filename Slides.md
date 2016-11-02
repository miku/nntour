Hello World
========

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

----

![](images/Escape_from_the_local_minimum.jpg)

----

| A | B | A AND B |   |   |
|---|---|---------|---|---|
|  1 | 2  | 3        | 5  | 6  |
|  1 | 2  | 3        | 4  | 5  |
|  7 |7   | 7        |  6 |  5 |

----

Representing Boolean Algebra as Classifiers
=====
| $x_1$ | $x_2$ | AND | OR | XOR | 
|---|---|-----|----|-----|
| 0 | 0 |  0  | 0  | 0   | 
| 0 | 1 |  0  | 1  | 1   | 
| 1 | 0 |  0  | 1  | 1   | 
| 1 | 1 |  1  | 1  | 0   |

---
AND is linearly seperable 
===
<img src="images/AND.jpg" width="600"> 

---
One possible AND perceptron
===
<img src="images/AND_perceptron.jpg" width="600"> 

---
OR(/NOR) is linearly seperable
===
<img src="images/OR.jpg" width="600"> 

---
One possible AND perceptron
===
<img src="images/OR_perceptron.jpg" width="600"> 

---
XOR is not linearly separable
===
<img src="images/XOR.jpg" width="600"> 

---
XOR can be represented by a combination of two mappings
===
|point| $x_1$ | $x_2$ | $($AND | $\lor$ | NOR $)$ | XOR|
|----|---|---|-----|----|-----|-|
| a  | 0 | 0 		|  0  | 0  | 1   | 0 | 
| b  | 0 | 1 		|  0  | 1  | 0   | 1 |
| c  | 1 | 0 		|  0  | 1  | 0   | 1 |
| d  | 1 | 1 		|  1  | 0  | 0   | 0 |

---
The extra mapping can be visualized
===
<img src="images/XOR_map.jpg" width="600"> 

---
One possible XOR Net 
===
The ones are fixed input (bias) units
<img src="images/XOR_perceptron_1.jpg" width="600"> 

---
One alternative XOR Net
===
The number within the perceptron represents the inherent bias unit/or a translational shift when the unit jumps. 
<img src="images/XOR_perceptron_2.jpg" width="600"> 

