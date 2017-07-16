---
layout: post_default
title:  "Introduction to inverse optimization"
date:   2017-08-30
categories: tutorial
excerpt_separator: ""
comments: true
---



This post will cover the following points:

+ Background on linear programming
+ What is inverse optimization?
+ Applications of inverse optimization
+ And example with GAMS & R code: bidding in electricity markets
+ Further reading


# Background on linear programming

Inverse optimization is a framework for estimating parameters of a mathematical model. This might seem very abstract for the folks that are not familiar with mathematical models, but, in fact, inverse optimization is oriented to solving real-life situations. In the context of this blog post, a mathematical model describe certain area of reality by using simple equations. This simplified view of reality is then used for several purposes: understanding what happened in the past, forecasting the future, and helping on the decision-making.


Let us step back and recall the basic concepts of [Linear programming](https://en.wikipedia.org/wiki/Linear_programming), which are the ones we focus on in this tutorial. In _traditional_ optimization, the main goal is to find the optimal value of the decision variable, often denoted by _x_. _x_ can answer questions like: "how should I organize the transportation of my goods so that I save money and fuel?", "how do I select my portfolio of investment optimally?", or even "how to do I go from A to B following the fastest route?".


Linear problems are composed of two pieces. The first piece is the **objective function** that we are interested in maximizing. For example, benefits, expressed as an equation dependent on _x_. The second piece is the **constraints**, namely, a set of equations that define our reality. Often linear problems are written as follows: 



$$
\underset{\boldsymbol x}{\text{Maximize}} \   \color{orange} \boldsymbol c^{\rm{T}} \color{black} \boldsymbol x 
$$

$$
\text{subject to} \    \color{orange} \boldsymbol A \color{black} \boldsymbol x \leq \color{orange} \boldsymbol b 
$$

where $$\color{orange}\boldsymbol A,\boldsymbol b$$ and $$\color{orange} \boldsymbol c$$ are our known parameters and $$x$$ is our set of unknown decision variables.

There are many methods to find the optimal value, being the [simplex algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm) and the [interior point method](https://en.wikipedia.org/wiki/Interior_point_method) the two most famous ones. Further reading on linear programming is found in [this book](http://web.mit.edu/15.053/www/AMP-Chapter-01.pdf), or in page 23 of [this PhD thesis](http://web.mit.edu/15.053/www/AMP-Chapter-01.pdf).


# What is inverse optimization?

In an inverse-optimization framework, the solution to the problem (called often $$x^*$$) is **known**. What is **unknown** is the model parameters. If you think this is too strange to even happen in reality, in the section I give a few examples of real cases where this happens.

Roughly speaking, an inverse optimization problem looks very similar as before:

$$
\underset{\color{orange} \boldsymbol x}{\text{Maximize}} \   \boldsymbol c^{\rm{T}}  \color{orange} \boldsymbol x 
$$

$$
\text{subject to} \    \boldsymbol A \color{orange} \boldsymbol x \color{black} \leq \boldsymbol b 
$$

with the significant difference that now **A**,**b** and **c** are decision variables and $$\color{orange} x^{*}$$ is a known parameter.


Before solving an inverse optimization problem it needs to be reformulated, since the notation above is not really mathematically correct and it cannot not be understood by any optimization software. Here, we make use of a basic property of linear problems and re-formulate it by its [dual problem](http://web.mit.edu/15.053/www/AMP-Chapter-04.pdf):



$$
\underset{\boldsymbol c, \epsilon, \boldsymbol \lambda }{\text{Minimize}} \     \epsilon
$$

$$
 \text{subject to} \    \boldsymbol c^{\rm{T}} \boldsymbol x' + \epsilon = \boldsymbol b^{\rm{T}} \boldsymbol \lambda 
$$

$$
\boldsymbol A^{\rm{T}} \boldsymbol \lambda = \boldsymbol c 
$$

$$
\boldsymbol \lambda\ \geq \boldsymbol 0. 
$$


The second equation corresponds to the relaxed strong duality conditions from the original linear problem presented above, and the third and fourth equations are its dual feasibility constraints. This formulation is unfortunately non-linear due to the term $$\boldsymbol b^{\rm{T}} \boldsymbol \lambda $$. Computationally attractive methods to solve this type of problems is given in [this paper](https://www.researchgate.net/publication/317645589_Inverse_Optimization_and_Forecasting_Techniques_Applied_to_Decision-making_in_Electricity_Markets?channel=doi&linkId=59464faaaca2722db4a5dd2a&showFulltext=true). We will not go into details here.

<!-- Note that, because we know $$\boldsymbol A$$ and $$\boldsymbol b$$, the primal constraints of the original problem are omitted. -->




# An example: forecasting electricity loads

[Coming soon](https://www.researchgate.net/publication/317645589_Inverse_Optimization_and_Forecasting_Techniques_Applied_to_Decision-making_in_Electricity_Markets?channel=doi&linkId=59464faaaca2722db4a5dd2a&showFulltext=true)



# Other applications of inverse optimization

### Geophysical studies

### Prostate cancer treatment

### Revealing competitor's prices

