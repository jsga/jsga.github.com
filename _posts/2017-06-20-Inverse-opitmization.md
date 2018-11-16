---
layout: post_default
title:  "Introduction to inverse optimization"
date:   2017-07-16
categories: tutorial
excerpt_separator: ""
comments: true
---


This post will cover the following points:

+ Background on linear programming
+ What is inverse optimization?
+ Applications of inverse optimization with some code


# Background on linear programming

Inverse optimization is a framework for estimating parameters of a mathematical optimization model. This might seem very abstract for the folks that are not familiar with mathematical models, but, in fact, inverse optimization is oriented to solving real-life situations. In the context of this blog post, a mathematical model describes certain area of reality by using simple equations. This simplified view of reality is then used for several purposes: understanding what happened in the past, forecasting the future, and helping on the decision-making.


Let us step back and recall the basic concepts of [Linear programming](https://en.wikipedia.org/wiki/Linear_programming), which are the ones we focus on in this tutorial. In _traditional_ optimization, the main goal is to find the optimal value of the decision variable, often denoted by "_x_". _x_ can answer questions like: "how should I organize the transportation of my goods so that I save money and fuel?", "how do I select my portfolio of investment optimally?", or even "how to do I go from A to B following the fastest route?".


Linear problems are composed of two pieces. The first piece is the **objective function** that we are interested in maximizing. For example, benefits, expressed as an equation dependent on _x_. The second piece is the **constraints**, namely, a set of equations that define our reality. Often linear problems are written as follows:



$$
\underset{\boldsymbol x}{\text{Maximize}} \   \color{orange} \boldsymbol c^{\rm{T}} \color{black} \boldsymbol x
$$

$$
\text{subject to} \    \color{orange} \boldsymbol A \color{black} \boldsymbol x \leq \color{orange} \boldsymbol b
$$

where $$\color{orange}\boldsymbol A,\boldsymbol b$$ and $$\color{orange} \boldsymbol c$$ are our known parameters and $$x$$ is our set of unknown decision variables.

There are many methods to find the optimal value, being the [simplex algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm) and the [interior point method](https://en.wikipedia.org/wiki/Interior_point_method) the two most famous ones. Further reading on linear programming is found in [this book](http://web.mit.edu/15.053/www/AMP-Chapter-01.pdf), or in page 23 of [this PhD thesis](https://www.researchgate.net/publication/317645589_Inverse_Optimization_and_Forecasting_Techniques_Applied_to_Decision-making_in_Electricity_Markets).


# What is inverse optimization?

In an inverse-optimization framework, the solution to the problem (so-called $$x^*$$) is **known**. What is **unknown** is the model parameters. If you think this is too strange to even happen in reality, in the section I give a few examples of real cases where this applies.

Roughly speaking, an inverse optimization problem looks very similar as before:

$$
\underset{\boldsymbol A, \boldsymbol b, \boldsymbol c  }{\text{Maximize}} \  \boldsymbol c^{\rm{T}} \color{orange} \boldsymbol x^*
$$

$$
\text{subject to} \    \boldsymbol A \color{orange} \boldsymbol x^* \color{black} \leq \boldsymbol b
$$

with the significant difference that now **A**,**b** and **c** are decision variables and $$\color{orange} x^{*}$$ is a known parameter.


Before solving an inverse optimization problem it needs to be reformulated, since the notation above is not really mathematically correct and it cannot not be understood by any optimization software. Here, we make use of a basic property of linear problems and re-formulate it by its [dual problem](http://web.mit.edu/15.053/www/AMP-Chapter-04.pdf):



$$
\underset{\boldsymbol A, \boldsymbol b, \boldsymbol c, \epsilon, \boldsymbol \lambda }{\text{Minimize}} \     \epsilon
$$

$$
 \text{subject to} \    \boldsymbol c^{\rm{T}} \boldsymbol x^* + \epsilon = \boldsymbol b^{\rm{T}} \boldsymbol \lambda
$$

$$
\boldsymbol A^{\rm{T}} \boldsymbol \lambda = \boldsymbol c
$$

$$
\boldsymbol \lambda\ \geq \boldsymbol 0.
$$


The second equation corresponds to the relaxed strong duality conditions from the original linear problem presented above, and the third and fourth equations are its dual feasibility constraints. This formulation is unfortunately non-linear due to the terms $$\boldsymbol b^{\rm{T}} \boldsymbol \lambda $$ and $$\boldsymbol A^{\rm{T}} \boldsymbol \lambda$$. Computationally attractive methods to solve this type of problems is given in [this paper](https://www.researchgate.net/publication/317645589_Inverse_Optimization_and_Forecasting_Techniques_Applied_to_Decision-making_in_Electricity_Markets?channel=doi&linkId=59464faaaca2722db4a5dd2a&showFulltext=true). We will not go into details here but feel free to contact me of leave a comment.




# Applications

### Forecasting electricity loads

One of the topics of my PhD thesis is to forecast the electricity consumptions of a pool of price-responsive houses. The work, published in [this paper](https://www.researchgate.net/publication/305638628_Short-term_Forecasting_of_Price-responsive_Loads_Using_Inverse_Optimization), explains the theoretical approach and some of the results. The code is found [in this repository](https://github.com/jsga/Inverse_optim_forecast_and_simulation).

### Market bidding, applied to electricity trading

A market bid with consists of an utility function, ramp limits and energy bounds for every traded period. Using an inverse-optimization approach one can treat the historical consumption/production as the "known" $$x^{*}$$, and the market parameters as the quantities to be estimated by the inverse model. Regressors can be used to make such estimations adaptive: the so-called _Dynamic Inverse Optimization_ as in [page 30 of my PhD thesis](https://www.researchgate.net/publication/317645589_Inverse_Optimization_and_Forecasting_Techniques_Applied_to_Decision-making_in_Electricity_Markets). Further details are also explained in [this paper](https://www.researchgate.net/publication/295832540_A_Data-Driven_Bidding_Model_for_a_Cluster_of_Price-Responsive_Consumers_of_Electricity) or in this presentation given at INFORMS in Philadelphia:

<center><object data="{{ site.url }}/assets/INFORMS_Philadelphia_Inverse.pdf#view=fitBH" type="application/pdf" width="80%" height="400px"> </object></center>


### Other applications

* [Prostate cancer treatment](http://pubsonline.informs.org/doi/abs/10.1287/opre.2014.1267?journalCode=opre)
* [Revealing competitor's prices](http://ieeexplore.ieee.org/abstract/document/6423235/?reload=true)
* [Shortest path, assignment, minimum cut...](https://www.researchgate.net/publication/265461398_Inverse_Optimization)
