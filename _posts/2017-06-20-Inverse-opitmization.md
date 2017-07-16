---
layout: post_default
title:  "Tutorial: inverse optimization"
date:   0001-01-01
categories: tutorial
excerpt_separator: ""
comments: true
---


This post will cover the following points:

+ What is inverse optimization?
+ Applications of inverse optimization
+ And example with GAMS & R code: bidding in electricity markets
+ Further reading


# What is inverse optimization?

Inverse optimization is a framework for estimating parameters of a mathematical model. This might seem very abstract for the folks that are not familiar with mathematical models, but, in fact, inverse optimization is oriented to solving real-life situations. In the context of this blog post, a mathematical model describe certain area of reality by using simple equations. This simplified view of reality is then used for several purposes: understanding what happened in the past, forecasting the future, and helping on the decision-making.


Let us step back and recall the basic concepts of [Linear programming](https://en.wikipedia.org/wiki/Linear_programming). In _traditional_ or _straight_ optimization, the main goal is to find the optimal value of the decision variable, often denoted by _x_. There are many methods to find the optimal value, being the [simplex algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm) and the [interior point method](https://en.wikipedia.org/wiki/Interior_point_method) the two most famous ones.

$$
\begin{subequations} \label{eq:MTH_inverse1}
 \ \  & \underset{\boldsymbol x}{\text{Maximize}} \  & \boldsymbol c^{\rm{T}}  \boldsymbol x \label{eq:MTH_inv-obj}
\\
 & \text{subject to} \    & \boldsymbol A \boldsymbol x \leq \boldsymbol b \label{eq:MTH_inv-cons}
\end{subequations}
$$




# Applications of inverse optimization



# An example: bidding in electricity markets

[Coming soon](https://www.researchgate.net/publication/317645589_Inverse_Optimization_and_Forecasting_Techniques_Applied_to_Decision-making_in_Electricity_Markets?channel=doi&linkId=59464faaaca2722db4a5dd2a&showFulltext=true)

