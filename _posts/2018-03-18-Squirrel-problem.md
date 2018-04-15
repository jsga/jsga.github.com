---
layout: post_default
title:  "Shortest path across Spain: the squirrel problem"
date:   2018-03-18
categories: R
excerpt_separator: ""
comments: true
---

**(Jump straight to the [solution](http://jsaezgallego.com/GlobCover_maps_squirrel/).)**

This post answers the following simple question. _Can a squirrel cross from north to south Spain without touching the ground?_

The answer is: **obviously not**. But, another question arises, and this one is not so easy to answer:_If a squirrel had to go from the north of Spain to the south, touching the ground as little as possible: which way would it follow?_


The answer is not trivial.

# The history

The legend says that Spain was once so thickly-forested that a squirrel could cross the peninsula hopping from tree to tree. Even though we all learned this at school as a ground truth, it seems that, in fact, this was just a [legend](https://copepodo.wordpress.com/2009/05/11/la-espana-de-la-ardilla-y-la-espana-del-conejo/). Spain was in fact "land of rabbits", as the Romans used to call it. 

Much [has been](https://www.facebook.com/Una-ardilla-podr%C3%ADa-cruzar-Espa%C3%B1a-saltando-de-gilipollas-en-gilipollas-185947181436539/) said about that squirrel, but none have proved it analytically. Here you have found the answer!

# The procedure

I used the [GlovCover](http://due.esrin.esa.int/page_globcover.php) maps from the ESA to get information of the land use. The resolution is around 200m, accurate enough for this purpose. The map spans the whole world but here we are just interested in Spain. So the first step is to load the image raster and crop it.


{% include image.html url= "/assets/Screenshot_squirrel.png" description="Screenshot of the solution" width="500px"%}



The cropped raster image needs to be converted to a directed graph:

  * Each pixel is a node
  * Each node is connected to its adjacent pixels (edge)
  * The weights of each edge correspond to the "roughness" level of the destination pixel. Less roughness means more trees.
  
Having such a graph, the last step is to calculate the shortest path between two nodes - and there we have the path the smart squirrel would follow!

# The coding

The code is purely R and uses several packages: raster, rgdal, Matrix, igraph... Creating the network matrix runs really slow, so some optimization in terms of code to be done.

# The solution

The solution is presented [HERE](http://jsaezgallego.com/GlobCover_maps_squirrel/) as a interactive dashboard. Unfortunately, the poor squirrel needs to touch the ground at several places as it gets south.

# Download path coordinates

Download them [here](https://github.com/jsga/GlobCover_maps_squirrel/blob/master/Files/path_coordinates_solution.csv)
