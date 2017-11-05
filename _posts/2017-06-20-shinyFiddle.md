---
layout: post_default
title:  "shinyFiddle R package: how to modify colors in tabs"
date:   2017-11-05
categories: R, shiny
excerpt_separator: ""
comments: true
---

# Introducing shinyFiddle R package
[R-shiny](http://shiny.rstudio.com/gallery/) is a great tool to create dashboard and easily share visual results with non-technical users. The possibilities for creating amazing interactive graphs are endless.

Recently I had to *fiddle* with some out-of-the-box functionalities:

	* add placeholder to numericInput elements
	* change the border color of a numericInput
	* change the color of the tab title
	* change background color of a tab

In an attempt to create some value for the community I created and [published](https://github.com/jsga/shinyFiddle) the _shinyFiddle_ R package. Install it and try it out by running the following R code:

```R
require(devtools)
install_github('jsga/shinyFiddle')
shinyFiddle_example()
```

***

<!-- <iframe src="https://jsaezgallego.shinyapps.io/shinyfiddle/" style="border: none; width: 440px; height: 500px"></iframe> -->

 