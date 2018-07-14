---
layout: post_default
title:  "Generating Don Quijote sentences using a deep LSTM network"
date:   2018-07-18
categories: Python
excerpt_separator: ""
comments: true
---

# Work in progress!

Check out the repository in [Github](https://github.com/jsga/DonQuijote_RNN) for further details while I finish this post :)


# The model

The basic block of the model is an Long-Short Term Memory (LSTM)[https://en.wikipedia.org/wiki/Long_short-term_memory] block. LSTM-based models have proved to work really well in practice thanks to their forgetting and updating capabilities.

{% include image.html url= "/assets/LSTM.jpg " description="LSTM block definition. Credits to deeplearning.ai" width="500px"%}

For this problem we stack 3 LSTM layers with a dropout layer in between. A dropout layer basically randomly "switches off" some neurons at each pass, making the model less prone to overfitting. In Keras, the model is defined as follows:

```python
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

```

# Training on FloydHub

Training the model on [FloydHub](https://www.floydhub.com/) is really easy and straightforward. Here it is a quick quide on how you can do the same with only a few lines of code


1. First you need to create an account
2. Install their command-line tool and login
	```
	pip3 install -U floyd-cli
	floyd login
	```
3. Create a [project](https://www.floydhub.com/projects)
4. Go to the folder where the code is ans initialize the project:
	```
	cd DonQuijote_RNN
	floyd init DonQuijote_RNN
	```
5. Run a script that trains the model and saves the weights on a folder
	```
	floyd run --gpu --env tensorflow-1.3 "python3 DonQuijote.py"
	```


# Further ideas

- Increase number of epochs and/or batch size
- word sequence prediction
- Word2Vec



# References and further readings

* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [How to Develop a Word-Level Neural Language Model and Use it to Generate Text](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
* [keras/examples/lstm_text_generation.py](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)