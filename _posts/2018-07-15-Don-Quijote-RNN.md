---
layout: post_default
title:  "Generating Don Quijote sentences using a deep LSTM network"
date:   2018-07-18
categories: Python
excerpt_separator: ""
comments: true
---

Work in progress! Check out the repository in [Github](https://github.com/jsga/DonQuijote_RNN) for further details while I finish this post :)


_"El ingenioso hidalgo don Quijote de la Mancha"_, in English translated as _"The Ingenious Nobleman Sir Quixote of La Mancha"_ or simply _"Don Quixote"_, is one of the classical novels from the Spanish literature written by Cervantes in the 17th century. The style of the original book is rather difficult to read these days as it is written in a form of [old castilian](https://en.wikipedia.org/wiki/Old_Spanish_language), and certainly even more difficult to emulate these days. Can we use deep learning to solve this problem? The answer is YES!


The data that is feed into the model is the book of _El Quijote_ in plain text. Luckily it is freely available in [The Project Gutenberg](http://www.gutenberg.org/cache/epub/2000/pg2000.txt). I used chapters 1 to 10, with roughly 110k characters and 20k words.

# The model: a recurrent neural network with 3 layers of LSTM

The model takes as input a sequence of characters, say 100 characters. The goal of the model is to predict the next character. The training data (i.e., the next) is sliced so that, at each training sample, X is a column vector of 100 dimensions and Y is a single number.

The basic block of the model is a Long-Short Term Memory (LSTM)[https://en.wikipedia.org/wiki/Long_short-term_memory] block. LSTM-based models have proved to work really well in practice thanks to their forgetting and updating capabilities.

{% include image.html url= "/assets/LSTM.png" description="LSTM block definition. Credits to deeplearning.ai" width="500px"%}

For this problem we stack 3 LSTM layers with a dropout layer in between. A dropout layer basically randomly "switches off" some neurons at each pass, making the model less prone to over-fitting. In Keras, the model is defined as follows:

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

The rest of the code is publicly available in [this repository](https://github.com/jsga/DonQuijote_RNN).


## Generating sequences of (possibly new) words

```
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5
```

An example of the output:

> con todo eso, se le dejaron de ser su romance v me dejase, porque no le dejare y facilidad de su modo que de la lanza en la caballeriza, por el mesmo camino, y la donde se le habia de haber de los que el campo, porque el estaba la cabeza que le parece a le puerto y de contento, son de la primera entre algunas cosas de la venta, con tanta furia de su primer algunos que a los caballeros andantes a su lanza, y, aunque el no puede le dios te parecian y a tu parte, se dios ser puede los viera en la caballeria en la caballeria en altas partes de la mancha, 

It is surprisingly good! Some thoughts:

* The style of the book is definitely really similar to Cervantes style. Some sentences are really funny.
* The syntax, however, is not correct: _(...) se le dejaron de ser su romance (...)_ is wrong even though the words do exist
* Some of the generated words do not exist, even though it is not very often.


## Generating sequences of existing words

A minor modification to the generating algorithm is done such that the newly generated words must exist in the book. If not, that word is rejected. The process is repeated until reasonable words are generated.


```python
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5 -o True
```

> al cual le parecieron don quijote de la mancha, en cuando le daba a le senor tio en el corral, y tio que andaba muy acerto los dos viejos, y, al caso de van manera con el de tu escudero. don quijote y mas venta a su asno, con toda su amo pasa dios de la caballeria y de al que habia leido, no habia de ser tu escudero: la suelo del camino de la venta, de que san caballo de los que le habia dejado; a este libro es este es el mismo coche, como te ve don mucho deseos de los que el caballero le hallaba; y al corral con la cabeza que aquel sabio en la gente de la lanza y tan las demas y camas de tu escudero, 

The syntax of all the words is correct now


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
The model has been trained on a 61 GB GPU for roughly 5 hours. The [weights](https://github.com/jsga/DonQuijote_RNN/blob/master/weights-improvement-3L-512-23-1.2375.hdf5) of the model are updated to the repository.


# Further ideas on how to improve the model

- Train for longer epochs with a bigger training set. This is a safe way of getting better results.
- Implement a beam search algorithm character-wise.
- Model whole words instead of characters. This will certainly generate more sensible sentences.
- Generate full sentences with a bi-directional RNN.
- Apply a pre-trained embedding matrix (word2vec for example), even though with an old-style of writing as Don Quijote it will most likely not work very well



# References and further readings

* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [How to Develop a Word-Level Neural Language Model and Use it to Generate Text](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
* [keras/examples/lstm_text_generation.py](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)