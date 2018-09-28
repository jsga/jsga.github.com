---
layout: post_default
title:  "Learning insights from neural networks: a summary with code"
date:   2018-09-29
categories: Python
excerpt_separator: ""
comments: true
---


The goal of this notebook is to compile and share some specialized techniques to extract real-world insights from black-box models. It gives a concise summary with some code so that hopefully you will get the main ideas of the course. Also, so that I don't forget about them!

I have been strongly inspired by the challenge named ["Machine learning of insights"](https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values) from Kaggle.

The presented techniques have several characteristics in common:

1. They are used to gain insights from machine learning models. They are not used (generally speaking) for model building
2. They are model independent, hence can be used with black-box models.
3. The model needs to be used for predicting numerous times, hence these techniques are mostly useful when the computational time of running the model is relatively low. We do not need to re-train the model more than once, so they are especially useful in cases when training the model is time-consuming.
4. These techniques are meant to be run on tabulated data (i.e., regression or classification). In theory, they can accept pixel data as inputs even though the interpretation will be less obvious.
5. The conclusions are clear and concise. Beautiful plots can be easily generated.
6. There exists libraries in Python to achieve all of this without coding much

In the following chapters I will briefly introduce the dataset and cover the following topics:

1. Feature importance
2. Partial dependence plots
3. SHAP values



# The data: Air quality in Madrid

The data has been collected from [Kaggle datasets](https://www.kaggle.com/decide-soluciones/air-quality-madrid). It consists of a list of measurements of air quality in the city of Madrid. We will explore the following components:

- O_3: ozone level measured in μg/m³. High levels can produce asthma, bronchytis or other chronic pulmonary diseases in sensitive groups or outdoor workers.
- SO_2: sulphur dioxide level measured in μg/m³. High levels of sulphur dioxide can produce irritation in the skin and membranes, and worsen asthma or heart diseases in sensitive groups.
- CO: carbon monoxide level measured in mg/m³. Carbon monoxide poisoning involves headaches, dizziness and confusion in short exposures and can result in loss of consciousness, arrhythmias, seizures or even death in the long term.
- NO: nitric oxide level measured in μg/m³. This is a highly corrosive gas generated among others by motor vehicles and fuel burning processes.
- NO_2: nitrogen dioxide level measured in μg/m³. Long-term exposure is a cause of chronic lung diseases, and are harmful for the vegetation.
- PM10: particles smaller than 10 μm. Even though the cannot penetrate the alveolus, they can still penetrate through the lungs and affect other organs. Long term exposure can result in lung cancer and cardiovascular complications.
- NOx: nitrous oxides level measured in μg/m³. Affect the human respiratory system worsening asthma or other diseases, and are responsible of the yellowish-brown color of photochemical smog.

The aim is to predict the Ozone levels given measurements from the rest of the variables. Let's get to it!



```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
```

# Exploratory analysis

We start by loading the dataset in _.h5_ format. A very helpful explanation if given by the author of the dataset in [this](https://www.kaggle.com/diegovicente/a-short-introduction-to-hdf5-files) kernel. We will use the records of a single station out of the 18 available. Moreover, we select three features only as explanatory variables for the O3 concentration. Recall that this analysis does not focus on the model building itself, but instead on the model interpretation - gathering insights from a black-box model.


```python
with pd.HDFStore('madrid.h5') as data:
    df = data['28079016']


df = df.sort_index()
print(df.columns.values)
x_label = ['CO','NO_2','PM10','SO_2','NOx']
y_label = ['O_3']
df = df[y_label + x_label]
df = df.dropna() # There are quite a few nans so lets just remove them. We have enough data for our purposes
df.describe()

```

    ['CO' 'NO' 'NO_2' 'NOx' 'O_3' 'PM10' 'SO_2']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>O_3</th>
      <th>CO</th>
      <th>NO_2</th>
      <th>PM10</th>
      <th>SO_2</th>
      <th>NOx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>75746.000000</td>
      <td>75746.000000</td>
      <td>75746.000000</td>
      <td>75746.000000</td>
      <td>75746.000000</td>
      <td>75746.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>43.638169</td>
      <td>0.421912</td>
      <td>45.429437</td>
      <td>26.449752</td>
      <td>10.107252</td>
      <td>76.065385</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30.281593</td>
      <td>0.432375</td>
      <td>29.638354</td>
      <td>23.140751</td>
      <td>5.724532</td>
      <td>84.098601</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16.170000</td>
      <td>0.200000</td>
      <td>23.389999</td>
      <td>9.940000</td>
      <td>6.700000</td>
      <td>28.389999</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.245001</td>
      <td>0.310000</td>
      <td>39.150002</td>
      <td>19.930000</td>
      <td>8.430000</td>
      <td>50.130001</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>64.370003</td>
      <td>0.510000</td>
      <td>60.877501</td>
      <td>35.610001</td>
      <td>11.730000</td>
      <td>91.347498</td>
    </tr>
    <tr>
      <th>max</th>
      <td>184.300003</td>
      <td>10.520000</td>
      <td>324.200012</td>
      <td>397.000000</td>
      <td>108.300003</td>
      <td>1444.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pair plot
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x122680e48>




![png]({{ site.url }}/assets/insights_post/output_4_1.png)


There are seems to be some sort of linear-ish patterns so we can expect our model to work decently, at least for the purpose of this work.

# Regression model: a neural network with several layers
Next we split the dataset into a training and test set and afterwards define the model. The model we have chosen here is a neural network with 3 layers stacked on top of each other.


```python
# Split the data into test and training sets.
np.random.seed(100)
X_train, X_test, y_train, y_test = train_test_split(df[x_label],df[y_label],test_size=0.1)
# Print the dimensions
print('Training set dimensions X, y: ' + str(X_train.shape) + ' ' +str(y_train.shape))
print('Test set dimensions X, y: ' + str(X_test.shape) + ' '+ str(y_test.shape))
```

    Training set dimensions X, y: (68171, 5) (68171, 1)
    Test set dimensions X, y: (7575, 5) (7575, 1)



```python
# Define regression model in Keras
def regression_model():
    # Define model
    model = Sequential()
    model.add(Dense(5, input_dim=5, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])

    return model

# Use KerasRegressor wrapper (from Keras to sklearn)
# The packages we use are meant to be run with sklearn models
estimator = KerasRegressor(build_fn=regression_model, validation_split = 0.2, batch_size=100, epochs=100, verbose=0)
history = estimator.fit(X_train, y_train)
```


```python
# summarize history loss
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()
```

    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])



![png]({{ site.url }}/assets/insights_post/output_8_1.png)


The model seems to be fitted after a couple of epochs. Let's have a quick look at the residuals


```python
fitted = estimator.predict(X_train)
residuals = y_train['O_3'] - fitted
```


```python
# Two plots
fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(12,6))

# 1. Histogram of residuals
sns.distplot(residuals, ax=ax1)
ax1.set_title('Histogram of residuals')

# Fitted vs residuals
x1 = pd.Series(fitted, name='Fitted O_3')
x2 = pd.Series(y_train['O_3'], name="O_3 values")
sns.kdeplot(x1, x2, n_levels=40,ax = ax2)
sns.regplot(x=x1,y=x2, scatter=False, ax = ax2)
ax2.set_title('Fitted vs actual values')
ax2.set_xlim([0,120])
ax2.set_ylim([0,120])
ax2.set_aspect('equal')

```


![png]({{ site.url }}/assets/insights_post/output_11_0.png)


There is nothing too strange that stands out so let's keep on going. There are several points one would explore until this model can be considered satisfactory:

1. Time dependencies should be considered. The Ozone exhibits a really strong diurnal seasonality.
2. Hyperparameter tuning: shall we add/remove layers? What about increasing/decreasing the bath size or the learning rate? [Grid search](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/) on the hyperparameters is the way to go
3. Can we find more data, for example, related to traffic (like the author of the dataset did in the [last](https://www.kaggle.com/diegovicente/particle-levels-prediction-using-lstm) section), temperature or solar irradiance, that help explaining the Ozone levels?
4. Data could be normalized. I decided to skip this step in order to make the interpretations of the model easier.

The aim of this notebook is not to build the best possible model but instead to get some insights from it, especially when the model is a neural network and its effects are rather _"black-box"_.

Once we are satisfied with our model search, let's have a look at what we can learn from it.

# Permutation importance
We answer the following questions: _which of the explanatory variables is most relevant when predicting the O3 levels? And which one is not significant at all, and shall be removed?_

This question is almost as old as the field of statistics itself. When it comes to linear model and other white-box approaches, the straight-forward answer is given in traditional statistical books. What happens when the model is not as interpretable as a simple linear model? It is now not enough to look at the coefficients themselves. Instead, we make use of the computation power of our computers and calculate the so-called **Permutation importance**.

The intuition behind permutation importance in quite simple. The only requirement is to have fitted a model, either for regression or classification. If a feature is considered not important by the model, we could shuffle it (re-arrange the rows) and the performance of the model would not be altered much. On the other hand, if the feature is relevant, shuffling its rows with affect negatively the prediction accuracy.

Below we calculate the permutation importance for our model. Clearly there are two most importance features that should be studied more carefully. The least importance features, namely, SO_2 could be safely removed from the model.



```python
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(estimator, random_state=1).fit(X_train,y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist())
```





    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                2297.2632

                    &plusmn; 7.2732

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                NOx
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.41%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                470.6033

                    &plusmn; 3.9636

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                NO_2
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.40%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                124.4992

                    &plusmn; 4.9419

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                PM10
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                14.6793

                    &plusmn; 0.7235

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                CO
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                5.0095

                    &plusmn; 0.7353

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                SO_2
            </td>
        </tr>


    </tbody>
</table>























# Partial dependence plots
Permutation importance allowed us to find out which variable are most important in terms of predicting the Ozone levels. The next question comes naturally: _What is the effect of such variables in the Ozone concentrations?_ In the world of linear models this question is answered by looking at the coefficients. In the black-box world, we look at the [partial dependence](https://towardsdatascience.com/introducing-pdpbox-2aa820afd312) plots (PDP).


The underlying idea behind these plots is to marginalize the effect of one or two variables over the predicted values. When a neural network has several features and layers, it is really hard or even impossible to asses the impact of a single feature on the outcome by simply looking at coefficients. Instead, the marginal effect is approximated by a Monte Carlo approach. Generally speaking, we run predictions for a set of features and then average them out over the features we are interested in knowing their effects.





```python
from pdpbox import pdp, get_dataset, info_plots

# Gather pdp data
pdp_goals_NOx = pdp.pdp_isolate(model = estimator,
                                dataset = X_train,
                                model_features = x_label,
                                feature='NOx')
```


```python
# plot NOX pdp
pdp.pdp_plot(pdp_goals_NOx, 'NOx',
             x_quantile=False,
            plot_pts_dist=False)
plt.show()
```


![png]({{ site.url }}/assets/insights_post/output_17_0.png)



```python
# Gather pdp data
pdp_goals_NO2 = pdp.pdp_isolate(model = estimator,
                                dataset = X_train,
                                model_features = x_label,
                                feature='NO_2')
```


```python
# plot NO_2 pdp
pdp.pdp_plot(pdp_goals_NO2, 'NO_2',
            x_quantile=False,
            plot_pts_dist=False)
plt.show()
```


![png]({{ site.url }}/assets/insights_post/output_19_0.png)


From the figures above we can draw some interesting conclusions, for example:

- Higher levels of Ozone are predicted for lower levels of NOx and higher levels of NO_2
- The change in NO_2 has, generally speaking, a similar impact than the NOx but with opposite sign. There are some cases with extremely high NO_2 where the relationship with Ozone is more than 3 times the usual.
- The impact of NOx on the Ozone stabilizes after a values of NOx greater than 200

Of course, these conclusions should be confirmed with an expert in air quality. The conclusions can change if we change the model.

To finalize the example, we could have a a look at the combined effect of both NOx and NO_2. In the linear-model world, this would be equivalent to looking at the coefficient of the interaction between NOx and NO_2. In this case the effect is more complex so it is not enough to look at one number - instead we look at a 3d plot. In this case there was not much to show, below I paste the code for completeness.

```python
inter1 = pdp.pdp_interact(
    model = estimator,
    dataset=X_train,
    model_features = x_label,
    features = ['NOx', 'NO_2'],
    num_grid_points = [10, 10])

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out = inter1,
    feature_names=['NOx', 'NO_2'],
    plot_type='contour',
    plot_pdp=False,
    x_quantile=False    
)

```

# Shapely values

Shap values contribute understanding the model in a analogous way to coefficients from a linear model. _Given a prediction: how much of it is affected by the explanatory variables? What features contribute positively and negatively?_


The Shapley value is the average marginal contribution of a feature value over all possible combination of the other features ([wiki page](odesays.com/solutions-to-training-by-codility/) and an easier to read [book chapter](https://christophm.github.io/interpretable-ml-book/shapley.html)). They are useful for understanding the contributions the features of the model when we produce a prediction. Shapely values answer "why" the prediction is different than the mean prediction.



```python
import shap

# SHAP expects model functions to take a 2D numpy array as input, so we define a wrapper function around the original Keras predict function.
def f_wrapper(X):
    return estimator.predict(X).flatten()

# Too many input data - use a random slice
# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
X_train_summary = shap.kmeans(X_train, 20)

# Compute Shap values
explainer = shap.KernelExplainer(f_wrapper,X_train_summary)

# Make plot with combined shap values
# The training set is too big so let's sample it. We get enough point to draw conclusions
X_train_sample = X_train.sample(400)
shap_values  = explainer.shap_values(X_train_sample)
shap.summary_plot(shap_values, X_train_sample)

```

    100%|██████████| 400/400 [00:02<00:00, 135.37it/s]



![png]({{ site.url }}/assets/insights_post/output_22_1.png)


Each row corresponds to a feature, the color represents the feature value (red high, blue low) , and each dot corresponds to a training sample.

From the combined Shap plot values we observe the following:

- NOx is the feature with the highest impact. We already found this out when calculating the permutation importance. Nevertheless, it is always a good idea to double-check your conclusions. The higher NOx (red dots at the first row), the lower the Ozone predictions are.
- The effect of NO_2 is reverse to the NOx: higher NO_2 levels imply more Ozone
- PM10 seems somewhat relevant even though less important than the levels of nitrogen
- The CO and SO_2 are way less relevant and they could easily be removed from the model


# Conclusion

We have seen three ways of exploring the effect of different features on the predicted values of a model. These techniques are model-independent and are specially useful for black-box models like neural networks and random forests. **Shap values** is the **most useful** of all with the ability to show the positive/negative effect of each feature on the predicted variable in a very compact format. The drawback of the Shap values is their computational complexity. A deep analysis of any black-box model should include also a table of **permutation importance** and a set of **partial dependence plots**.

### Further reading

- Interpretable machine learning [book, chapter 6](https://christophm.github.io/interpretable-ml-book/pdp.html)
- ML for Insights [Challenge](https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values) from Kaggle
- Basic introduction to [pdp](https://towardsdatascience.com/introducing-pdpbox-2aa820afd312) from the author of the python library PDPBox
- Python library for calculating [Shap](https://github.com/slundberg/shap) values.
