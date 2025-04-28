# Introduction 

This project intends to explore building a Long Short Term Memory Neural Network and XGBoost from scratch for the fun of it. I'll be using Financial Data from Yahoo by an API.

![LSTM Model](./Images/LSTM%20Model.png)


# Methodology

## RNNs and LSTMs

There are different uses for recursive neural networks and its variants.

They can either be a:
1. Sequence to sequence: $(x^1, x^2, \dots, x^T)$, $(y^1, y^2, \dots, y^T)$
2. Sequence to one: $(x^1, x^2, \dots, x^T)$, $y^T$

Sequence to sequence accumulates a loss across the entire output vector.
Sequence to one only calculates the loss for the final prediciton and passes that along. 

Note that usually the output vectors are shifted versions of your original dataset.

While recursive neural netowrks and their variants can theoretically predict for any variable sequence of length, there will likely be a performance correlation with the shape of their data.

If your input shape is 5 steps during the training phase, your model will perform the best for 5 steps and suffer as you deviate from this.

This will be discussed in x section.

There are many methods to deal with this either by altering your prediction set to be structured in the same way as your training set, or altering your training set to make your neural network more robust. 

## Back Propogation Through Time

In general we can define our loss function as the mean of an objective function across our $T$ length sample:

```math
L = \frac{1}{T}\sum^T_{t=1}l(y_t,\hat{y}_t)
```

### Sequence to Sequence

In sequence to sequence, this loss stays the same.

Defining the fomrulas:


There is a dependency on the previous iteration resulting in a recursive nature. This intuitevely makes sense, each prediction from an earlier stage, contributes to a later stage and thus results in a recursive derivative.

### Sequence to One
In sequence to one, we only evaluate the final output of the last input $x^T$ after feeding in the rest of the sequence to the ground truth. 

```math
L = l(y_T,\hat{y}_T)
```

## First Neural Network Design 

### Overall Build

We'll use a 1 cell 1 layer LSTM to a single linear activation neural node to train on financial data of a company.

###


### LSTM Portion

An LSTM is constructed of 4 gates:
1. Forget Gate
2. Input Gate
3. State Candidate Gate
4. Output Gate

Forget, Input and Output gates all share the same structure and hence will inherit the same class Sigmoid Neuron. The State Candidate Gate for the sake of abstraction and clarity will also inherit from the Tanh Neuron class. 

We will use Mean Squared Error (MSE) as our loss function.

Data structure required for training an LSTM is the following:
X-coords, x-time steps, and then a prediction,.
Loop and feed the time steps until the last one.

The number of inputs required = number of features. 


### Derivation of gradients for back propigation 

To perform Stochastic Gradient Descent (SGD), we need the derivatives of the loss function with respect to the weights and biases. Given there are 4 gates with $n$ inputs, there will be $4$ biases and $4 * n$ weights, resulting in a lot of gradients. However, we do not need to calculate them all. In fact, we only need to calculate the gradients for the biases and weights for the Sigmoid and Tanh Neuron. This means we need $2$ bias gradients and $2 * n$ weight gradients.

However, given that the formula is a linear combination, we can generalise the formula. Therefore, only $2$ weight gradients need to be calculated (one for Tanh and one for Sigmoid) and can be extrapolated to the other weights in the formula. 

Let:
1. $W_{Si}$ be the weight for the i'th input in the Sigmoid Neuron
2. $b_{S}$ be the bias in the Sigmoid Neuron
3. $W_{Ti}$ be the weight for the i'th input in the Tanh Neuron
4. $b_{T}$ be the bias in the Tanh Neuron
5. $L = \frac{1}{n}\sum_{i=1}^n(y_{i}-\hat{y}_{i})$
6. $\eta$ be the learning rate

Performing SGD requires us to calculate their relavent gradients:
$$
W_{Si}^{\text{New}} = W_{Si}^{\text{Old}} - \eta \frac{dL}{dW_{Si}} 
$$
$$
b_{S}^{\text{New}} = b_{S}^{\text{Old}} - \eta \frac{dL}{b_{S}} 
$$
$$
W_{Ti}^{\text{New}} = W_{Ti}^{\text{Old}} - \eta \frac{dL}{dW_{Ti}} 
$$
$$
b_{T}^{\text{New}} = b_{T}^{\text{Old}} - \eta \frac{dL}{b_{T}} 
$$

#### Gradient Calculation


How is a LSTM trained?

Using chain rule, we can decompose the gradients. 

Let:
1. $\hat{y_i} = a_i + b_ih_t$
2. $h_t = S_t \times \tanh(c_t)$ - where $c_t$ is the cell state update
3. $S_t = \sigma(f_t)$ - where $\sigma$ is the sigmoid function
4. $c_t = {fo}_t \times c_{t-1} + i_{t} \times g_{t}$
5. $g_t = \tanh(f_t)$

PROBABLY SHOULD BE SEPARATING THE BACKWARD PASS OF MY LINEAR ACTIVATIon NODE.

Note that for number 1, this is due to a linear activation node at the end.
Note that for number 4, $n$ =number of features.

##### Sigmoid Neural Node

We can decompose the derivative of our loss function with respect to the weights of the Sigmoid Neural Node. This can also be applied to the additional weight for the hidden state ($h_{t-1}$) since we are dealing with a linear sum.

Let:

2. $f_t = \sum_{i=1}^n W_{Si}x_{Si} + W_{Sh_{t-1}}h_{t-1}+b_{St}$

###### Weights
```math
\frac{dL}{dW_{Si}} = \frac{dL}{d\hat{y}_{i}} \times \frac{d\hat{y}_{i}}{dh_{t}}
\times \frac{dh_{t}}{dS_{t}} \times \frac{dS_{t}}{df_{t}} \times \frac{df_{t}}{dW_{Si}} 
```

```math
\begin{align*}
\frac{dL}{d\hat{y}_{i}} &= \frac{2}{n}(\hat{y}_{i}-y_{i})  \\
\frac{d\hat{y}_{i}}{dh_{t}} &= b_{i} \\
\frac{dh_{t}}{dS_{t}} &= \tanh(c_{t})  \\
\frac{dS_{t}}{df_{t}} &= \sigma(f_{t})(1-\sigma(f_{t})) \\
\frac{df_{t}}{dW_{Si}} &= x_{Si}
\end{align*}
```
```math
\frac{dL}{dW_{Si}} = \frac{2}{n}(\hat{y}_{i}-y_{i}) \times b_{i} \times \tanh(c_{t}) \times \sigma(f_{t}) \times (1-\sigma(f_{t})) \times x_{Si}
```

###### Biases
```math
\frac{dL}{db_{S}} = \frac{dL}{d\hat{y}_{i}} \times \frac{d\hat{y}_{i}}{dh_{t}}
\times \frac{dh_{t}}{dS_{t}} \times \frac{dS_{t}}{df_{t}} \times \frac{df_{t}}{db_{St}} 
```

```math
\begin{align*}
\frac{dL}{d\hat{y}_{i}} &= \frac{2}{n}(\hat{y}_{i}-y_{i})  \\
\frac{d\hat{y}_{i}}{dh_{t}} &= b_{i} \\
\frac{dh_{t}}{dS_{t}} &= \tanh(c_{t})  \\
\frac{dS_{t}}{df_{t}} &= \sigma(f_{t})(1-\sigma(f_{t})) \\
\frac{df_{t}}{db_{St}} &= b_{St}
\end{align*}
```
```math
\frac{dL}{db_{St}} = \frac{2}{n}(\hat{y}_{i}-y_{i}) \times b_{i} \times \tanh(c_{t}) \times \sigma(f_{t}) \times (1-\sigma(f_{t})) \times b_{St}
```

##### Tanh Neural Node

We can decompose the derivative of our loss function with respect to the weights of the Tanh Neural Node. This can also be applied to the additional weight for the hidden state ($h_{t-1}$) since we are dealing with a linear sum.
$$
\frac{dL}{dW_{Si}} = \frac{dL}{d\hat{y}_{i}} \times \frac{d\hat{y}_{i}}{dh_{t}}
\times \frac{dh_{t}}{dS_{t}} \times \frac{dS_{t}}{df_{t}} \times \frac{df_{t}}{dW_{Si}} 
$$

Let:
1. $f_t = \sum_{i=1}^n W_{Ti}x_{Ti} + W_{Th_{t-1}}h_{t-1}+b_{Tt}$

###### Weights
```math
\frac{dL}{dW_{Ti}} = \frac{dL}{d\hat{y}_{i}} \times \frac{d\hat{y}_{i}}{dh_{t}}
\times \frac{dh_{t}}{dc_{t}} \times \frac{dc_{t}}{dg_{t}} \times \frac{dg_{t}}{df_{t}} \times \frac{df_{t}}{dW_{Ti}} 
```

```math
\begin{align*}
\frac{dL}{d\hat{y}_{i}} &= \frac{2}{n}(\hat{y}_{i}-y_{i})  \\
\frac{d\hat{y}_{i}}{dh_{t}} &= b_{i} \\
\frac{dh_{t}}{dc_{t}} &= (1-\tanh^2(c_t))   \\
\frac{dc_{t}}{dg_{t}} &= i_t \\
\frac{dg_{t}}{df_{t}}  &= i_t \\
\frac{df_{t}}{dW_{Ti}} &= x_{Ti}
\end{align*}
```
```math
\frac{dL}{dW_{Si}} = \frac{2}{n}(\hat{y}_{i}-y_{i}) \times b_{i} \times \tanh(c_{t}) \times \sigma(f_{t}) \times (1-\sigma(f_{t})) \times x_{Si}
```

###### Biases
```math
\frac{dL}{db_{S}} = \frac{dL}{d\hat{y}_{i}} \times \frac{d\hat{y}_{i}}{dh_{t}}
\times \frac{dh_{t}}{dS_{t}} \times \frac{dS_{t}}{df_{t}} \times \frac{df_{t}}{db_{St}} 
```

```math
\begin{align*}
\frac{dL}{d\hat{y}_{i}} &= \frac{2}{n}(\hat{y}_{i}-y_{i})  \\
\frac{d\hat{y}_{i}}{dh_{t}} &= b_{i} \\
\frac{dh_{t}}{dS_{t}} &= \tanh(c_{t})  \\
\frac{dS_{t}}{df_{t}} &= \sigma(f_{t})(1-\sigma(f_{t})) \\
\frac{df_{t}}{db_{St}} &= b_{St}
\end{align*}
```
```math
\frac{dL}{db_{St}} = \frac{2}{n}(\hat{y}_{i}-y_{i}) \times b_{i} \times \tanh(c_{t}) \times \sigma(f_{t}) \times (1-\sigma(f_{t})) \times b_{St}
```

### Second Network Design 

We'll use a 2 cell 1 layer LSTM to a single linear activation neural node to trade on financial data and a risk index of the stock market.