# Diagram

<div style="display: inline-block; margin-right: 10px;">
    <img src="./Images/LSTM Neural Network-Sigmoid Node.drawio.png" width="200" height="200">
</div>
<div style="display: inline-block;">
    <img src="./Images/LSTM Neural Network-Tanh Node.drawio.png" width="200" height="200">
</div>

<img src=".\Images\LSTM Neural Network-LTSM - Gates.drawio.png" width="300" height="200">

<img src=".\Images\LSTM Neural Network-LTSM.drawio.png" width="300" height="200">

<img src=".\Images\LSTM Neural Network-LTSM Notation.drawio.png" width="300" height="200">

# Definitions

First, we define the sum as: $u_{kt} = \sum_{i=1}^N W_{ki}x_{ti} +W_{kh}h_{t-1}+b_k $ where:
1. $k$ is the letter belonging to the node
2. $t$ is the sequence step
3. $N$ is the number of features

 This makes notation much easier to read.

1. Forget Gate: $f_t = \sigma(u_{ft})$
2. Input Gate: $I_t = \sigma(u_{It})$
3. Candidate Cell State: $s_t = \tanh(u_{st})$
4. Output Gate: $o_t = \sigma(u_{ot})$

Long Term Memory will be referred to as $c_t$ and Short Term Memory will be referred to as $h_t$

# Forward Propagation 

1. Long Term Memory Update: $c_t = f_t \times c_{t-1} + I_t \times s_t$
2. Short Term Memory Update: $h_t = o_t \times \tanh(c_t)$

# Backward Propagation 

Let's take the simple case, where we are training a sequence to one. We only compare the final output to the ground truth.

```math
\begin{align*}
L &= l(y_t, \hat{y}_t)\\
 &= l(y_t, h_t)
\end{align*}
```

Since we want to calculate the differential of the loss with respect to the weights, we need to do this for each of the gates due to the fact each gradient passes by a different path. 

Since we are just dealing with a linear sum, if we find the derivative of the loss with respect to one of the weights, then this will be correct for all of the weights for the given node.

```math
\begin{align*}
\frac{dL}{dW_{oi}} &= \frac{dL}{dh_t} \frac{dh_t}{do_t} \frac{do_t}{du_{ot}} \frac{du_{ot}}{dW_{oi}} \\
\frac{dL}{dW_{fi}} &= \frac{dL}{h_t} \frac{dh_t}{dc_t} \frac{dc_t}{df_t} \frac{df_t}{du_{ft}}\frac{du_{ft}}{dW_{fi}}\\
\frac{dL}{dW_{Ii}} &= \frac{dL}{dh_t} \frac{dh_t}{dc_t} \frac{dc_t}{dI_t} \frac{dI_t}{du_{It}}\frac{du_{It}}{dW_{Ii}}
\end{align*}
```

However, there is one big issue. Calculating their respective $u_{kt}$ with respect to the weight. This is due to the recursive nature of the formula and is what motivates backpropogation through time.

# Illustrative example of the problem of calculating the derivative

Let us attempt to calculate $\frac{du_{ot}}{dW_{oi}}$

```math
\begin{align*}
\frac{du_{ot}}{dW_{oi}}  &= \frac{d}{dW_{oi}}\left(\sum_{i=1}^N W_{oi}x_{ti} +W_{oh}h_{t-1}+b_o \right) \\
&= \frac{d}{dW_{oi}}\left( W_{oi}x_{ti} +W_{oh}h_{t-1}\right) \\
&= x_{ti} +W_{oh}\frac{d}{dW_{oi}} (h_{t-1}) \\
\end{align*}
```

But, $h_{t-1} = o_{t-1} \times \tanh(c_{t-1})$ and $o_{t-1}$ contains $W_{oh}$.

```math
\begin{align*}
\frac{du_{ot}}{dW_{oi}} &= x_{ti} +W_{oh}\frac{dh_{t-1}}{dW_{oi}} \\
&=x_{ti}+W_{oh}\frac{dh_{t-1}}{do_{t-1}}\frac{do_{t-1}} {dW_{oi}} \\
&=x_{ti}+W_{oh}\frac{dh_{t-1}}{do_{t-1}}\frac{do_{t-1}} {du_{o, t-1}} \frac{du_{o, t-1}}{dW_{oi}}
\end{align*}
```

This will rpeat rescursively from $t, t-1, \dots, 1$. This is intutive because we share the same weight when making all the predictions and each short term prediction takes the long term memory state to calculate the short term memroy state which then influcense the next or simply:
1. Use short term memory $t-1$ and long term memory $t-1$ to calculate long term memory at $t$
2. Use long term memory $t$ to calculate shorter term memory $t$

Therefore, the prediction from stage 1 will intuitively influence the prediction at stage 2 and stage 3, since we continuously add it onto the long term memroy. Therefore, if our prediction at stage 1 is off, then stage 2 and stage 3 and so on will ahve to correct it eveen further.

However, since this is recursive we can beegin with t=1, as $h_{t-1}$ is set to $0$ at the start and then iteravely add. This back propogation through time.

```math
\begin{align*}
\frac{du_{o1}}{dW_{oi}}
&=x_{1i} \\
\frac{du_{o2}}{dW_{oi}}
&=x_{2i}+W_{oh}\frac{dh_{1}}{do_{1}}\frac{do_{1}} {du_{o, 1}} \frac{du_{o, 1}}{dW_{oi}} \\
&=x_{2i}+W_{oh}\frac{dh_{1}}{do_{1}}\frac{do_{1}} {du_{o, 1}} x_{1i}
\end{align*}
```
Now it's a question of what $\frac{dh_{t}}{do_{t}},\frac{do_{t}} {du_{o, t}}$ are:
```math
\begin{align*}
\frac{dh_{t}}{do_{t}} &=\frac{d}{do_{t}} o_t \tanh(c_t) \\
&= \tanh(c_t) \\
\frac{do_{t}} {du_{o, t}} &=  \frac{d} {du_{o, t}} \sigma(u_{ot}) \\
&= \sigma(u_{ot})(1-\sigma(u_{ot})) \\
&= o_t(1-o_t) 
\end{align*}
```
Therefore:
```math
\begin{align*}
\frac{du_{ot}}{dW_{oi}} &=x_{ti}+W_{oh}\frac{dh_{t-1}}{do_{t-1}}\frac{do_{t-1}} {du_{o, t-1}} \frac{du_{o, t-1}}{dW_{oi}} \\
 &=x_{ti}+W_{oh}\tanh(c_{t-1})o_{t-1}(1-o_{t-1}) \frac{du_{o, t-1}}{dW_{oi}}
\end{align*}
```

Where we rescursievely calculate and add it on $\frac{du_{o, t-1}}{dW_{oi}}$. Given that when we do a forward proprogation, we will already have the $c_{t-1}$ and $o_{t-1}$ values. This hsouldn't be computationally expesnive and we can represent this in a matrix format.

Also note that the gradient doesn't suffer from a vansihing gradient as there will always be $x_{ti}$ unaffected by the multiplication. However, if previous gradients are large, or $W_{oh}$ is large, this can still suffer from gradient explosion. However, this should be in specific cases since tanh is bounded between-1 and 1nd therefore will always be regulating the product. 


We can repeat this process for:
```math
\begin{align*}
\frac{du_{ot}}{dW_{oh}} &= h_{t-1} + \frac{du_{ot}}{dh_{t-1} } \frac{dh_{t-1} }{dW_{oh}} \\
&= h_{t-1} + W_{oh}\frac{dh_{t-1}}{dW_{oh}} \\
&= h_{t-1} + W_{oh}\frac{dh_{t-1}}{do_{t-1}}\frac{do_{t-1}} {du_{o, t-1}} \frac{du_{o, t-1}}{dW_{oh}} \\
&= h_{t-1}+W_{oh}\tanh(c_{t-1})o_{t-1}(1-o_{t-1}) \frac{du_{o, t-1}}{dW_{oh}}
\end{align*}
```

As mentioned beforehand, this is near identical as we are deriving the linear sum and the derivatives will be the same for all weighted components and will just use the input variable it takes. 

Lastly, lets just calculate the derivative with respect to the bias (to show working out).

```math
\begin{align*}
\frac{du_{ot}}{db_o} &= 1 + \frac{du_{ot}}{dh_{t-1} } \frac{dh_{t-1} }{db_o} \\
&= 1 + W_{oh}\frac{dh_{t-1} }{db_o} \\
&= 1 + W_{oh}\frac{dh_{t-1}}{do_{t-1}}\frac{do_{t-1}} {du_{o, t-1}} \frac{du_{o, t-1}}{db_{o}} \\
&= 1+W_{oh}\tanh(c_{t-1})o_{t-1}(1-o_{t-1}) \frac{du_{o, t-1}}{dW_{bo}}
\end{align*}
```
Again, near identical. The 1 is just the coefficient in front of $b_{o}$.

## Recap

So far, we've calculated the backward propogation through time equations for:

```math
\begin{align*}

\frac{du_{ot}}{dW_{oi}} &=  x_{ti}+W_{oh}\tanh(c_{t-1})o_{t-1}(1-o_{t-1}) \frac{du_{o, t-1}}{dW_{oi}} \\

\frac{du_{ot}}{dW_{oh}} &= h_{t-1}+W_{oh}\tanh(c_{t-1})o_{t-1}(1-o_{t-1}) \frac{du_{o, t-1}}{dW_{oh}} \\

\frac{du_{ot}}{db_o} &= 1+W_{oh}\tanh(c_{t-1})o_{t-1}(1-o_{t-1}) \frac{du_{o, t-1}}{dW_{bo}}
\end{align*}
```

Again, only the first term and derivative at the end is different. We will exploit this pattern in the calculation of the other gradients.

# Backward Propogation Through Time

I will derive the rest of the equations.

## $\frac{du_{ft}}{dW_{fi}}$

The only thing that differs in each derivation in the chain breakdown to get from one step to another. I will illustrate this once and use this method of thinking for the rest.

```math
\begin{align*}
\frac{du_{ft}}{dW_{fi}} &= x_{fi} + \frac{du_{ft}}{dh_{t-1}}\frac{dh_{t-1}}{dW_{fi}} \\
&= x_{fi} + W_{fh}\frac{dh_{t-1}}{dW_{fi}}
\end{align*}
```

This is the same so far, but notice the difference in decomposition of $\frac{dh_{t-1}}{dW_{fi}}$ by chain rule compared to $\frac{dh_{t-1}}{dW_{oi}}$. This is because the gradient follows a different path through the gates. Remember, the weights and biases are the only thing shared across time.

```math
\begin{align*}
\frac{du_{ft}}{dW_{fi}} &= x_{fi} + W_{fh}\frac{dh_{t-1}}{dW_{fi}} \\
&= x_{fi} + W_{fh}\frac{dh_{t-1}}{dc_{t-1}}\frac{dc_{t-1}}{dW_{fi}} \\
&= x_{fi} + W_{fh}\frac{dh_{t-1}}{dc_{t-1}}\frac{dc_{t-1}}{df_{t-1}}\frac{df_{t-1}}{dW_{fi}} \\ 
&= x_{fi} +W_{fh}\frac{dh_{t-1}}{dc_{t-1}}\frac{dc_{t-1}}{df_{t-1}}\frac{df_{t-1}}{du_{f,t-1}}\frac{du_{f,t-1}}{dW_{fi}} 
\end{align*}
```
And like before, we just need to calculae each of these gradients excluding $\frac{du_{f,t-1}}{dW_{fi}}$ as we solve for this recursively.

```math
\begin{align*}

\frac{dh_{t}}{dc_{t}} &= o_t\frac{d}{dc_{t}} \tanh(c_t) \\
&= o_t(1-\tanh^2(c_t))\\
\frac{dc_{t}}{df_{t}} &= \frac{d}{df_{t}}\left(f_t  c_{t-1} + I_t  s_t \right) \\
&= c_{t-1}\\
\frac{df_{t}}{du_{f,t}} &= \frac{d}{du_{f,t}}\sigma(u_{ft}) \\
 &= f_t(1-f_t)
\end{align*}
```


```math
\begin{align*}
\frac{du_{ft}}{dW_{fi}} &= x_{fi} +W_{fh}o_{t-1}(1-\tanh^2(c_{t-1}))c_{t-2}f_t(1-f_t)\frac{du_{f,t-1}}{dW_{fi}} 
\end{align*}
```

Therefore, exploiting the pattern because of the linear summation:

```math
\begin{align*}
\frac{du_{ft}}{dW_{fi}} &= x_{fi} +W_{fh}o_{t-1}(1-\tanh^2(c_{t-1}))c_{t-2}f_t(1-f_t)\frac{du_{f,t-1}}{dW_{fi}} \\
\frac{du_{ft}}{dW_{fh}} &= h_{t-1} +W_{fh}o_{t-1}(1-\tanh^2(c_{t-1}))c_{t-2}f_t(1-f_t)\frac{du_{f,t-1}}{dW_{fh}} \\
\frac{du_{ft}}{db_f} &= 1 +W_{fh}o_{t-1}(1-\tanh^2(c_{t-1}))c_{t-2}f_t(1-f_t)\frac{du_{f,t-1}}{db_f} 
\end{align*}
```

## $\frac{du_{st}}{dW_{si}}$

## $\frac{du_{it}}{dW_{it}}$


# Applying these results to calculate $\frac{dL}{dW_{kt}}$ 


# Applying these results to a sequence to sequence LSTM model


```math
\begin{align*}
L &= l(y_T-\hat{y}_T)^2 \\
 &= (y_T-h_T)^2
\end{align*}
```