# Structure




Consider:
First, we define the sum as: $u_{kt} = \sum_{i=1}^N W_{ki}x_{ti} +W_{kh}h_{t-1}+b_k $ where:
1. $k$ is the letter belonging to the node
2. $t$ is the sequence step
3. $N$ is the number of features


Also define:
```math
h_t = \phi(u_{t})
```

# Derivation of Back Propogation

## Important Recursive Identity

I will derive the identity here for those who are curious, but you can skip to the end and just use the identity.

Given, $a_t = b_t + c_ta_{t-1}$ and $a_0 = 0$ and $ t = {1,2, \dots, T}$:

```math
\begin{align*}
a_1 &= b_1 \\
a_2 &= b_2 + c_2b_1 \\
a_3 &= b_3 + c_3(b_2 + c_1b_1) \\
&= b_3 + c_3 b_2 + c_3 c_2b_1 \\ 
a_4 &= b_4 + c_4(b_3 + c_3 b_2 + c_3 c_2b_1) \\
&= b_4 + c_4b_3 + c_4c_3 b_2 + c_4c_3 c_2b_1
\end{align*}
```
As you can see, there is a pattern, we start accumulating $c_k+1, c_k+2, \dots$ product wise after the first instance of $b_k$.

We can also express our $a_t$ as a sum of $b_t$ and $c_t$ as long as we collapse the recursive sum from $T, T-1, \dots, 1$ where $T$ is our time/sequence period.

So how do we define this as a formula? Let's start from the basics.

```math
a_t = b_t + \sum b_i

```

Ignore, the $c_t$ terms for now. We recognise that for every collapse we do in the recursive formula we add on an additional $b_t$ term. We recognise from the pattern above there will be a sequence of $b_t$ from $1, 2, \dots, t$. Since we already have $b_t$ in the formula. Therefore, the sum is quite easy to define. 

```math
a_t = b_t + \sum_{i=1}^{t-1} b_i

```

Now, incorporating the $c_t$ terms, which is a chain of products.

```math
a_t = b_t + \sum_{i=1}^{t-1}(\prod c_j) b_i

```
The question now is what do we define the bounds of the product as. This is pretty simple. Recognising that we start incrementing after the index of the $b_i$ (e.g. if we have $b_3$, then our first $c$ term will be $c_4$), $j$ must start counting from $i+1$ and end at $t$ (examining the pattern from above).

```math
a_t = b_t + \sum_{i=1}^{t-1}\left(\prod_{j=i+1}^t c_j\right) b_i

```

Remember this formula, it is important.

## Derivation

In order to use Stochastic Gradient Descent, we need to calculate:

```math
\frac{dL}{dW_h}, \frac{dL}{dW_i}
```

Let our loss function be $L_t = l(y_t, h_t)$

```math
\begin{align*}
\frac{dL}{dW_h} &= \sum_{t=1}^T \frac{dL_t}{dW_h} \\
\frac{dL_t}{dW_h} &= \frac{dL_t}{dh_t}\frac{dh_t}{du_t}\frac{du_t}{dW_h}
\end{align*}
```

Calculating

```math
\begin{align*}
\frac{du_t}{dW_h} &=  \frac{du_t}{dW_h} + \frac{du_t}{dh_{t-1}}\frac{dh_{t-1}}{dW_h} \\

\frac{dh_t}{dW_h} &= \frac{dh_t}{du_t}\left(\frac{du_t}{dW_h} + \frac{du_t}{dh_{t-1}}\frac{dh_{t-1}}{dW_h} \right) \\
&= \phi'(u_t)\left(\frac{du_t}{dW_h} + \frac{du_t}{dh_{t-1}}\frac{dh_{t-1}}{dW_h} \right) \\
&= \phi'(u_t)\frac{du_t}{dW_h} + \phi'(u_t)\frac{du_t}{dh_{t-1}}\frac{dh_{t-1}}{dW_h} \\
&= \phi'(u_t)h_{t-1} + \phi'(u_t)W_h\frac{dh_{t-1}}{dW_h}
\end{align*}
```
This equation should look familar because it is identical to $a_t = b_t + c_ta_{t-1}$ which can be expressed as $a_t = b_t + \sum_{i=1}^{t-1}\left(\prod_{j=i+1}^t c_j\right) b_i$ where:
1. $a_t = \frac{dh_t}{dW_h}$  
2. $b_t = \phi'(u_t)h_{t-1}$
3. $c_t = \phi'(u_t)W_h$

Plugging this into the identity, we get:

```math
\frac{dh_t}{dW_h} = \phi'(u_t)h_{t-1} + \sum_{i=1}^{t-1}\left(\prod_{j=i+1}^t \phi'(u_j)W_h\right) \phi'(u_i)h_{i-1}
```

While this formula doesn't require us to recursively calculate our gradients, it is quite computaintially expensive to calculate from the start. This formula would be useful if we only need to do it a few times.

If we calculate the gradients one at a time and then just add, it'll be faster overall as we need to perform this operation multiple times and we can store the gradients sequentially. We also do need the gradients to be resused multiple times, so the recursive formula makes more sense.

However, this formula can help us prove gradient explosion and disappearnace. 

# Proof of Gradient Explosion/Disappearance 

RNNs suffer from gradient exploision and disaparance. However, I couldn't find a good mathematical proof that didn't glaze over important details or made mistakes. Plus, I like proofs of inefficinces in a model design since you can then alter it with potential solutions (hint: LSTMs)

Beginning with:

```math
\begin{align*}
\frac{dh_t}{dW_h} &= \phi'(u_t)h_{t-1} + \sum_{i=1}^{t-1}\left(\prod_{j=i+1}^t \phi'(u_j)W_h\right) \phi'(u_i)h_{i-1} \\
\frac{dh_t}{dW_h} &= \phi'(u_t)h_{t-1} + \sum_{i=1}^{t-1}\red{{W_h^{t-i}}}\left(\prod_{j=i+1}^t \phi'(u_j)\right) \phi'(u_i)h_{i-1}
\end{align*}
```

Since $W_h$ is a constant, we can take it out of the product. From $i+1$ terms to $t$ terms, there are $t-i$ terms in between. Therefore, $W_h$ will multiply by itself $t-i$ terms, i.e. raise to the power of $t-i$.

This is why RNNs suffer from gradient explosion. We know that for $x>0$, $x^t \rightarrow \infty$ as $t \rightarrow \infty$. While the sum is well behaved for low $t$ and latest/closer terms behave well, as it continues to grow terms at earlier/farther time steps blow up because of the power to the weight since it is shared across al time periods.

This could be interprted as the model experience a kind of memory overflow, where in the attempt to remember every single step it collapses in on itself. When using this on SGD, coefficients become unstable and tend towards infinity as the gradients tends towards infinity.

On the other hand, gradient disappearnace can be interpeted as the model forgetting the contributions of the prior time steps.

First, consider both the $\tanh$ and $\sigma$ activation functions and their derivatives.

1. If $t(x) = \tanh(x)$, $t'(x) = 1-t(x)^2$
2. If $s(x) = \sigma(x)$, $s'(x) = s(x)(1-s(x))$

Considering their bounds: 
1. $t(x) \in (-1,1)$ then $t'(x) \in (0,1]$
2. $s(x) \in (0,1)$ then $s'(x) \in (0,\frac{1}{4}]$

Now notice:

```math
\begin{align*}
\frac{dh_t}{dW_h} &= \phi'(u_t)h_{t-1} + \sum_{i=1}^{t-1}{W_h^{t-i}}\left(\prod_{j=i+1}^t \red{\phi'(u_j)}\right) \phi'(u_i)h_{i-1}
\end{align*}
```
If $\phi(u_j) = \tanh(u_j)$ or $\phi(u_j) = \sigma(u_j)$.

If you multiply lots of these derivatives of the activations at a time, you get an incredibly small number since the factors are all less than $1$. This means earlier time steps get forgotten and modelling their long term impact becomes difficult. This is a problem as we want to use long term patterns for future output. Their gradients vanish as the sum grows.

Also consider the gradient we pass backwards:
```math
\begin{align*}
\frac{dh_t}{dx_{it}} &= \frac{dh_t}{du_t}\frac{du_t}{dx_{it}} \\
&= \phi'(u_t)W_{ki}

\end{align*}
```

So how do we preserve long term information? <a href = "Financial-Data-Prediction\Theory Documentation\LSTM Theory.md"> LSTMs!</a>