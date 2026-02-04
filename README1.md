`N`-neurons Neural Networks
===========================

[Math](https://github.com/sjbiaga/nnn/blob/main/README.md#math) (cont'd)
------------------------------------------------------------------------

Consider the same neural network with three hidden layers (the last of each is the output
layer) and three neurons per each layer, like in the following table:

| Input | Layer | Layer | Layer/Output |
|:-----:|:-----:|:-----:|:------------:|
|  $1$  |  $1$  |  $1$  |              |
| $x_1$ | $a_{11} {\phi_{11} \atop \longrightarrow} h_{11}$ | $a_{12} {\phi_{12} \atop \longrightarrow}h_{12}$ | $a_{13} {\phi_{13} \atop \longrightarrow} y_1$ |
| $x_2$ | $a_{21} {\phi_{21} \atop \longrightarrow} h_{21}$ | $a_{22} {\phi_{22} \atop \longrightarrow}h_{22}$ | $a_{23} {\phi_{23} \atop \longrightarrow} y_2$ |
| $x_3$ | $a_{31} {\phi_{31} \atop \longrightarrow} h_{31}$ | $a_{32} {\phi_{32} \atop \longrightarrow}h_{32}$ | $a_{33} {\phi_{33} \atop \longrightarrow} y_3$ |

The layers are fully connected, in the sense of the following nine equations:


```math
\begin{align*}
a_{11} = w_{11}^0 \times 1 + w_{11}^1 \times x_1 + w_{11}^2 \times x_2 + w_{11}^3 \times x_3 & & a_{12} = w_{12}^0 \times 1 + w_{12}^1 \times h_{11} + w_{12}^2 \times h_{21} + w_{12}^3 \times h_{31} & & a_{13} = w_{13}^0 \times 1 + w_{13}^1 \times h_{12} + w_{13}^2 \times h_{22} + w_{13}^3 \times h_{32} \\
a_{21} = w_{21}^0 \times 1 + w_{21}^1 \times x_1 + w_{21}^2 \times x_2 + w_{21}^3 \times x_3 & & a_{22} = w_{22}^0 \times 1 + w_{22}^1 \times h_{11} + w_{22}^2 \times h_{21} + w_{22}^3 \times h_{31} & & a_{23} = w_{23}^0 \times 1 + w_{23}^1 \times h_{12} + w_{23}^2 \times h_{22} + w_{23}^3 \times h_{32} \\
a_{31} = w_{31}^0 \times 1 + w_{31}^1 \times x_1 + w_{31}^2 \times x_2 + w_{31}^3 \times x_3 & & a_{32} = w_{32}^0 \times 1 + w_{32}^1 \times h_{11} + w_{32}^2 \times h_{21} + w_{32}^3 \times h_{31} & & a_{33} = w_{33}^0 \times 1 + w_{33}^1 \times h_{12} + w_{33}^2 \times h_{22} + w_{33}^3 \times h_{32} \\
\end{align*}
```

where $w_{ij}^k$ is the weight of the $i^{th}$ neuron on the $j^{th}$ layer with
respect to the $k^{th}$ output from the previous layer. We have, in matrix form:

```math
\begin{align*}
\begin{pmatrix}
a_{11} \\
\\
a_{21} \\
\\
a_{31}
\end{pmatrix} =
\begin{pmatrix}
w_{11}^0 & w_{11}^1 & w_{11}^2 & w_{11}^3 \\
\\
w_{21}^0 & w_{21}^1 & w_{21}^2 & w_{21}^3 \\
\\
w_{31}^0 & w_{31}^1 & w_{31}^2 & w_{31}^3
\end{pmatrix} \cdot
\begin{pmatrix}
1 \\
\\
x_1 \\
\\
x_2 \\
\\
x_3
\end{pmatrix} & &
\begin{pmatrix}
a_{12} \\
\\
a_{22} \\
\\
a_{32}
\end{pmatrix} =
\begin{pmatrix}
w_{12}^0 & w_{12}^1 & w_{12}^2 & w_{12}^3 \\
\\
w_{22}^0 & w_{22}^1 & w_{22}^2 & w_{22}^3 \\
\\
w_{32}^0 & w_{32}^1 & w_{32}^2 & w_{32}^3
\end{pmatrix} \cdot
\begin{pmatrix}
1 \\
\\
h_{11} \\
\\
h_{21} \\
\\
h_{31}
\end{pmatrix} & &
\begin{pmatrix}
a_{13} \\
\\
a_{23} \\
\\
a_{33}
\end{pmatrix} =
\begin{pmatrix}
w_{13}^0 & w_{13}^1 & w_{13}^2 & w_{13}^3 \\
\\
w_{23}^0 & w_{23}^1 & w_{23}^2 & w_{23}^3 \\
\\
w_{33}^0 & w_{33}^1 & w_{33}^2 & w_{33}^3
\end{pmatrix} \cdot
\begin{pmatrix}
1 \\
\\
h_{12} \\
\\
h_{22} \\
\\
h_{32}
\end{pmatrix}
\end{align*}
```

The outputs can be written more briefly using indices:

```math
\begin{align*}
h_{ij} = \phi_{ij}(a_{ij}) & & i = \overline{1 \dots 3}, j = \overline{1 \dots 2} \\
y_i = \phi_{i3}(a_{i3}) & & i = \overline{1 \dots 3}
\end{align*}
```

In order to compute the partial derivatives of the loss function $L$ with respect
to the weights of the first neuron in the first hidden layer, the following are
the nine _paths_ that enter the calculation; three for the $y_1$ output:

```math
\begin{align*}
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{12} {\phi_{12} \atop \longrightarrow}h_{12} & \longrightarrow & a_{13} {\phi_{13} \atop \longrightarrow} y_1 \\
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{22} {\phi_{22} \atop \longrightarrow}h_{22} & \longrightarrow & a_{13} {\phi_{13} \atop \longrightarrow} y_1 \\
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{32} {\phi_{32} \atop \longrightarrow}h_{32} & \longrightarrow & a_{13} {\phi_{13} \atop \longrightarrow} y_1 \\
\end{align*}
```

three for the $y_2$ output:

```math
\begin{align*}
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{12} {\phi_{12} \atop \longrightarrow}h_{12} & \longrightarrow & a_{23} {\phi_{23} \atop \longrightarrow} y_2 \\
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{22} {\phi_{22} \atop \longrightarrow}h_{22} & \longrightarrow & a_{23} {\phi_{23} \atop \longrightarrow} y_2 \\
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{32} {\phi_{32} \atop \longrightarrow}h_{32} & \longrightarrow & a_{23} {\phi_{23} \atop \longrightarrow} y_2 \\
\end{align*}
```
and three for the $y_3$ output:

```math
\begin{align*}
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{12} {\phi_{12} \atop \longrightarrow}h_{12} & \longrightarrow & a_{33} {\phi_{33} \atop \longrightarrow} y_3 \\
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{22} {\phi_{22} \atop \longrightarrow}h_{22} & \longrightarrow & a_{33} {\phi_{33} \atop \longrightarrow} y_3 \\
a_{11} {\phi_{11} \atop \longrightarrow} h_{11} & \longrightarrow & a_{32} {\phi_{32} \atop \longrightarrow}h_{32} & \longrightarrow & a_{33} {\phi_{33} \atop \longrightarrow} y_3 \\
\end{align*}
```

There are nine other computation paths for the weights of the second neuron, and
nine other paths for the weights of the third neuron (in the first hidden layer).

We shall further illustrate the computations only for the $k^{th}$ weight of the
_first_ neuron (in the _first_ hidden layer), where $k = \overline{0 \dots 3}$:

```math
\begin{align*}
\frac{\partial{L}}{\partial{w_{11}^k}} & = \frac{\partial{L}}{\partial{y_1}} \times \frac{\partial{\phi_{13}}}{\partial{a_{13}}} \times \frac{\partial{a_{13}}}{\partial{h_{12}}} \times \frac{\partial{\phi_{12}}}{\partial{a_{12}}} \times \frac{\partial{a_{12}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_1}} \times \frac{\partial{\phi_{13}}}{\partial{a_{13}}} \times \frac{\partial{a_{13}}}{\partial{h_{22}}} \times \frac{\partial{\phi_{22}}}{\partial{a_{22}}} \times \frac{\partial{a_{22}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_1}} \times \frac{\partial{\phi_{13}}}{\partial{a_{13}}} \times \frac{\partial{a_{13}}}{\partial{h_{32}}} \times \frac{\partial{\phi_{32}}}{\partial{a_{32}}} \times \frac{\partial{a_{32}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_2}} \times \frac{\partial{\phi_{23}}}{\partial{a_{23}}} \times \frac{\partial{a_{23}}}{\partial{h_{12}}} \times \frac{\partial{\phi_{12}}}{\partial{a_{12}}} \times \frac{\partial{a_{12}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_2}} \times \frac{\partial{\phi_{23}}}{\partial{a_{23}}} \times \frac{\partial{a_{23}}}{\partial{h_{22}}} \times \frac{\partial{\phi_{22}}}{\partial{a_{22}}} \times \frac{\partial{a_{22}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_2}} \times \frac{\partial{\phi_{23}}}{\partial{a_{23}}} \times \frac{\partial{a_{23}}}{\partial{h_{32}}} \times \frac{\partial{\phi_{32}}}{\partial{a_{32}}} \times \frac{\partial{a_{32}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_3}} \times \frac{\partial{\phi_{33}}}{\partial{a_{33}}} \times \frac{\partial{a_{33}}}{\partial{h_{12}}} \times \frac{\partial{\phi_{12}}}{\partial{a_{12}}} \times \frac{\partial{a_{12}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_3}} \times \frac{\partial{\phi_{33}}}{\partial{a_{33}}} \times \frac{\partial{a_{33}}}{\partial{h_{22}}} \times \frac{\partial{\phi_{22}}}{\partial{a_{22}}} \times \frac{\partial{a_{22}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_3}} \times \frac{\partial{\phi_{33}}}{\partial{a_{33}}} \times \frac{\partial{a_{33}}}{\partial{h_{32}}} \times \frac{\partial{\phi_{32}}}{\partial{a_{32}}} \times \frac{\partial{a_{32}}}{\partial{h_{11}}} \times \frac{\partial{\phi_{11}}}{\partial{a_{11}}} \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```

Rewriting all but the first and last factor in each term, we obtain:

```math
\begin{align*}
\frac{\partial{L}}{\partial{w_{11}^k}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a_{12}) \times w_{12}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a_{22}) \times w_{22}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a_{32}) \times w_{32}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a_{12}) \times w_{12}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a_{22}) \times w_{22}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a_{32}) \times w_{32}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a_{12}) \times w_{12}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a_{22}) \times w_{22}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a_{32}) \times w_{32}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```

We now group the first, fourth, and seventh terms, the second, fifth, and eighth terms,
respectively, the third, sixth, and ninth terms, then factor out; we get:

```math
\begin{align*}
\frac{\partial{L}}{\partial{w_{11}^k}} & = \left( \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a_{12}) + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a_{12}) + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a_{12}) \right) \times w_{12}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \left( \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a_{22}) + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a_{22}) + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a_{22}) \right) \times w_{22}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```
```math
\begin{align*}
& + \left( \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a_{32}) + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a_{32}) + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a_{32}) \right) \times w_{32}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```

Notice how the expressions in parentheses are exactly the elements of the $\delta$
matrix:

```math
\begin{align*}
\frac{\partial{L}}{\partial{w_{11}^k}} & = \delta_{11} \times w_{12}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
& + \delta_{12} \times w_{22}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
& + \delta_{13} \times w_{32}^1 \times \phi_{11}'(a_{11}) \times \frac{\partial{a_{11}}}{\partial{w_{11}^k}}
\end{align*}
```

For example, the partial derivative of $L$ with respect to the weight $w_{11}^2$
corresponding to the $x_2$ input, is the following:

```math
\begin{align*}
\frac{\partial{L}}{\partial{w_{11}^2}} & = \delta_{11} \times w_{12}^1 \times \phi_{11}'(a_{11}) \times x_2
& + \delta_{12} \times w_{22}^1 \times \phi_{11}'(a_{11}) \times x_2
& + \delta_{13} \times w_{32}^1 \times \phi_{11}'(a_{11}) \times x_2
\end{align*}
```

Let us observe what is the product of the transpose of the $2^{nd}$ layer's
weights matrix, and (current) delta:

```math
{W^{(2)}}^T \cdot \delta =
\begin{pmatrix}
w_{12}^0 & w_{22}^0 & w_{32}^0 \\
w_{12}^1 & w_{22}^1 & w_{32}^1 \\
w_{12}^2 & w_{22}^2 & w_{32}^2 \\
w_{12}^3 & w_{22}^3 & w_{32}^3
\end{pmatrix} \cdot
\begin{pmatrix}
\delta_{11} \\
\delta_{12} \\
\delta_{13}
\end{pmatrix} =
\begin{pmatrix}
\delta_{11} \times w_{12}^0 + \delta_{12} \times w_{22}^0 + \delta_{13} \times w_{32}^0 \\
\delta_{11} \times w_{12}^1 + \delta_{12} \times w_{22}^1 + \delta_{13} \times w_{32}^1 \\
\delta_{11} \times w_{12}^2 + \delta_{12} \times w_{22}^2 + \delta_{13} \times w_{32}^2 \\
\delta_{11} \times w_{12}^3 + \delta_{12} \times w_{22}^3 + \delta_{13} \times w_{32}^3
\end{pmatrix}
```

Next, we drop the first (unused) row, and apply the following Hadamard product:

```math
\left( {W^{(2)}}^T \cdot \delta \right)^* \odot
\begin{pmatrix}
\phi_{11}'(a_{11}) \\
\\
\phi_{21}'(a_{21}) \\
\\
\phi_{31}'(a_{31})
\end{pmatrix} =
\begin{pmatrix}
\delta_{11} \times w_{12}^1 + \delta_{12} \times w_{22}^1 + \delta_{13} \times w_{32}^1 \\
\delta_{11} \times w_{12}^2 + \delta_{12} \times w_{22}^2 + \delta_{13} \times w_{32}^2 \\
\delta_{11} \times w_{12}^3 + \delta_{12} \times w_{22}^3 + \delta_{13} \times w_{32}^3
\end{pmatrix} \odot
\begin{pmatrix}
\phi_{11}'(a_{11}) \\
\\
\phi_{21}'(a_{21}) \\
\\
\phi_{31}'(a_{31})
\end{pmatrix} =
\begin{pmatrix}
\delta_{11} \times w_{12}^1 \times \phi_{11}'(a_{11}) + \delta_{12} \times w_{22}^1 \times \phi_{11}'(a_{11}) + \delta_{13} \times w_{32}^1 \times \phi_{11}'(a_{11}) \\
\delta_{11} \times w_{12}^2 \times \phi_{21}'(a_{21}) + \delta_{12} \times w_{22}^2 \times \phi_{21}'(a_{21}) + \delta_{13} \times w_{32}^2 \times \phi_{21}'(a_{21}) \\
\delta_{11} \times w_{12}^3 \times \phi_{31}'(a_{31}) + \delta_{12} \times w_{22}^3 \times \phi_{31}'(a_{31}) + \delta_{13} \times w_{32}^3 \times \phi_{31}'(a_{31})
\end{pmatrix}
```

All that remains now to be done is match this against the equation for $\frac{\partial{L}}{\partial{w_{11}^k}}$,
follow our intuition, and multiply the above matrix with the one-row matrix corresponding
to the input layer, yielding $\nabla^{(1)}$, i.e., the partial derivatives of $L$
with respect to the weights on the $1^{st}$ layer:

```math
\nabla^{(1)} =
\begin{pmatrix}
\frac{\partial{L}}{\partial{w_{11}^0}} & \frac{\partial{L}}{\partial{w_{11}^1}} & \frac{\partial{L}}{\partial{w_{11}^2}} & \frac{\partial{L}}{\partial{w_{11}^3}} \\
\\
\frac{\partial{L}}{\partial{w_{21}^0}} & \frac{\partial{L}}{\partial{w_{21}^1}} & \frac{\partial{L}}{\partial{w_{21}^2}} & \frac{\partial{L}}{\partial{w_{21}^3}} \\
\\
\frac{\partial{L}}{\partial{w_{31}^0}} & \frac{\partial{L}}{\partial{w_{31}^1}} & \frac{\partial{L}}{\partial{w_{31}^2}} & \frac{\partial{L}}{\partial{w_{31}^3}}
\end{pmatrix} =
\left( {W^{(2)}}^T \cdot \delta \right)^* \odot
\begin{pmatrix}
\phi_{11}'(a_{11}) \\
\\
\phi_{21}'(a_{21}) \\
\\
\phi_{31}'(a_{31})
\end{pmatrix} \cdot
\begin{pmatrix}
1 & x_1 & x_2 & x_3
\end{pmatrix}
```

After each training epoch, we employ gradient descent to update the weight matrices
for a layer $j = \overline{1 \dots 3}$, given a learning rate $η$, thus:

```math
W^{(j)} := W^{(j)} - η\nabla^{(j)}
```
