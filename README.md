`N`-neurons Neural Networks
===========================

Simple neural networks with constant number `N` of neurons per each layer (when the
fixed `N` is actually part of the type of matrices and vectors) or even differing
between layers.

The type `Matrix[A, M, N]` specifies that the matrix contains elements of type `A`,
has `M` number of rows and `N` number of columns. `A` must have a `Ring` typeclass
instance, as defined in [spire](https://spire-math.org).

The type `Vector[A, N]` specifies that the vectors contains elements of type `A`
and has `N` number of rows. `A` must have a `Ring` typeclass instance, as defined
in [spire](https://spire-math.org).

The type `Tensor[A, M, N, O]` specifies that the `3D` tensor contains elements of
type `A`, has `M` number of rows, `N` number of columns and `O` as depth. `A` must
have a `Ring` typeclass instance, as defined in [spire](https://spire-math.org).

Each operation performed with tensors, matrices, or vectors must type check, meaning one
can multiply a `Matrix[A, M, N]` and a `Matrix[A, N, P]`, yielding a `Matrix[A, M, P]`,
but one cannot multiply the former with a `Matrix[B, N, P]` or a `Matrix[A, P, Q]`,
because `A` differs from `B`, respectively, `N` differs from `P`.

`N` Neural Network
------------------

A neural network `Network[N]` is composed of `M` layers each of `N` neurons, while
`N+1` is the number of weights per neuron plus `1` (for the bias). For arbitrary
precision arithmetic using `spire.math.Real`, for instance, each neuron of type
`Neuron[N]` has a `Vector[Real, N]` of weights, a `Real` bias and an `Activation`
function, where the latter is a Scala3 `enum`. This means that the `Activation`
functions may differ with each neuron. The definition and the derivative of an
`Activation` function must be known and given.

For a neural network, it is implemented training with backpropagation, as well
as the straighter prediction, once the neurons' (biases and) weights have been
trained.

For examples, see also [this blog](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example)
or [this `java-toy-neural-network` project](https://github.com/lexesj/java-toy-neural-network).

Neural Network
--------------

The package `nnn.float` builds around a generalized neural network, where
the assumption of a constant number of neurons per each layer is relaxed, and
thus the number of neurons may differ between layers.

Although the dimension types of the variables and values involved in the algorithms
are the wildcard `?`, operations on matrices and vectors are type-safe using _shapes_.
Even assignment is safe - though not the method `:=` by itself - because the algorithms
perform reassignment only, and thus the types of matrices/vectors are asserted before
assignment.

An example of a neural network with two inputs, a hidden layer with ten neurons,
and one output is the following:

```Scala
type N[L <: Int] = L match { case 0 => 2 case 1 => 10 case 2 => 1 }

given List[Int] = 2 :: 10 :: 1 :: Nil

Network[N, 2](loss = MSE[1](), learningRate = 3, ...)
```

Each type mapped by the higher-kinded type `N[_]` differs with the type argument:
`N[0]` is the number of inputs, `N[1]` is the number of neurons in the hidden layer,
while `N[2]` is the number of outputs. It is hence called a shape.

The implicit `given_List_Int` is the shape as values. Both kinds of shape (types
and values) must be given. The neural network is defined as `Network[N, 2](...)`,
where `N` is the shape and `2` is the number of hidden layers, while the (values)
shape is passed as implicit parameter. Then, `1` occuring in the argument
`loss = MSE[1]()` is the same number of neurons in the output layer.

Note that the output layer is also a hidden layer.

Convolutional Neural Network
----------------------------

The package `nnn.cnn.float` builds around a generalized _convolutional_ neural network.
A modified version of `LeNet` with `Kaiming` initialization, `LeakyReLU` activation,
and `Max Pooling` (instead of `Xavier/Gorot` initialization, `Sigmoid` activation, and
`Subsampling`) tests the implementation of `CNN`s. The `MNIST` databases are used and
the four `.gz` files must be placed under the root-relative "`./data/MNIST`" folder:

    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz

According to [this](https://stackoverflow.com/questions/66577151/http-error-when-trying-to-download-mnist-data),
the following URLs can be used for download:

    https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

It takes about 16 hours (on an Intel i5 CPU at 2.90GHz) to train the entire shuffled
60000 images, in 100 batches and 20 epochs, then testing the entire shuffled 10000
test images with an accuracy of 98%.

An experimental reduced version of `AlexNet` with 10 classes and 512 neurons in
the dense layers (instead of 1000 and 4096), and without the original optimizations,
is also included for purpose of demonstration, but disabled because of being slow.
The `CIFAR-10` database must be downloaded:

    https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

and unpacked in the root-relative "`./data`" folder:

	batches.meta.txt
	data_batch_1.bin
	data_batch_2.bin
	data_batch_3.bin
	data_batch_4.bin
	data_batch_5.bin
	readme.html
	test_batch.bin

The `RGB` images are `32x32` and scaled to `227x227`.

Testing
-------

Use, for instance, the following `sbt` command:

    sbt:N Neural Networks> testOnly *double*Network*

This will run all tests with the word "`Network`" in package "`nnn.double`" (where all
values, functions, or networks are based on the `Double` type): there is only one
such suite, `nnn.double.NetworkSuite`.

Math
----

Consider a neural network with three hidden layers (the last of each is the output
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

These were the equations corresponding to the forward pass. For backpropagation,
we proceed backwards, from the output layer towards the input layer. Assume $L$
is the loss function; it has three known partial derivatives:
$\frac{\partial{L}}{\partial{y_i}}$, where $i = \overline{1 \dots 3}$.

From these, we start with the derivatives of $L$ with respect to the weights in
the last (output) layer ($w_{i3}^k$, where $i = \overline{1 \dots 3}$ and
$k = \overline{0 \dots 3}$). Using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule)
the following equations hold:

```math
\begin{align}
\frac{\partial{L}}{\partial{w_{13}^0}} = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times 1 & & \frac{\partial{L}}{\partial{w_{13}^1}} = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times h_{12} & & \frac{\partial{L}}{\partial{w_{13}^2}} = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times h_{22} & & \frac{\partial{L}}{\partial{w_{13}^3}} = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times h_{32} & (1)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{23}^0}} = \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times 1 & & \frac{\partial{L}}{\partial{w_{23}^1}} = \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times h_{12} & & \frac{\partial{L}}{\partial{w_{23}^2}} = \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times h_{22} & & \frac{\partial{L}}{\partial{w_{23}^3}} = \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times h_{32} & (2)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{33}^0}} = \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times 1 & & \frac{\partial{L}}{\partial{w_{33}^1}} = \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times h_{12} & & \frac{\partial{L}}{\partial{w_{33}^2}} = \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times h_{22} & & \frac{\partial{L}}{\partial{w_{33}^3}} = \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times h_{32} & (3)
\end{align}
```

We note that $\frac{\partial{L}}{\partial{y_i}} \times \phi_{i3}'(a_{i3})$,
where $i = \overline{1 \dots 3}$, occur repeatedly: we may thus introduce
the following matrix named $\delta$ (using the Hadamard product $\odot$):

```math
\delta =
\begin{pmatrix}
\frac{\partial{L}}{\partial{y_1}} \\
\\
\frac{\partial{L}}{\partial{y_2}} \\
\\
\frac{\partial{L}}{\partial{y_3}}
\end{pmatrix} \odot
\begin{pmatrix}
\phi_{13}'(a_{13}) \\
\\
\phi_{23}'(a_{23}) \\
\\
\phi_{33}'(a_{33})
\end{pmatrix}
```

Then, the previous equations become (under the notation $\nabla^{(3)}$ - the
partial derivatives of $L$ with respect to the weights on the $3^{rd}$ layer):

```math
\nabla^{(3)} =
\begin{pmatrix}
\frac{\partial{L}}{\partial{w_{13}^0}} & \frac{\partial{L}}{\partial{w_{13}^1}} & \frac{\partial{L}}{\partial{w_{13}^2}} & \frac{\partial{L}}{\partial{w_{13}^3}} \\
\\
\frac{\partial{L}}{\partial{w_{23}^0}} & \frac{\partial{L}}{\partial{w_{23}^1}} & \frac{\partial{L}}{\partial{w_{23}^2}} & \frac{\partial{L}}{\partial{w_{23}^3}} \\
\\
\frac{\partial{L}}{\partial{w_{33}^0}} & \frac{\partial{L}}{\partial{w_{33}^1}} & \frac{\partial{L}}{\partial{w_{33}^2}} & \frac{\partial{L}}{\partial{w_{33}^3}}
\end{pmatrix} =
\delta \cdot
\begin{pmatrix}
1 & h_{12} & h_{22} & h_{32}
\end{pmatrix}
```

Let us now write the twelve partial derivatives of $L$ with respect to the weights
on the $2^{nd}$ layer, each equation being a sum of three terms:

```math
\begin{align}
\frac{\partial{L}}{\partial{w_{12}^0}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a12) \times 1 & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a12) \times 1 & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a12) \times 1 & (4) \\
\frac{\partial{L}}{\partial{w_{12}^1}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a12) \times h_{11} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a12) \times h_{11} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a12) \times h_{11} & (5)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{12}^2}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a12) \times h_{21} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a12) \times h_{21} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a12) \times h_{21} & (6) \\
\frac{\partial{L}}{\partial{w_{12}^3}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a12) \times h_{31} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a12) \times h_{31} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a12) \times h_{31} & (7)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{22}^0}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a22) \times 1 & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a22) \times 1 & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a22) \times 1 & (8) \\
\frac{\partial{L}}{\partial{w_{22}^1}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a22) \times h_{11} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a22) \times h_{11} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a22) \times h_{11} & (9)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{22}^2}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a22) \times h_{21} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a22) \times h_{21} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a22) \times h_{21} & (10) \\
\frac{\partial{L}}{\partial{w_{22}^3}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a22) \times h_{31} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a22) \times h_{31} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a22) \times h_{31} & (11)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{32}^0}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a32) \times 1 & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a32) \times 1 & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a32) \times 1 & (12) \\
\frac{\partial{L}}{\partial{w_{32}^1}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a32) \times h_{11} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a32) \times h_{11} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a32) \times h_{11} & (13)
\end{align}
```
```math
\begin{align}
\frac{\partial{L}}{\partial{w_{32}^2}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a32) \times h_{21} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a32) \times h_{21} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a32) \times h_{21} & (14) \\
\frac{\partial{L}}{\partial{w_{32}^3}} & = \frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a32) \times h_{31} & + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a32) \times h_{31} & + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a32) \times h_{31} & (15)
\end{align}
```

Let us now observe what is the product of the transpose of the $3^{rd}$ layer's
weights matrix, and delta:

```math
{W^{(3)}}^T \cdot \delta =
\begin{pmatrix}
w_{13}^0 & w_{23}^0 & w_{33}^0 \\
\\
w_{13}^1 & w_{23}^1 & w_{33}^1 \\
\\
w_{13}^2 & w_{23}^2 & w_{33}^2 \\
\\
w_{13}^3 & w_{23}^3 & w_{33}^3 \\
\end{pmatrix} \cdot
\begin{pmatrix}
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \\
\\
\frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \\
\\
\frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33})
\end{pmatrix} =
```
```math
=
\begin{pmatrix}
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^0 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^0 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^0 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \\
\end{pmatrix}
```

Let us further drop the first row (because it is not used) in the above matrix
(denoting this transient matrix by ${\left( {W^{(3)}}^T \cdot \delta \right)}^*$):

```math
{\left( {W^{(3)}}^T \cdot \delta \right)}^* =
\begin{pmatrix}
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \\
\end{pmatrix}
```

and apply the following Hadamard product:

```math
{\left( {W^{(3)}}^T \cdot \delta \right)}^* \odot
\begin{pmatrix}
\phi_{12}'(a_{12}) \\
\\
\phi_{22}'(a_{22}) \\
\\
\phi_{32}'(a_{32})
\end{pmatrix} =
\begin{pmatrix}
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \\
\end{pmatrix} \odot
\begin{pmatrix}
\phi_{12}'(a_{12}) \\
\\
\phi_{22}'(a_{22}) \\
\\
\phi_{32}'(a_{32})
\end{pmatrix}
```

We obtain the following result, again reassigned to $\delta$:

```math
\delta =
\begin{pmatrix}
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^1 \times \phi_{12}'(a_{12}) + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^1 \times \phi_{12}'(a_{12}) + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^1 \times \phi_{12}'(a_{12}) \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^2 \times \phi_{22}'(a_{22}) + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^2 \times \phi_{22}'(a_{22}) + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^2 \times \phi_{22}'(a_{22}) \\
\\
\frac{\partial{L}}{\partial{y_1}} \times \phi_{13}'(a_{13}) \times w_{13}^3 \times \phi_{32}'(a_{32}) + \frac{\partial{L}}{\partial{y_2}} \times \phi_{23}'(a_{23}) \times w_{23}^3 \times \phi_{32}'(a_{32}) + \frac{\partial{L}}{\partial{y_3}} \times \phi_{33}'(a_{33}) \times w_{33}^3 \times \phi_{32}'(a_{32}) \\
\end{pmatrix}
```

The first row of this matrix corresponds to equations $(4)-(7)$, the second row to
equations $(8)-(11)$, and the third row to equations $(12)-(15)$.

Then, the equations $(4)-(15)$ become (under the notation $\nabla^{(2)}$ - the
partial derivatives of $L$ with respect to the weights on the $2^{nd}$ layer):

```math
\nabla^{(2)} =
\begin{pmatrix}
\frac{\partial{L}}{\partial{w_{12}^0}} & \frac{\partial{L}}{\partial{w_{12}^1}} & \frac{\partial{L}}{\partial{w_{12}^2}} & \frac{\partial{L}}{\partial{w_{12}^3}} \\
\\
\frac{\partial{L}}{\partial{w_{22}^0}} & \frac{\partial{L}}{\partial{w_{22}^1}} & \frac{\partial{L}}{\partial{w_{22}^2}} & \frac{\partial{L}}{\partial{w_{22}^3}} \\
\\
\frac{\partial{L}}{\partial{w_{32}^0}} & \frac{\partial{L}}{\partial{w_{32}^1}} & \frac{\partial{L}}{\partial{w_{32}^2}} & \frac{\partial{L}}{\partial{w_{32}^3}}
\end{pmatrix} =
\delta \cdot
\begin{pmatrix}
1 & h_{11} & h_{21} & h_{31}
\end{pmatrix}
```

### Math ([cont'd](https://github.com/sjbiaga/nnn/blob/main/README1.md))
