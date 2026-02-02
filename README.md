N-neurons Neural Networks
=========================

Simple neural networks with constant number `N` of neurons per each layer. The
fixed `N` is actually part of the type of matrices and vectors.

The type `Matrix[A, M, N]` specifies that the matrix contains elements of type `A`,
has `M` number of rows and `N` number of columns. `A` must have a `Ring` typeclass
instance, as defined in [spire](https://spire-math.org).

The type `Vector[A, N]` specifies that the vectors contains elements of type `A`
and has `N` number of rows. `A` must have a `Ring` typeclass instance, as defined
in [spire](https://spire-math.org).

Each operation performed with matrices and vectors must type check, meaning one
can multiply a `Matrix[A, M, N]` and a `Matrix[A, N, P]`, yielding a `Matrix[A, M, P]`,
but one cannot multiply the former with a `Matrix[B, N, P]` or a `Matrix[A, P, Q]`,
because `A` differs from `B`, respectively, `N` differs from `P`.

N Neural Network
----------------

A neural network `Network[N, N1]` is composed of `L`-layers each of `N` neurons,
while `N1` corressponds to number `N` plus `1` (for the bias). For arbitrary
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

The package `float` builds around a generalized neural network, where
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

The higher-kinded type `N[_]` differs with the type argument: `N[0]` is the
number of inputs, `N[1]` is the number of neurons in the hidden layer, while
`N[2]` is the number of outputs. It is hence called a shape.

The implicit `given_List_Int` is the shape as values. Both shapes (types and
values) must be given. The neural network is then defined as `Network[N, 2](...)`,
where `N` is the shape and `2` is the number of hidden layers. Then `1` occuring
in the argument `loss = MSE[1]()` is the number of neurons in the output layer.

Note that the output layer is also considered a hidden layer.

Testing
-------

Use, for instance, the following `sbt` command:

    sbt:N Neural Networks> testOnly *double*Network*

This will run all tests with the word "`Network`" in package "`double`" (where all
values, functions, or networks are based on the `Double` type): there is only one
such suite, `nnn.double.NetworkSuite`.
