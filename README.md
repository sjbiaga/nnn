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

Neural Network
--------------

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

Testing
-------

Use, for instance, the following `sbt` command:

    sbt:N Neural Networks> testOnly *double*Network*

This will run all tests with the word "`Network`" in package "`double`" (where all
values, functions, or networks are based on the `Double` type): there is only one
such suite, `nnn.double.NetworkSuite`.
