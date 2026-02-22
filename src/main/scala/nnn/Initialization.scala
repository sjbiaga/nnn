package nnn

import breeze.stats.distributions.Rand.VariableSeed.randBasis

import spire.algebra.Ring
import spire.implicits.*


enum Initialization:

  case Constant(value: Double)

  case Gaussian(mean: Double = 0, stddev: Double = 1)

  case Glorot(inputs: Int, outputs: Int)

  case Xavier(inputs: Int, outputs: Int)

  case Kaiming(inputs: Int, α: Double = 0.01)

  def apply[A: Ring]()(using c: Conversion[Double, A]): A =
    c {
      this match
        case Constant(value) => value
        case Gaussian(mean, stddev) =>
          breeze.stats.distributions.Gaussian(mean, stddev).sample()
        case Glorot(inputs, outputs) =>
          val x = sqrt(6d / (inputs + outputs))
          breeze.stats.distributions.Uniform(-x, x).sample()
        case Xavier(inputs, outputs) =>
          breeze.stats.distributions.Gaussian(0, sqrt(2d / (inputs + outputs))).sample()
        case Kaiming(inputs, α) =>
          breeze.stats.distributions.Gaussian(0, sqrt(2 / ((1 + α*α) * inputs))).sample()
    }
