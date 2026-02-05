package nnn

import breeze.stats.distributions.Rand.VariableSeed.randBasis

import spire.algebra.Ring
import spire.implicits.*


enum Initialization:

  case Gaussian(mean: Double = 0, stddev: Double = 1)

  case Xavier(inputs: Int, outputs: Int, normal: Boolean = false)

  case Kaiming(inputs: Int)

  def apply[A: Ring]()(using c: Conversion[Double, A]): A =
    c {
      this match
        case Gaussian(mean, stddev) =>
          breeze.stats.distributions.Gaussian(mean, stddev).sample()
        case Xavier(inputs, outputs, true) =>
          val x = sqrt(6d / (inputs + outputs))
          breeze.stats.distributions.Uniform(-x, x).sample()
        case Xavier(inputs, outputs, _) =>
          breeze.stats.distributions.Gaussian(0, sqrt(2d / (inputs + outputs))).sample()
        case Kaiming(inputs) =>
          breeze.stats.distributions.Gaussian(0, sqrt(2d / inputs)).sample()
    }
