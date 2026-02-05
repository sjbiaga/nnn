package nnn
package double

import spire.math.log
import spire.implicits.*

import Util.*


enum Loss[N <: Int: ValueOf](val apply: (Vector[Double, N], Vector[Double, N]) => Double,
                             val partial: (Vector[Double, N], Vector[Double, N]) => Int => Double):
  require(valueOf[N] > 0)

  /**
    * Binary Cross Entropy
    */
  case BCE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) =>
                                                  -(1d / valueOf[N]) * (ideal.underlying zip real.underlying).map { (y, ŷ) =>
                                                    y * log(ŷ) + (1d - y) * log(1d - ŷ)
                                                  }.sum
                                                },
                                                { (ideal, real) => i => -(1d / valueOf[N]) * (ideal(i) / real(i) - (1d - ideal(i)) / (1d - real(i))) })

  /**
    * Mean Squared Error
    */
  case MSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (1d / valueOf[N]) * (ideal + real.op(-_)).op(sqr).sum },
                                                { (ideal, real) => i => (2d / valueOf[N]) * (real(i) - ideal(i)) })

  /**
    * Sum of Squares Error
    */
  case SSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (ideal + real.op(-_)).op(sqr).sum },
                                                { (ideal, real) => i => 2 * (real(i) - ideal(i)) })
