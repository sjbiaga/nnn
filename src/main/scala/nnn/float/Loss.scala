package nnn
package float

import spire.implicits.*

import Util.*


enum Loss[N <: Int: ValueOf](val apply: (Vector[Float, N], Vector[Float, N]) => Float,
                             val partial: (Vector[Float, N], Vector[Float, N]) => Int => Float):
  require(valueOf[N] > 0)

  /**
    * Binary Cross Entropy
    */
  case BCE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) =>
                                                  -(1f / valueOf[N]) * (ideal.underlying zip real.underlying).map { (y, ŷ) =>
                                                    y * log(ŷ) + (1f - y) * log(1f - ŷ)
                                                  }.sum
                                                },
                                                { (ideal, real) => i => -(1f / valueOf[N]) * (ideal(i) / real(i) - (1f - ideal(i)) / (1f - real(i))) })

  /**
    * Mean Squared Error
    */
  case MSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (1f / valueOf[N]) * (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => (2f / valueOf[N]) * (real(i) - ideal(i)) })

  /**
    * Sum of Squares Error
    */
  case SSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => 2 * (real(i) - ideal(i)) })
