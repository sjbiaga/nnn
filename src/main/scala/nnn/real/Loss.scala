package nnn
package real

import spire.math.{ log, Real }
import spire.compat.numeric

import Util.*


enum Loss[N <: Int: ValueOf](val apply: (Vector[Real, N], Vector[Real, N]) => Real,
                             val partial: (Vector[Real, N], Vector[Real, N]) => Int => Real):
  require(valueOf[N] > 0)

  /**
    * Binary Cross Entropy
    */
  case BCE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) =>
                                                  -(Real(1) / valueOf[N]) * (ideal.underlying zip real.underlying).map { (y, ŷ) =>
                                                    y * log(ŷ) + (Real(1) - y) * log(Real(1) - ŷ)
                                                  }.sum
                                                },
                                                { (ideal, real) => i => -(Real(1) / valueOf[N]) * (ideal(i) / real(i) - (Real(1) - ideal(i)) / (Real(1) - real(i))) })

  /**
    * Mean Squared Error
    */
  case MSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (Real(1) / valueOf[N]) * (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => (Real(2) / valueOf[N]) * (real(i) - ideal(i)) })

  /**
    * Sum of Squares Error
    */
  case SSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => 2 * (real(i) - ideal(i)) })
