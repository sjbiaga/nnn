package nnn
package double

import spire.implicits.*

import Util.*


enum Loss[N <: Int: ValueOf](val apply: (Vector[Double, N], Vector[Double, N]) => Double,
                             val partial: (Vector[Double, N], Vector[Double, N]) => Int => Double):
  require(valueOf[N] > 0)

  /**
    * Mean squared error.
    */
  case MSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (1d / valueOf[N]) * (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => (2d / valueOf[N]) * (real(i) - ideal(i)) })

  /**
    * Sum of squares error.
    */
  case SSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => 2 * (real(i) - ideal(i)) })
