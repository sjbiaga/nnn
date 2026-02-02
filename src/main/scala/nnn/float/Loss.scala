package nnn
package float

import spire.implicits.*

import Util.*


enum Loss[N <: Int: ValueOf](val apply: (Vector[Float, N], Vector[Float, N]) => Float,
                             val partial: (Vector[Float, N], Vector[Float, N]) => Int => Float):
  require(valueOf[N] > 0)

  /**
    * Mean squared error.
    */
  case MSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (1f / valueOf[N]) * (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => (2f / valueOf[N]) * (real(i) - ideal(i)) })

  /**
    * Sum of squares error.
    */
  case SSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => 2 * (real(i) - ideal(i)) })
