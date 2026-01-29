package nnn
package real

import spire.math.Real
import spire.implicits.*

import Util.*


enum Loss[N <: Int: ValueOf](val apply: (Vector[Real, N], Vector[Real, N]) => Real,
                             val partial: (Vector[Real, N], Vector[Real, N]) => Int => Real):
  require(valueOf[N] > 0)

  /**
    * Mean squared error.
    */
  case MSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (Real(1) / valueOf[N]) * (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => (Real(2) / valueOf[N]) * (real(i) - ideal(i)) })

  /**
    * Sum of squares error.
    */
  case SSE[N <: Int: ValueOf]() extends Loss[N]({ (ideal, real) => (ideal + -real).op(sqr).sum },
                                                { (ideal, real) => i => 2 * (real(i) - ideal(i)) })
