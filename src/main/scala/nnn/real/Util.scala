package nnn
package real

import spire.math.*
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Real): Real = Real(1) / (Real(1) + exp(-x))

  inline def sqr[A: Ring](x: A): A = x.pow(2)

  given [N: Numeric]: Conversion[N, Real] with
    def apply(self: N): Real = self match
      case it: Int    => Real(it)
      case it: Long   => Real(it)
      case it: Float  => Real(it)
      case it: Double => Real(it)
      case it: Real   => it

  given Conversion[Real, Int] = _.toInt
  given Conversion[Real, Long] = _.toLong
  given Conversion[Real, Float] = _.toFloat
  given Conversion[Real, Double] = _.toDouble
