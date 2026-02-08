package nnn
package real

import spire.math.*
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Real): Real = Real(1) / (Real(1) + exp(-x))

  inline def sqr[A: Ring](x: A): A = x.pow(2)

  inline def kronecker(i: Int)(j: Int): Double =
    if i == j then 1d else 0d

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

  object softmax:

    def apply[N <: Int](z: Vector[Real, N]): Vector[Real, N] =
      val x = z.op(exp(_))
      x.op(_ / x.sum)

    def prime[N <: Int](z: Vector[Real, N])(k: Int): Vector[Real, N] =
      val q = apply(z)
      val qk = q(k)
      q.op { (qi, i) => qk * (kronecker(i)(k) - qi) }
