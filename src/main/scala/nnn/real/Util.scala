package nnn
package real

import spire.math.*
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Real): Real = Real(1) / (Real(1) + exp(-x))

  inline def sqr[A: Ring](x: A): A = x.pow(2)

  inline def kronecker(i: Int)(j: Int): Real =
    if i == j then 1 else 0

  given [N: Numeric]: Conversion[N, Real] with
    def apply(self: N): Real = self match
      case it: Int    => Real(it)
      case it: Long   => Real(it)
      case it: Float  => Real(it)
      case it: Double => Real(it)
      case it: Real   => it

  inline given Conversion[Real, Int] = _.toInt
  inline given Conversion[Real, Long] = _.toLong
  inline given Conversion[Real, Float] = _.toFloat
  inline given Conversion[Real, Double] = _.toDouble

  object softmax:

    def apply[N <: Int](z: Vector[Real, N]): Vector[Real, N] =
      val x = z.op(exp(_))
      x.op(_ / x.sum)

    def prime[N <: Int](z: Vector[Real, N])(k: Int): Vector[Real, N] =
      val q = apply(z)
      val qk = q(k)
      q.op { (qi, i) => qk * (kronecker(i)(k) - qi) }

  object logsoftmax:

    def apply[N <: Int](x: Vector[Real, N]): Vector[Real, N] =
      val l = log(x.op(exp(_)).sum)
      x.op(_ - l)

    def prime[N <: Int](z: Vector[Real, N])(k: Int): Vector[Real, N] =
      val xk = softmax.apply(z)(k)
      z.op { (_, i) => kronecker(i)(k) - xk }
