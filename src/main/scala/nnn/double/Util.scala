package nnn
package double

import spire.math.*
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Double): Double = 1d / (1d + exp(-x))

  inline def sqr[A: Ring](x: A): A = x.pow(2)

  inline def kronecker(i: Int)(j: Int): Double =
    if i == j then 1d else 0d

  given [N: Numeric]: Conversion[N, Double] with
    def apply(self: N): Double = self match
      case it: Int    => it.toDouble
      case it: Long   => it.toDouble
      case it: Float  => it.toDouble
      case it: Double => it

  inline given Conversion[Double, Int] = _.toInt
  inline given Conversion[Double, Long] = _.toLong
  inline given Conversion[Double, Float] = _.toFloat

  extension [N <: Int: ValueOf](self: Vector[Double, N])
    def truncate(n: Int): Vector[Double, N] =
      self.op { it => (it * 10d.pow(n)).toInt.toDouble / 10d.pow(n) }

  object softmax:

    def apply[N <: Int](z: Vector[Double, N]): Vector[Double, N] =
      val x = z.op(exp(_))
      x.op(_ / x.sum)

    def prime[N <: Int](z: Vector[Double, N])(k: Int): Vector[Double, N] =
      val q = apply(z)
      val qk = q(k)
      q.op { (qi, i) => qk * (kronecker(i)(k) - qi) }

  object logsoftmax:

    def apply[N <: Int](x: Vector[Double, N]): Vector[Double, N] =
      val l = log(x.op(exp(_)).sum)
      x.op(_ - l)

    def prime[N <: Int](z: Vector[Double, N])(k: Int): Vector[Double, N] =
      val xk = softmax.apply(z)(k)
      z.op { (_, i) => kronecker(i)(k) - xk }
