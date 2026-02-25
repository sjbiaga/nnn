package nnn
package float

import breeze.stats.distributions.Rand.VariableSeed.randBasis

import spire.math.{ exp => _, log => _, * }
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Float): Float = 1f / (1f + exp(-x))

  inline def exp(x: Float): Float = spire.math.exp[Float](x)

  inline def log(x: Float): Float = spire.math.log[Float](x)

  inline def sqr[A: Ring](x: A): A = x.pow(2)

  inline def kronecker(i: Int)(j: Int): Float =
    if i == j then 1f else 0f

  def dropout[N <: Int: ValueOf](keep: Float, inverse: Boolean = true): Vector[Float, N] =
    val b = breeze.stats.distributions.Bernoulli(keep)
    Vector[Float](b.sample(valueOf[N]).map(if _ then (if inverse then 1f/keep else 1f) else 0f)*)

  given [N: Numeric]: Conversion[N, Float] with
    def apply(self: N): Float = self match
      case it: Int    => it.toFloat
      case it: Long   => it.toFloat
      case it: Double => it.toFloat
      case it: Float  => it

  given Conversion[Float, Int] = _.toInt
  given Conversion[Float, Long] = _.toLong
  given Conversion[Float, Double] = _.toDouble

  extension [N <: Int: ValueOf](self: Vector[Float, N])
    def truncate(n: Int): Vector[Float, N] =
      self.op { it => (it * 10f.pow(n)).toInt.toFloat / 10f.pow(n) }

  object softmax:

    def apply[N <: Int](z: Vector[Float, N]): Vector[Float, N] =
      val x = z.op(exp(_))
      x.op(_ / x.sum)

    def prime[N <: Int](z: Vector[Float, N])(k: Int): Vector[Float, N] =
      val q = apply(z)
      val qk = q(k)
      q.op { (qi, i) => qk * (kronecker(i)(k) - qi) }

  object logsoftmax:

    def apply[N <: Int](x: Vector[Float, N]): Vector[Float, N] =
      val l = log(x.op(exp(_)).sum)
      x.op(_ - l)

    def prime[N <: Int](z: Vector[Float, N])(k: Int): Vector[Float, N] =
      val xk = softmax.apply(z)(k)
      z.op { (_, i) => kronecker(i)(k) - xk }
