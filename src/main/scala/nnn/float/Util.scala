package nnn
package float

import spire.math.*
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Float): Float = 1f / (1f + exp(-x))

  inline def exp(x: Float) = spire.math.exp[Float](x)

  inline def sqr[A: Ring](x: A) = x.pow(2)

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
