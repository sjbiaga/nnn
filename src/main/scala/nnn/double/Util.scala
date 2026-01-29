package nnn
package double

import spire.math.*
import spire.algebra.Ring
import spire.implicits.*


object Util:

  def sigmoid(x: Double) = 1d / (1d + exp(-x))

  inline def sqr[A: Ring](x: A) = x.pow(2)

  given [N: Numeric]: Conversion[N, Double] with
    def apply(self: N): Double = self match
      case it: Int    => it.toDouble
      case it: Long   => it.toDouble
      case it: Float  => it.toDouble
      case it: Double => it

  given Conversion[Double, Int] = _.toInt
  given Conversion[Double, Long] = _.toLong
  given Conversion[Double, Float] = _.toFloat

  extension [N <: Int: ValueOf](self: Vector[Double, N])
    def truncate(n: Int): Vector[Double, N] =
      self.op { it => (it * 10.pow(n)).toInt.toDouble / 10.pow(n) }
