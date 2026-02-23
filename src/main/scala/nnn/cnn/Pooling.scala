package nnn
package cnn

import scala.reflect.ClassTag

import scala.compiletime.ops.int.{ +, -, /, * }

import spire.algebra.{ Field, Ring }
import spire.implicits.*

import Image.*


enum Pooling[
  A: Ring: ClassTag,
  P <: Int: ValueOf,
  Q <: Int: ValueOf,
  S <: Int: ValueOf
]:

  val rows = valueOf[P]
  val cols = valueOf[Q]
  val size = rows * cols
  val stride = valueOf[S]

  case avg[
    A: Ring: ClassTag,
    P <: Int: ValueOf,
    Q <: Int: ValueOf,
    S <: Int: ValueOf]() extends Pooling[A, P, Q, S]

  case max[
    A: Ring: ClassTag,
    P <: Int: ValueOf,
    Q <: Int: ValueOf,
    S <: Int: ValueOf](min: A) extends Pooling[A, P, Q, S]

  case subsampling[
    A: Ring: ClassTag,
    P <: Int: ValueOf,
    Q <: Int: ValueOf,
    S <: Int: ValueOf](beta: Vector[A, ?], bias: Vector[A, ?]) extends Pooling[A, P, Q, S]

  def apply[
    M <: Int,
    N <: Int,
    O <: Int: ValueOf
  ](fm: FeatureMap[A, M, N, O]
  )(using Field[A], Ordering[A]
  ): FeatureMap[A, (M-P)/S+1, (N-Q)/S+1, O] =
    this match
      case avg() =>
        Image[A, (M-P)/S+1, (N-Q)/S+1, O](null, fm.volume.data[S].avg[P, Q])
      case max(min) =>
        Image[A, (M-P)/S+1, (N-Q)/S+1, O](null, fm.volume.data[S].max[P, Q](min))
      case subsampling(β: Vector[A, O], b: Vector[A, O]) =>
        given avg: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O] = fm.volume.data[S].avg[P, Q]
        Image[A, (M-P)/S+1, (N-Q)/S+1, O](null, avg.op { (it, ijk) => it * β(ijk._3) + b(ijk._3) })

  /**
    * gradient distribution
    *
    * @param h input feature map (forward pass)
    * @param a output feature map (forward pass)
    * @param δ input gradient (backpropagation)
    * @return output gradient (backpropagation)
    */
  def apply[
    M <: Int,
    N <: Int,
    O <: Int
  ](h: FeatureMap[A, M, N, O],
    a: FeatureMap[A, (M-P)/S+1, (N-Q)/S+1, O],
    δ: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]
  ): Tensor[A, M, N, O] =
    val result = Array.fill(h.length, h.breadth, h.depth)(Ring[A].zero)
    for
      d <- 0 until h.depth
      i <- 0 until h.length-rows+1 by stride
      j <- 0 until h.breadth-cols+1 by stride
      iʹ = i/stride
      jʹ = j/stride
    do
      this match
        case avg() | subsampling(_, _) =>
          for
            k <- 0 until rows
            l <- 0 until cols
          do
            result(i+k)(j+l)(d) += δ(iʹ, jʹ, d)
        case max(_) =>
          val max = a.volume.asInstanceOf[MaxPoolingVolume[A, (M-P)/S+1, (N-Q)/S+1, O, P, Q, S]].max
          val (k, l) = max(iʹ, jʹ, d)
          result(k)(l)(d) += δ(iʹ, jʹ, d)
    Image[A, M, N, O](null, Tensor[A, M, N, O](result))


object Pooling:

  object subsampling:

    def apply[
      A: Ring: ClassTag,
      P <: Int: ValueOf,
      Q <: Int: ValueOf,
      S <: Int: ValueOf,
      D <: Int: ValueOf
    ](initialization: Initialization)(using Conversion[Double, A]): subsampling[A, P, Q, S] =
      val β = if initialization eq null then Vector[A].one[D]
              else Vector[A][D](_ => initialization())
      val b = Vector[A].zero[D]
      Pooling.subsampling[A, P, Q, S](β, b)
