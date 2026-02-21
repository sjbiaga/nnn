package nnn
package cnn

import scala.compiletime.ops.int.{ +, -, /, * }

import scala.reflect.ClassTag

import spire.algebra.Ring
import spire.implicits.*

import Image.*


case class Image[
  A: Ring: ClassTag,
  L <: Int,
  B <: Int,
  D <: Int
](label: Any, volume: Volume[A, L, B, D]) { self =>

  val depth = volume.data.depth
  val length = volume.data.rows
  val breadth = volume.data.cols
  val size = length * breadth * depth

  inline def apply(k: Int): Matrix[A, L, B] =
    volume.data(k)

  /**
    * w/ stride
    */
  def apply[S <: Int: ValueOf] = PartiallyAppliedStrideOps[S]

  final class PartiallyAppliedStrideOps[S <: Int: ValueOf]:

    /**
      * cross-correlation w/ stride
      */
    def ⋆[P <: Int, Q <: Int](kernels: Kernel[A, P, Q, D]*)[D <: Int: ValueOf]: Image[A, (L-P)/S+1, (B-Q)/S+1, D] =
      Image[A, (L-P)/S+1, (B-Q)/S+1, D](null, Tensor[A][(L-P)/S+1, (B-Q)/S+1, D](kernels.map(that => (self.volume.data[S] ⋆ that.volume.data).reduce(_ + _).op(_ + that.bias))*))

  /**
    * flatten
    */
  def toSeq: Seq[A] =
    volume.data.toSeq.map(_.flatten).flatten

}


object Image:

  /**
    * alias for [[Tensor]] but subclasses cache extra data
    */
  case class Volume[
    A: Ring: ClassTag,
    L <: Int,
    B <: Int,
    D <: Int
  ](data: Tensor[A, L, B, D])

  class SubsamplingVolume[
    A: Ring: ClassTag,
    M <: Int,
    N <: Int,
    O <: Int,
    P <: Int,
    Q <: Int,
    S <: Int
  ](val avg: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O],
    data: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]
  ) extends Volume(data)

  class MaxPoolingVolume[
    A: Ring: ClassTag,
    M <: Int,
    N <: Int,
    O <: Int,
    P <: Int,
    Q <: Int,
    S <: Int
  ](val max: Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O],
    data: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]
  ) extends Volume(data)

  given [
    A: Ring: ClassTag,
    M <: Int,
    N <: Int,
    O <: Int
  ]: Conversion[Tensor[A, M, N, O], Volume[A, M, N, O]] = Volume.apply

  given [
    A: Ring: ClassTag,
    M <: Int,
    N <: Int,
    O <: Int
  ]: Conversion[Image[A, M, N, O], Tensor[A, M, N, O]] = _.volume.data

  given [
    A: Ring: ClassTag,
    M <: Int,
    N <: Int,
    O <: Int,
    P <: Int,
    Q <: Int,
    S <: Int
  ]: Conversion[(Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O], Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]), Volume[A, (M-P)/S+1, (N-Q)/S+1, O]] with
    def apply(self: (Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O], Tensor[A, (M-P)/S+1, (N-Q)/S+1, O])): Volume[A, (M-P)/S+1, (N-Q)/S+1, O] =
      MaxPoolingVolume[A, M, N, O, P, Q, S](self._1, self._2)

  given [
    A: Ring: ClassTag,
    M <: Int,
    N <: Int,
    O <: Int,
    P <: Int,
    Q <: Int,
    S <: Int
  ](using avg: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]): Conversion[Tensor[A, (M-P)/S+1, (N-Q)/S+1, O], Volume[A, (M-P)/S+1, (N-Q)/S+1, O]] = SubsamplingVolume[A, M, N, O, P, Q, S](avg, _)

  type FeatureMap = Image

  class Kernel[
    A: Ring: ClassTag,
    L <: Int,
    B <: Int,
    D <: Int: ValueOf
  ](var bias: A, volume: Volume[A, L, B, D])
      extends Image[A, L, B, D](null, volume)

  object Kernel:

    def apply[
      A: Ring: ClassTag,
      L <: Int: ValueOf,
      B <: Int: ValueOf,
      D <: Int: ValueOf
    ](initialization: Initialization)[O <: Int: ValueOf](using Conversion[Double, A]): Seq[Kernel[A, L, B, D]] =
      (1 to valueOf[O]).map(_ => new Kernel[A, L, B, D](initialization[A](), Tensor[A][L, B, D](initialization)))
