package nnn
package cnn

import scala.compiletime.ops.int.{ +, -, /, * }

import scala.reflect.ClassTag

import spire.math.pow
import spire.algebra.{ Field, Ring }
import spire.implicits.{ pow => _, * }

import Image.*
import Volume.{ *, given }


case class Image[
  A: Ring: ClassTag,
  L <: Int,
  B <: Int,
  D <: Int
](label: Any, volume: Volume[A, L, B, D]) { self =>

  def shaped[P <: Int] = PartiallyAppliedShapedPadded[P]
  final class PartiallyAppliedShapedPadded[P <: Int]:
    def apply[M[_ <: Int] <: Int,
              N[_ <: Int] <: Int,
              O[_ <: Int] <: Int](using l: Int): Tensor[A, M[l.type]+2*P, N[l.type]+2*P, O[l.type]] =
      self.asInstanceOf[Image[A, M[l.type]+2*P, N[l.type]+2*P, O[l.type]]]
  def shaped[M[_ <: Int] <: Int,
             N[_ <: Int] <: Int,
             O[_ <: Int] <: Int](using l: Int): Image[A, M[l.type], N[l.type], O[l.type]] =
    this.asInstanceOf[Image[A, M[l.type], N[l.type], O[l.type]]]
  def shaped[M <: Int, N <: Int, O <: Int]: Image[A, M, N, O] =
    this.asInstanceOf[Image[A, M, N, O]]

  val length = volume.data.rows
  val breadth = volume.data.cols
  val depth = volume.data.depth
  val size = length * breadth * depth

  val shape = volume.data.shape

  inline def apply(i: Int, j: Int, k: Int): A =
    volume.data(i, j, k)

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
    * unsafe padding
    */
  def pad(padding: Int): Image[A, L+2*padding.type, B+2*padding.type, D] =
    if padding == 0
    then
      this.asInstanceOf[Image[A, L+2*padding.type, B+2*padding.type, D]]
    else
      type P = padding.type
      given ValueOf[P] = ValueOf(padding)
      Image[A, L+2*P, B+2*P, D](label, volume.data.pad[P, P])

  /**
    * local response normalization
    */
  def lrn(k: A, n: Int, α: A, β: A)(sum: Boolean = true)
         (using Conversion[A, Double], Conversion[Double, A])
         (using Field[A], Numeric[A]): (Tensor[A, L, B, D], Image[A, L, B, D]) =
                                    //  ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^
                                    //  local sum           normalized values
    val N = depth
    val n_2 = n/2
    val ps = {
      for
        i <- 0 until length
        j <- 0 until breadth
      yield
        for
          c <- 0 until N
          s = 0 max (c-n_2)
          e = (N-1) min (c+n_2)
        yield
          val S = k + α * {
            for
              h <- s to e
              it = this(i, j, h)
            yield
              it*it
          }.sum
          (S: A, (this(i, j, c) / pow(S, β)): A)
    }.flatten
    given ValueOf[L] = ValueOf(length.asInstanceOf[L])
    given ValueOf[B] = ValueOf(breadth.asInstanceOf[B])
    given ValueOf[D] = ValueOf(depth.asInstanceOf[D])
    (if sum then Tensor[A][L, B, D](ps.map(_._1)*) else null) -> Image[A, L, B, D](label, Tensor[A][L, B, D](ps.map(_._2)*))

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
  sealed abstract class Volume[
    A: Ring: ClassTag,
    L <: Int,
    B <: Int,
    D <: Int
  ]:
    val data: Tensor[A, L, B, D]

  object Volume:

    def apply[
      A: Ring: ClassTag,
      L <: Int,
      B <: Int,
      D <: Int
    ](tensor: Tensor[A, L, B, D]) =
      new Volume[A, L, B, D]:
        override val data: Tensor[A, L, B, D] = tensor

    case class SubsamplingVolume[
      A: Ring: ClassTag,
      M <: Int,
      N <: Int,
      O <: Int,
      P <: Int,
      Q <: Int,
      S <: Int
    ](avg: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O],
      data: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]
    ) extends Volume[A, (M-P)/S+1, (N-Q)/S+1, O]

    case class MaxPoolingVolume[
      A: Ring: ClassTag,
      M <: Int,
      N <: Int,
      O <: Int,
      P <: Int,
      Q <: Int,
      S <: Int
    ](max: Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O],
      data: Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]
    ) extends Volume[A, (M-P)/S+1, (N-Q)/S+1, O]

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

    case class bias[A: Ring: ClassTag](bias: A):

      def apply[
        L <: Int: ValueOf,
        B <: Int: ValueOf,
        D <: Int: ValueOf
      ](initialization: Initialization)[O <: Int: ValueOf](using Conversion[Double, A]): Seq[Kernel[A, L, B, D]] =
        (1 to valueOf[O]).map(_ => new Kernel[A, L, B, D](bias, Tensor[A][L, B, D](initialization)))

    def apply[
      A: Ring: ClassTag,
      L <: Int: ValueOf,
      B <: Int: ValueOf,
      D <: Int: ValueOf
    ](initialization: Initialization)[O <: Int: ValueOf](using Conversion[Double, A]): Seq[Kernel[A, L, B, D]] =
      (1 to valueOf[O]).map(_ => new Kernel[A, L, B, D](initialization[A](), Tensor[A][L, B, D](initialization)))
