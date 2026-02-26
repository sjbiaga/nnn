package nnn

import scala.compiletime.ops.int.{ +, -, /, * }

import scala.math.Ordering.Implicits.*
import scala.reflect.ClassTag

import spire.algebra.{ Field, Ring }
import spire.implicits.*


/**
  * Matrix of vectors
  */
case class Tensor[
  A: Ring: ClassTag,
  M <: Int,
  N <: Int,
  O <: Int
](underlying: Array[Array[Array[A]]]) { self =>

  override def equals(any: Any): Boolean = any match
    case that: Tensor[?, ?, ?, ?] if this.rows == that.rows && this.cols == that.cols =>
      var b = true
      for
        h <- 0 until depth
        i <- 0 until rows
        j <- 0 until cols
        if b
      do
        b &&= this.underlying(i)(j)(h) == that.underlying(i)(j)(h)
      b
    case _ => false

  val rows = underlying.size
  val cols = underlying(0).size
  val depth = underlying(0)(0).size
  val size = rows * cols * depth

  def apply(k: Int): Matrix[A, M, N] =
    require(0 <= k && k < depth)
    Matrix[A, M, N](underlying.map(_.map(_(k))))

  def apply[K <: Int: ValueOf](): Matrix[A, M, N] =
    apply(valueOf[K])

  def apply(i: Int, j: Int): Vector[A, O] =
    require(0 <= i && i < rows && 0 <= j && j < cols)
    Vector[A, O](underlying(i)(j).toArray)

  def apply[I <: Int: ValueOf, J <: Int: ValueOf](): Vector[A, O] =
    apply(valueOf[I], valueOf[J])

  def apply(i: Int, j: Int, k: Int): A =
    require(0 <= i && i < rows && 0 <= j && j < cols && 0 <= k && k < depth)
    underlying(i)(j)(k)

  def apply[I <: Int: ValueOf, J <: Int: ValueOf, K <: Int: ValueOf](): A =
    apply(valueOf[I], valueOf[J], valueOf[K])

  def update(ijk: (Int, Int, Int), it: A): this.type =
    require(0 <= ijk._1 && ijk._1 < rows && 0 <= ijk._2 && ijk._2 < cols && 0 <= ijk._3 && ijk._3 < depth)
    underlying(ijk._1)(ijk._2)(ijk._3) = it
    this

  def sum: A =
    var r = Ring[A].zero
    for
      i <- 0 until rows
      j <- 0 until cols
      k <- 0 until depth
    do
      r += underlying(i)(j)(k)
    r

  def reduce(fun: (A, A) => A): Matrix[A, M, N] =
    val result = Array.fill(rows, cols)(Ring[A].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) = underlying(i)(j).reduce(fun)
    Matrix[A, M, N](result)

  underlying.map(_.map(_.reduce(_ + _)).reduce(_ + _)).reduce(_ + _)

  /**
    * binary operation
    */
  def op2[B, C: Ring: ClassTag](that: Tensor[B, M, N, O])(fun: (A, B) => C): Tensor[C, M, N, O] =
    require(this.size == that.size)
    val result = Array.fill(rows, cols, depth)(Ring[C].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
      k <- 0 until depth
    do
      result(i)(j)(k) += fun(this.underlying(i)(j)(k), that.underlying(i)(j)(k))
    Tensor[C, M, N, O](result)

  /**
    * addition
    */
  def +(that: Tensor[A, M, N, O]): Tensor[A, M, N, O] =
    op2(that)(_ + _)

  /**
    * subtraction
    */
  def -(that: Tensor[A, M, N, O]): Tensor[A, M, N, O] =
    op2(that)(_ - _)

  /**
    * element multiplication (Hadamard product)
    */
  def ⊙(that: Tensor[A, M, N, O]): Tensor[A, M, N, O] =
    op2(that)(_ * _)

  /**
    * unary operation
    */
  def op[B: Ring: ClassTag](fun: A => B): Tensor[B, M, N, O] =
    val result = Array.fill(rows, cols, depth)(Ring[B].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
      k <- 0 until depth
    do
      result(i)(j)(k) = fun(underlying(i)(j)(k))
    Tensor[B, M, N, O](result)

  /**
    * unary operation w/ index
    */
  def op[B: Ring: ClassTag](fun: (A, (Int, Int, Int)) => B): Tensor[B, M, N, O] =
    val result = Array.fill(rows, cols, depth)(Ring[B].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
      k <- 0 until depth
    do
      result(i)(j)(k) = fun(underlying(i)(j)(k), (i, j, k))
    Tensor[B, M, N, O](result)

  /**
    * reset to zero
    */
  def reset: this.type =
    for
      i <- 0 until rows
      j <- 0 until cols
      k <- 0 until depth
    do
      underlying(i)(j)(k) = Ring[A].zero
    this

  /**
    * padding
    */
  def pad[P <: Int: ValueOf, Q <: Int: ValueOf]: Tensor[A, M+2*P, N+2*Q, O] =
    val P = valueOf[P]
    val Q = valueOf[Q]
    val result = Array.fill(rows+2*P, cols+2*Q, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      result(P+i)(Q+j)(h) = underlying(i)(j)(h)
    Tensor[A, M+2*P, N+2*Q, O](result)

  /**
    * cropping
    */
  def crop[P <: Int: ValueOf, Q <: Int: ValueOf]: Tensor[A, M-2*P, N-2*Q, O] =
    val P = valueOf[P]
    val Q = valueOf[Q]
    val result = Array.fill(rows-2*P, cols-2*Q, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until rows-2*P
      j <- 0 until cols-2*Q
    do
      result(i)(j)(h) = underlying(P+i)(Q+j)(h)
    Tensor[A, M-2*P, N-2*Q, O](result)

  /**
    * w/ stride
    */
  def apply[S <: Int: ValueOf] = PartiallyAppliedStrideOps[S]

  final class PartiallyAppliedStrideOps[S <: Int: ValueOf]:
    private val stride: S = valueOf[S]
    require(stride > 0)

    /**
      * dilation
      */
    def dilate: Tensor[A, M + (M-1)*(S-1), N + (N-1)*(S-1), O] =
      val rows = self.rows+(self.rows-1)*(stride-1)
      val cols = self.cols+(self.cols-1)*(stride-1)
      val result = Array.fill(rows, cols, depth)(Ring[A].zero)
      for
        h <- 0 until depth
        i <- 0 until rows by stride
        j <- 0 until cols by stride
        iʹ = i/stride
        jʹ = j/stride
      do
        result(i)(j)(h) = underlying(iʹ)(jʹ)(h)
      Tensor[A, M + (M-1)*(S-1), N + (N-1)*(S-1), O](result)

    /**
      * padding and dilation
      */
    def pad_and_dilate[P <: Int: ValueOf, Q <: Int: ValueOf]: Tensor[A, M + 2*(P-1) + (M-1)*(S-1), N + 2*(Q-1) + (N-1)*(S-1), O] =
      val P = valueOf[P]
      val Q = valueOf[Q]
      val rows = self.rows+(self.rows-1)*(stride-1)
      val cols = self.cols+(self.cols-1)*(stride-1)
      val result = Array.fill(rows+2*(P-1), cols+2*(Q-1), depth)(Ring[A].zero)
      for
        h <- 0 until depth
        i <- 0 until rows by stride
        j <- 0 until cols by stride
        iʹ = i/stride
        jʹ = j/stride
      do
        result(P-1+i)(Q-1+j)(h) = underlying(iʹ)(jʹ)(h)
      Tensor[A, M + 2*(P-1) + (M-1)*(S-1), N + 2*(Q-1) + (N-1)*(S-1), O](result)

    /**
      * cross-correlation w/ stride
      */
    def ⋆[P <: Int, Q <: Int](that: Tensor[A, P, Q, O]): Tensor[A, (M-P)/S+1, (N-Q)/S+1, O] =
      require(that.rows <= self.rows && that.cols <= self.cols)
      require((self.rows-that.rows)%stride == 0 && (self.cols-that.cols)%stride == 0)
      val result = Array.fill((self.rows-that.rows)/stride+1, (self.cols-that.cols)/stride+1, depth)(Ring[A].zero)
      for
        h <- 0 until depth
        i <- 0 until self.rows-that.rows+1 by stride
        j <- 0 until self.cols-that.cols+1 by stride
        iʹ = i/stride
        jʹ = j/stride
        k <- 0 until that.rows
        l <- 0 until that.cols
      do
        result(iʹ)(jʹ)(h) += self.underlying(i+k)(j+l)(h) * that.underlying(k)(l)(h)
      Tensor[A, (M-P)/S+1, (N-Q)/S+1, O](result)

    /**
      * max-pooling w/ stride
      */
    def max[P <: Int: ValueOf, Q <: Int: ValueOf](min: A)(using Ordering[A]): (Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O], Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]) =
      val rows = valueOf[P]
      val cols = valueOf[Q]
      require(rows <= self.rows && cols <= self.cols)
      require((self.rows-rows)%stride == 0, (self.cols-cols)%stride == 0)
      val max = Array.fill((self.rows-rows)/stride+1, (self.cols-cols)/stride+1, depth)((0, 0))
      val result = Array.fill((self.rows-rows)/stride+1, (self.cols-cols)/stride+1, depth)(min)
      for
        h <- 0 until depth
        i <- 0 until self.rows-rows+1 by stride
        j <- 0 until self.cols-cols+1 by stride
        iʹ = i/stride
        jʹ = j/stride
        k <- i until i+rows
        l <- j until j+cols
      do
        if result(iʹ)(jʹ)(h) < self.underlying(k)(l)(h)
        then
          max(iʹ)(jʹ)(h) = k -> l
          result(iʹ)(jʹ)(h) = self.underlying(k)(l)(h)
      Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O](max) -> Tensor[A, (M-P)/S+1, (N-Q)/S+1, O](result)

    /**
      * average-pooling w/ stride
      */
    def avg[P <: Int: ValueOf, Q <: Int: ValueOf](using Field[A]): Tensor[A, (M-P)/S+1, (N-Q)/S+1, O] =
      val rows = valueOf[P]
      val cols = valueOf[Q]
      require(rows <= self.rows && cols <= self.cols)
      require((self.rows-rows)%stride == 0, (self.cols-cols)%stride == 0)
      val size = rows * cols
      val result = Array.fill((self.rows-rows)/stride+1, (self.cols-cols)/stride+1, depth)(Ring[A].zero)
      for
        h <- 0 until depth
        i <- 0 until self.rows-rows+1 by stride
        j <- 0 until self.cols-cols+1 by stride
        iʹ = i/stride
        jʹ = j/stride
      do
        for
          k <- i until i+rows
          l <- j until j+cols
        do
          result(iʹ)(jʹ)(h) += self.underlying(k)(l)(h)
        result(iʹ)(jʹ)(h) /= size
      Tensor[A, (M-P)/S+1, (N-Q)/S+1, O](result)

  /**
    * unsafe assignment
    */
  def :=(that: Tensor[A, ?, ?, ?]): Unit =
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      this.underlying(i)(j)(h) = that.underlying(i)(j)(h)

  /**
    * addition and reassignment
    */
  def +=(that: Tensor[A, M, N, O]): Unit =
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      this.underlying(i)(j)(h) += that.underlying(i)(j)(h)

  def to[B: Ring: ClassTag](using c: Conversion[A, B]): Tensor[B, M, N, O] =
    val result = Array.fill(rows, cols, depth)(Ring[B].zero)
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j)(h) = c(underlying(i)(j)(h))
    Tensor[B, M, N, O](result)

  def toSeq: Seq[Seq[Seq[A]]] = underlying.map(_.map(_.toSeq).toSeq).toSeq

  override def toString: String = underlying.map(_.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]")).mkString("[", ", ", "]")

}


object Tensor:

  given [A: Ring: ClassTag, N <: Int, P <: Int]: Conversion[Tensor[A, 1, N, P], Matrix[A, N, P]] with
    def apply(self: Tensor[A, 1, N, P]): Matrix[A, N, P] =
      Matrix[A, N, P](self.underlying(0).toArray)

  given N1toN[A: Ring: ClassTag, M <: Int, N <: Int]: Conversion[Tensor[A, M, N, 1], Matrix[A, M, N]] with
    def apply(self: Tensor[A, M, N, 1]): Matrix[A, M, N] =
      Matrix[A, M, N](self.underlying.map(_.flatten))

  given [A]: Conversion[Tensor[A, 1, 1, 1], A] with
    def apply(self: Tensor[A, 1, 1, 1]): A =
      self.underlying(0)(0)(0)

  def apply[A: Ring: ClassTag] = PartiallyAppliedOps[A]

  final class PartiallyAppliedOps[A: Ring: ClassTag]:

    def apply[
      M <: Int: ValueOf,
      N <: Int: ValueOf,
      P <: Int: ValueOf
    ](elements: A*): Tensor[A, M, N, P] =
      val M = valueOf[M]
      val N = valueOf[N]
      val P = valueOf[P]

      require(M > 0 && N > 0 && P > 0)
      require(elements.size == M * N * P)

      Tensor[A, M, N, P](elements.sliding(N*P, N*P).map(_.sliding(P, P)).map(_.map(_.toArray).toArray).toArray)

    def apply[
      M <: Int,
      N <: Int,
      P <: Int: ValueOf
    ](matrices: Matrix[A, M, N]*): Tensor[A, M, N, P] =
      val P = valueOf[P]

      require(P > 0 && P == matrices.size)

      val M = matrices(0).rows
      val N = matrices(0).cols

      val result = Array.fill(M, N, P)(Ring[A].zero)
      for
        i <- 0 until M
        j <- 0 until N
        k <- 0 until P
      do
        result(i)(j)(k) = matrices(k).underlying(i)(j)
      Tensor[A, M, N, P](result)

    def apply[
      M <: Int: ValueOf,
      N <: Int: ValueOf,
      P <: Int: ValueOf
    ](initialization: Initialization)(using Conversion[Double, A]): Tensor[A, M, N, P] =
      require(valueOf[M] > 0 && valueOf[N] > 0 && valueOf[P] > 0)

      apply[M, N, P]({
        for
          _ <- 0 until valueOf[M]
          _ <- 0 until valueOf[N]
          _ <- 0 until valueOf[P]
        yield
          initialization()
      }*)

    def one[M <: Int: ValueOf, N <: Int: ValueOf, O <: Int: ValueOf]: Tensor[A, M, N, O] =
      constant(Ring[A].one)

    def zero[M <: Int: ValueOf, N <: Int: ValueOf, O <: Int: ValueOf]: Tensor[A, M, N, O] =
      constant(Ring[A].zero)

    def constant[M <: Int: ValueOf, N <: Int: ValueOf, O <: Int: ValueOf](element: A): Tensor[A, M, N, O] =
      require(valueOf[M] > 0 && valueOf[N] > 0 && valueOf[O] > 0)

      apply[M, N, O](Seq.fill(valueOf[M] * valueOf[N] * valueOf[O])(element)*)

    def stack[M <: Int, N <: Int](matrices: Matrix[A, M, N]*)[O <: Int: ValueOf]: Tensor[A, M, N, O] =
      val depth = valueOf[O]

      require(depth > 0)
      require(depth == matrices.size)

      val rows = matrices(0).rows
      val cols = matrices(0).cols

      val result = Array.fill(rows, cols, depth)(Ring.zero)

      for
        i <- 0 until rows
        j <- 0 until cols
        k <- 0 until depth
      do
        result(i)(j)(k) = matrices(k)(i, j)

      Tensor[A, M, N, O](result)
