package nnn

import scala.compiletime.ops.int.{ +, -, /, * }

import scala.math.Ordering.Implicits.*
import scala.reflect.ClassTag

import spire.algebra.{ Field, Ring }
import spire.implicits.*


case class Matrix[
  A: Ring: ClassTag,
  M <: Int,
  N <: Int
](underlying: Array[Array[A]]) { self =>

  override def equals(any: Any): Boolean = any match
    case that: Matrix[?, ?, ?] if this.rows == that.rows && this.cols == that.cols =>
      var b = true
      for
        i <- 0 until rows
        j <- 0 until cols
        if b
      do
        b &&= this.underlying(i)(j) == that.underlying(i)(j)
      b
    case _ => false

  val rows = underlying.size
  val cols = underlying(0).size

  def apply(i: Int): Vector[A, N] =
    require(0 <= i && i < rows)
    Vector[A, N](underlying(i).toArray)

  def apply[I <: Int: ValueOf](): Vector[A, N] =
    apply(valueOf[I])

  def apply(i: Int, j: Int): A =
    require(0 <= i && i < rows && 0 <= j && j < cols)
    underlying(i)(j)

  def apply[I <: Int: ValueOf, J <: Int: ValueOf](): A =
    apply(valueOf[I], valueOf[J])

  def update(ij: (Int, Int), it: A): Unit =
    require(0 <= ij._1 && ij._1 < rows && 0 <= ij._2 && ij._2 < cols)
    underlying(ij._1)(ij._2) = it

  def update[I <: Int: ValueOf, J <: Int: ValueOf](it: A): Unit =
    update(valueOf[I] -> valueOf[J], it)

  def sum: A = underlying.map(_.reduce(_ + _)).reduce(_ + _)

  /**
    * binary operation
    */
  def op2[B, C: Ring: ClassTag](that: Matrix[B, M, N])(fun: (A, B) => C): Matrix[C, M, N] =
    val result = Array.fill(rows, cols)(Ring[C].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) += fun(this.underlying(i)(j), that.underlying(i)(j))
    Matrix[C, M, N](result)

  /**
    * addition
    */
  def +(that: Matrix[A, M, N]): Matrix[A, M, N] =
    op2(that)(_ + _)

  /**
    * alias for multiplication using dot product
    */
  inline def apply[P <: Int](that: Matrix[A, N, P]): Matrix[A, M, P] =
    this ⋅ that

  /**
    * multiplication using dot product
    */
  def ⋅[P <: Int](that: Matrix[A, N, P]): Matrix[A, M, P] =
    val result = Array.fill(this.rows, that.cols)(Ring[A].zero)
    for
      i <- 0 until this.rows
      j <- 0 until that.cols
      k <- 0 until this.cols
    do
      result(i)(j) += this.underlying(i)(k) * that.underlying(k)(j)
    Matrix[A, M, P](result)

  /**
    * element multiplication
    */
  def ⊙(that: Matrix[A, M, N]): Matrix[A, M, N] =
    op2(that)(_ * _)

  /**
    * multiplication with a vector
    */
  def ⋅(that: Vector[A, N]): Vector[A, M] =
    val result = Array.fill(rows)(Ring[A].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i) += this.underlying(i)(j) * that.underlying(j)
    Vector[A, M](result)

  /**
    * unary operation
    */
  def op[B: Ring: ClassTag](fun: A => B): Matrix[B, M, N] =
    val result = Array.fill(rows, cols)(Ring[B].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) = fun(underlying(i)(j))
    Matrix[B, M, N](result)

  /**
    * transpose
    */
  def unary_~ : Matrix[A, N, M] =
    val result = Array.fill(cols, rows)(Ring[A].zero)
    for
      i <- 0 until cols
      j <- 0 until rows
    do
      result(i)(j) = underlying(j)(i)
    Matrix[A, N, M](result)

  /**
    * cross-correlation
    */
  def ⋆[P <: Int, Q <: Int](that: Matrix[A, P, Q]): Matrix[A, M-P+1, N-Q+1] =
    require(that.rows <= this.rows && that.cols <= this.cols)
    val result = Array.fill(this.rows-that.rows+1, this.cols-that.cols+1)(Ring[A].zero)
    for
      i <- 0 until this.rows-that.rows+1
      j <- 0 until this.cols-that.cols+1
      k <- 0 until that.rows
      l <- 0 until that.cols
    do
      result(i)(j) += this.underlying(i+k)(j+l) * that.underlying(k)(l)
    Matrix[A, M-P+1, N-Q+1](result)

  /**
    * 180° rotation
    */
  def unary_! : Matrix[A, M, N] =
    val result = Array.fill(rows, cols)(Ring[A].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) = underlying(rows-i-1)(cols-j-1)
    Matrix[A, M, N](result)

  /**
    * convolution
    */
  def ∗[P <: Int, Q <: Int](that: Matrix[A, P, Q]): Matrix[A, M-P+1, N-Q+1] =
    this ⋆ !that

  /**
    * max-pooling
    */
  def max[P <: Int: ValueOf, Q <: Int: ValueOf](min: A)(using Ordering[A]): Matrix[A, M-P+1, N-Q+1] =
    val rows = valueOf[P]
    val cols = valueOf[Q]
    require(rows <= this.rows && cols <= this.cols)
    val result = Array.fill(this.rows-rows+1, this.cols-cols+1)(min)
    for
      i <- 0 until this.rows-rows+1
      j <- 0 until this.cols-cols+1
      k <- 0 until rows
      l <- 0 until cols
    do
      result(i)(j) = result(i)(j) max this.underlying(i+k)(j+l)
    Matrix[A, M-P+1, N-Q+1](result)

  /**
    * average-pooling
    */
  def avg[P <: Int: ValueOf, Q <: Int: ValueOf](using Field[A]): Matrix[A, M-P+1, N-Q+1] =
    val rows = valueOf[P]
    val cols = valueOf[Q]
    require(rows <= this.rows && cols <= this.cols)
    val size = rows*cols
    val result = Array.fill(this.rows-rows+1, this.cols-cols+1)(Ring[A].zero)
    for
      i <- 0 until this.rows-rows+1
      j <- 0 until this.cols-cols+1
    do
      for
        k <- 0 until rows
        l <- 0 until cols
      do
        result(i)(j) += this.underlying(i+k)(j+l)
      result(i)(j) /= size
    Matrix[A, M-P+1, N-Q+1](result)

  /**
    * w/ stride
    */
  def apply[S <: Int: ValueOf] = PartiallyAppliedStrideOps[S]

  final class PartiallyAppliedStrideOps[S <: Int: ValueOf]:
    private val stride: S = valueOf[S]
    require(stride > 0)

    /**
      * cross-correlation w/ stride
      */
    def ⋆[P <: Int, Q <: Int](that: Matrix[A, P, Q]): Matrix[A, (M-P)/S+1, (N-Q)/S+1] =
      require(that.rows <= self.rows && that.cols <= self.cols)
      require((self.rows-that.rows)%stride == 0 && (self.cols-that.cols)%stride == 0)
      val result = Array.fill((self.rows-that.rows)/stride+1, (self.cols-that.cols)/stride+1)(Ring[A].zero)
      for
        i <- 0 until self.rows-that.rows+1 by stride
        j <- 0 until self.cols-that.cols+1 by stride
        iʹ = i/stride
        jʹ = j/stride
        k <- 0 until that.rows
        l <- 0 until that.cols
      do
        result(iʹ)(jʹ) += self.underlying(i+k)(j+l) * that.underlying(k)(l)
      Matrix[A, (M-P)/S+1, (N-Q)/S+1](result)

    /**
      * convolution w/ stride
      */
    def ∗[P <: Int, Q <: Int](that: Matrix[A, P, Q]): Matrix[A, (M-P)/S+1, (N-Q)/S+1] =
      this ⋆ !that

    /**
      * max-pooling w/ stride
      */
    def max[P <: Int: ValueOf, Q <: Int: ValueOf](min: A)(using Ordering[A]): Matrix[A, (M-P)/S+1, (N-Q)/S+1] =
      val rows = valueOf[P]
      val cols = valueOf[Q]
      require(rows <= self.rows && cols <= self.cols)
      require((self.rows-rows)%stride == 0, (self.cols-cols)%stride == 0)
      val result = Array.fill((self.rows-rows)/stride+1, (self.cols-cols)/stride+1)(min)
      for
        i <- 0 until self.rows-rows+1 by stride
        j <- 0 until self.cols-cols+1 by stride
        iʹ = i/stride
        jʹ = j/stride
        k <- 0 until rows
        l <- 0 until cols
      do
        result(iʹ)(jʹ) = result(iʹ)(jʹ) max self.underlying(i+k)(j+l)
      Matrix[A, (M-P)/S+1, (N-Q)/S+1](result)

    /**
      * average-pooling w/ stride
      */
    def avg[P <: Int: ValueOf, Q <: Int: ValueOf](using Field[A]): Matrix[A, (M-P)/S+1, (N-Q)/S+1] =
      val rows = valueOf[P]
      val cols = valueOf[Q]
      require(rows <= self.rows && cols <= self.cols)
      require((self.rows-rows)%stride == 0, (self.cols-cols)%stride == 0)
      val size = rows*cols
      val result = Array.fill((self.rows-rows)/stride+1, (self.cols-cols)/stride+1)(Ring[A].zero)
      for
        i <- 0 until self.rows-rows+1 by stride
        j <- 0 until self.cols-cols+1 by stride
        iʹ = i/stride
        jʹ = j/stride
      do
        for
          k <- 0 until rows
          l <- 0 until cols
        do
          result(iʹ)(jʹ) += self.underlying(i+k)(j+l)
        result(iʹ)(jʹ) /= size
      Matrix[A, (M-P)/S+1, (N-Q)/S+1](result)

  /**
    * unsafe assignment
    */
  def :=(that: Matrix[A, ?, ?]): Unit =
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      this.underlying(i)(j) = that.underlying(i)(j)

  /**
    * addition and reassignment
    */
  def +=(that: Matrix[A, M, N]): Unit =
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      this.underlying(i)(j) += that.underlying(i)(j)

  def to[B: Ring: ClassTag](using c: Conversion[A, B]): Matrix[B, M, N] =
    val result = Array.fill(rows, cols)(Ring[B].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) = c(underlying(i)(j))
    Matrix[B, M, N](result)

  def unary_- : Vector[A, M*N] =
    Vector[A, M*N](underlying.flatten.toArray)

  def toSeq: Seq[Seq[A]] = underlying.map(_.toSeq).toSeq

  override def toString: String = underlying.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]")

}


object Matrix:

  given [A: Ring: ClassTag, N <: Int]: Conversion[Matrix[A, 1, N], Vector[A, N]] with
    def apply(self: Matrix[A, 1, N]): Vector[A, N] =
      Vector[A, N](self.underlying(0).toArray)

  given [A]: Conversion[Matrix[A, 1, 1], A] with
    def apply(self: Matrix[A, 1, 1]): A =
      self.underlying(0)(0)

  def apply[A: Ring: ClassTag] = PartiallyAppliedOps[A]

  final class PartiallyAppliedOps[A: Ring: ClassTag]:

    def apply[M <: Int: ValueOf, N <: Int: ValueOf](elements: A*): Matrix[A, M, N] =
      require(valueOf[M] > 0 && valueOf[N] > 0)
      require(elements.size == valueOf[M] * valueOf[N])

      Matrix[A, M, N](elements.sliding(valueOf[N], valueOf[N]).map(_.toArray).toArray)

    def one[M <: Int: ValueOf, N <: Int: ValueOf]: Matrix[A, M, N] =
      require(valueOf[M] > 0 && valueOf[N] > 0)

      apply[M, N](Seq.fill(valueOf[M] * valueOf[N])(Ring.one)*)

    def zero[M <: Int: ValueOf, N <: Int: ValueOf]: Matrix[A, M, N] =
      require(valueOf[M] > 0 && valueOf[N] > 0)

      apply[M, N](Seq.fill(valueOf[M] * valueOf[N])(Ring.zero)*)

    def constant[M <: Int: ValueOf, N <: Int: ValueOf](element: A): Matrix[A, M, N] =
      require(valueOf[M] > 0 && valueOf[N] > 0)

      apply[M, N](Seq.fill(valueOf[M] * valueOf[N])(element)*)

    def diagonal[N <: Int: ValueOf](element: A): Matrix[A, N, N] =
      require(valueOf[N] > 0)

      val result = Array.fill(valueOf[N], valueOf[N])(Ring.zero)

      for
        i <- 0 until valueOf[N]
      do
        result(i)(i) = element

      Matrix[A, N, N](result)

    def identity[N <: Int: ValueOf]: Matrix[A, N, N] = diagonal[N](Ring.one)
