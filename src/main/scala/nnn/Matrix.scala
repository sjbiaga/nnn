package nnn

import scala.reflect.ClassTag

import spire.algebra.Ring
import spire.implicits.*


case class Matrix[
  A: Ring: ClassTag,
  M <: Int,
  N <: Int
](underlying: Array[Array[A]]):

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
    new Vector[A, N](underlying(i).toArray)

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
  def op2(that: Matrix[A, M, N])(fun: (A, A) => A): Matrix[A, M, N] =
    val result = Array.fill(rows, cols)(Ring[A].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) += fun(this.underlying(i)(j), that.underlying(i)(j))
    new Matrix[A, M, N](result)

  /**
    * addition
    */
  def +(that: Matrix[A, M, N]): Matrix[A, M, N] =
    op2(that)(_ + _)

  /**
    * multiplication
    */
  def ⋅[P <: Int: ValueOf](that: Matrix[A, N, P]): Matrix[A, M, P] =
    val result = Array.fill(this.rows, that.cols)(Ring[A].zero)
    for
      i <- 0 until this.rows
      j <- 0 until that.cols
      k <- 0 until this.cols
    do
      result(i)(j) += this.underlying(i)(k) * that.underlying(k)(j)
    new Matrix[A, M, P](result)

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
    new Vector[A, M](result)

  /**
    * negation
    */
  def unary_- : Matrix[A, M, N] =
    op(-_)

  /**
    * unary operation
    */
  def op(fun: A => A): Matrix[A, M, N] =
    val result = Array.fill(rows, cols)(Ring[A].zero)
    for
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j) = fun(underlying(i)(j))
    new Matrix[A, M, N](result)

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
    new Matrix[A, N, M](result)

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
    new Matrix[B, M, N](result)

  def toSeq: Seq[Seq[A]] = underlying.map(_.toSeq).toSeq

  override def toString: String = underlying.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]")


object Matrix:

  given [A: Ring: ClassTag, N <: Int: ValueOf]: Conversion[Matrix[A, 1, N], Vector[A, N]] with
    def apply(self: Matrix[A, 1, N]): Vector[A, N] =
      new Vector[A, N](self.underlying(0).toArray)

  def apply[A: Ring: ClassTag,
            M <: Int: ValueOf,
            N <: Int: ValueOf
  ](elements: A*): Matrix[A, M, N] =
    require(valueOf[M] > 0 && valueOf[N] > 0)
    require(elements.size == valueOf[M] * valueOf[N])

    new Matrix[A, M, N](elements.sliding(valueOf[N], valueOf[N]).map(_.toArray).toArray)

  def one[A: Ring: ClassTag,
          M <: Int: ValueOf,
          N <: Int: ValueOf
  ]: Matrix[A, M, N] =
    require(valueOf[M] > 0 && valueOf[N] > 0)

    Matrix[A, M, N](Seq.fill(valueOf[M] * valueOf[N])(Ring.one)*)

  def zero[A: Ring: ClassTag,
           M <: Int: ValueOf,
           N <: Int: ValueOf
  ]: Matrix[A, M, N] =
    require(valueOf[M] > 0 && valueOf[N] > 0)

    Matrix[A, M, N](Seq.fill(valueOf[M] * valueOf[N])(Ring.zero)*)

  def constant[A: Ring: ClassTag,
               M <: Int: ValueOf,
               N <: Int: ValueOf
  ](element: A): Matrix[A, M, N] =
    require(valueOf[M] > 0 && valueOf[N] > 0)

    Matrix[A, M, N](Seq.fill(valueOf[M] * valueOf[N])(element)*)

  def diagonal[A: Ring: ClassTag,
               N <: Int: ValueOf
  ](element: A): Matrix[A, N, N] =
    require(valueOf[N] > 0)

    val result = Array.fill(valueOf[N], valueOf[N])(Ring.zero)

    for
      i <- 0 until valueOf[N]
    do
      result(i)(i) = element

    new Matrix[A, N, N](result)

  def identity[A: Ring: ClassTag,
               N <: Int: ValueOf
  ]: Matrix[A, N, N] = diagonal[A, N](Ring.one)
