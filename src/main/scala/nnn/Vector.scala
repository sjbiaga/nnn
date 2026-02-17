package nnn

import scala.compiletime.ops.int.{ +, -, / }

import scala.reflect.ClassTag

import spire.algebra.Ring
import spire.implicits.*

import Vector.given


/**
  * Vector of scalars
  */
case class Vector[
  A: Ring: ClassTag,
  N <: Int
](underlying: Array[A]):

  override def equals(any: Any): Boolean = any match
    case that: Vector[?, ?] if this.rows == that.rows =>
      var b = true
      for
        k <- 0 until rows
        if b
      do
        b &&= this.underlying(k) == that.underlying(k)
      b
    case _ => false

  val size = underlying.size
  val cols = size
  val rows = size

  def apply(i: Int): A =
    require(0 <= i && i < rows)
    underlying(i)

  def apply[I <: Int: ValueOf](): A =
    apply(valueOf[I])

  def update(i: Int, it: A): this.type =
    require(0 <= i && i < rows)
    underlying(i) = it
    this

  def update[I <: Int: ValueOf](it: A): this.type =
    update(valueOf[I], it)

  def sum: A = underlying.reduce(_ + _)

  /**
    * binary operation
    */
  def op2[B, C: Ring: ClassTag](that: Vector[B, N])(fun: (A, B) => C): Vector[C, N] =
    val result = Array.fill(rows)(Ring[C].zero)
    for
      k <- 0 until rows
    do
      result(k) = fun(this.underlying(k), that.underlying(k))
    Vector[C, N](result)

  /**
    * addition
    */
  def +(that: Vector[A, N]): Vector[A, N] =
    op2(that)(_ + _)

  /**
    * subtraction
    */
  def -(that: Vector[A, N]): Vector[A, N] =
    op2(that)(_ - _)

  /**
    * alias for dot product
    */
  inline def apply(that: Vector[A, N]): A =
    this ⋅ that

  /**
    * dot product
    */
  def ⋅(that: Vector[A, N])(using DummyImplicit): A =
    var result = Ring[A].zero
    for
      k <- 0 until rows
    do
      result += this.underlying(k) * that.underlying(k)
    result

  /**
    * multiplication as matrices
    */
  def ⋅[M <: Int](that: Vector[A, M]): Matrix[A, N, M] =
    val result = Array.fill(this.rows, that.rows)(Ring[A].zero)
    for
      i <- 0 until this.rows
      j <- 0 until that.cols
    do
      result(i)(j) = this.underlying(i) * that.underlying(j)
    Matrix[A, N, M](result)

  /**
    * element multiplication (Hadamard product)
    */
  def ⊙(that: Vector[A, N]): Vector[A, N] =
    op2(that)(_ * _)

  /**
    * unary operation
    */
  def op[B: Ring: ClassTag](fun: A => B): Vector[B, N] =
    val result = Array.fill(rows)(Ring[B].zero)
    for
      k <- 0 until rows
    do
      result(k) = fun(underlying(k))
    Vector[B, N](result)

  /**
    * unary operation w/ index
    */
  def op[B: Ring: ClassTag](fun: (A, Int) => B): Vector[B, N] =
    val result = Array.fill(rows)(Ring[B].zero)
    for
      k <- 0 until rows
    do
      result(k) = fun(underlying(k), k)
    Vector[B, N](result)

  /**
    * reshape
    */
  def reshape[M <: Int: ValueOf]: Matrix[A, M, N/M] =
    val M = valueOf[M]
    require(rows % M == 0)
    given ValueOf[N/M] = ValueOf((rows/M).asInstanceOf[N/M])
    Matrix[A][M, N/M](toSeq*)

  /**
    * unsafe assignment
    */
  def :=(that: Vector[A, ?]): Unit =
    for
      k <- 0 until rows
    do
      this.underlying(k) = that.underlying(k)

  /**
    * addition and reassignment
    */
  def +=(that: Vector[A, N]): Unit =
    for
      k <- 0 until rows
    do
      this.underlying(k) += that.underlying(k)

  /**
    * opposite of [[--]]
    */
  def ++(it: A = Ring[A].zero): Vector[A, N+1] =
    val result = Array.fill(rows+1)(Ring[A].zero)
    result(0) = it
    for
      k <- 0 until rows
    do
      result(k+1) = underlying(k)
    Vector[A, N+1](result)

  /**
    * opposite of [[++]]
    */
  def -- : Vector[A, N-1] =
    val result = Array.fill(rows-1)(Ring[A].zero)
    for
      k <- 1 until rows
    do
      result(k-1) = underlying(k)
    Vector[A, N-1](result)

  def to[B: Ring: ClassTag](using c: Conversion[A, B]): Vector[B, N] =
    val result = Array.fill(rows)(Ring[B].zero)
    for
      k <- 0 until rows
    do
      result(k) = c(underlying(k))
    Vector[B, N](result)

  /**
    * flatten
    */
  def toSeq: Seq[A] = underlying.toSeq

  override def toString: String = underlying.mkString("[", ", ", "]")


object Vector:

  given [A: Ring, N <: Int]: Conversion[Vector[A, N+1-1], Vector[A, N]] = _.asInstanceOf[Vector[A, N]]

  given [A: Ring: ClassTag, N <: Int]: Conversion[Vector[A, N], Matrix[A, 1, N]] with
    def apply(self: Vector[A, N]): Matrix[A, 1, N] =
      Matrix[A, 1, N](Array(self.underlying.toArray))

  given NtoN1[A: Ring: ClassTag, N <: Int]: Conversion[Vector[A, N], Matrix[A, N, 1]] with
    def apply(self: Vector[A, N]): Matrix[A, N, 1] =
      Matrix[A, N, 1](self.underlying.map(Array(_)))

  given [A]: Conversion[Vector[A, 1], A] with
    def apply(self: Vector[A, 1]): A =
      self.underlying(0)

  def apply[A: Ring: ClassTag] = PartiallyAppliedOps[A]

  final class PartiallyAppliedOps[A: Ring: ClassTag]:

    def apply[N <: Int: ValueOf](elements: A*): Vector[A, N] =
      require(valueOf[N] > 0)
      require(elements.size == valueOf[N])

      Vector[A, N](elements.toArray)

    def apply[N <: Int: ValueOf](producer: Int => A): Vector[A, N] =
      require(valueOf[N] > 0)

      apply[N]((0 until valueOf[N]).map(i => producer(i)).toSeq*)

    def apply[N <: Int: ValueOf](initialization: Initialization)(using Conversion[Double, A]): Vector[A, N] =
      require(valueOf[N] > 0)

      apply[N]((0 until valueOf[N]).map(_ => initialization()).toSeq*)

    def one[N <: Int: ValueOf]: Vector[A, N] =
      require(valueOf[N] > 0)

      apply[N](Seq.fill(valueOf[N])(Ring[A].one)*)

    def zero[N <: Int: ValueOf]: Vector[A, N] =
      require(valueOf[N] > 0)

      apply[N](Seq.fill(valueOf[N])(Ring[A].zero)*)
