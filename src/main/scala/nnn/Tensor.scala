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

  def apply(i: Int): Matrix[A, N, O] =
    require(0 <= i && i < rows)
    Matrix[A, N, O](underlying(i).toArray)

  def apply[I <: Int: ValueOf](): Matrix[A, N, O] =
    apply(valueOf[I])

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

  def update[I <: Int: ValueOf, J <: Int: ValueOf, K <: Int: ValueOf](it: A): this.type =
    update((valueOf[I], valueOf[J], valueOf[K]), it)

  def sum: A = underlying.map(_.map(_.reduce(_ + _)).reduce(_ + _)).reduce(_ + _)

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
    val result = Array.fill(rows, cols, depth)(Ring[C].zero)
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j)(h) += fun(this.underlying(i)(j)(h), that.underlying(i)(j)(h))
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
    * alias for multiplication using dot product
    */
  inline def apply[P <: Int](that: Tensor[A, N, P, O]): Tensor[A, M, P, O] =
    this ⋅ that

  /**
    * multiplication using dot product
    */
  def ⋅[P <: Int](that: Tensor[A, N, P, O]): Tensor[A, M, P, O] =
    val result = Array.fill(this.rows, that.cols, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until this.rows
      j <- 0 until that.cols
      k <- 0 until this.cols
    do
      result(i)(j)(h) += this.underlying(i)(k)(h) * that.underlying(k)(j)(h)
    Tensor[A, M, P, O](result)

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
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j)(h) = fun(underlying(i)(j)(h))
    Tensor[B, M, N, O](result)

  /**
    * unary operation w/ index
    */
  def op[B: Ring: ClassTag](fun: (A, (Int, Int, Int)) => B): Tensor[B, M, N, O] =
    val result = Array.fill(rows, cols, depth)(Ring[B].zero)
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j)(h) = fun(underlying(i)(j)(h), (i, j, h))
    Tensor[B, M, N, O](result)

  /**
    * transpose
    */
  def unary_~ : Tensor[A, N, M, O] =
    val result = Array.fill(cols, rows, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until cols
      j <- 0 until rows
    do
      result(i)(j)(h) = underlying(j)(i)(h)
    Tensor[A, N, M, O](result)

  /**
    * cross-correlation
    */
  def ⋆[P <: Int, Q <: Int](that: Tensor[A, P, Q, O]): Tensor[A, M-P+1, N-Q+1, O] =
    require(that.rows <= this.rows && that.cols <= this.cols)
    val result = Array.fill(this.rows-that.rows+1, this.cols-that.cols+1, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until this.rows-that.rows+1
      j <- 0 until this.cols-that.cols+1
      k <- 0 until that.rows
      l <- 0 until that.cols
    do
      result(i)(j)(h) += this.underlying(i+k)(j+l)(h) * that.underlying(k)(l)(h)
    Tensor[A, M-P+1, N-Q+1, O](result)

  /**
    * 180° rotation
    */
  def unary_! : Tensor[A, M, N, O] =
    val result = Array.fill(rows, cols, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until rows
      j <- 0 until cols
    do
      result(i)(j)(h) = underlying(rows-i-1)(cols-j-1)(h)
    Tensor[A, M, N, O](result)

  /**
    * convolution
    */
  def ∗[P <: Int, Q <: Int](that: Tensor[A, P, Q, O]): Tensor[A, M-P+1, N-Q+1, O] =
    this ⋆ !that

  /**
    * max-pooling
    */
  def max[P <: Int: ValueOf, Q <: Int: ValueOf](min: A)(using Ordering[A]): (Tensor[(Int, Int), M-P+1, N-Q+1, O], Tensor[A, M-P+1, N-Q+1, O]) =
    val rows = valueOf[P]
    val cols = valueOf[Q]
    require(rows <= this.rows && cols <= this.cols)
    val max = Array.fill(this.rows-rows+1, this.cols-cols+1, depth)((0, 0))
    val result = Array.fill(this.rows-rows+1, this.cols-cols+1, depth)(min)
    for
      h <- 0 until depth
      i <- 0 until this.rows-rows+1
      j <- 0 until this.cols-cols+1
      k <- i until i+rows
      l <- j until j+cols
    do
      if result(i)(j)(h) < this.underlying(k)(l)(h)
      then
        max(i)(j)(h) = k -> l
        result(i)(j)(h) = this.underlying(k)(l)(h)
    Tensor[(Int, Int), M-P+1, N-Q+1, O](max) -> Tensor[A, M-P+1, N-Q+1, O](result)

  /**
    * average-pooling
    */
  def avg[P <: Int: ValueOf, Q <: Int: ValueOf](using Field[A]): Tensor[A, M-P+1, N-Q+1, O] =
    val rows = valueOf[P]
    val cols = valueOf[Q]
    require(rows <= this.rows && cols <= this.cols)
    val size = rows * cols
    val result = Array.fill(this.rows-rows+1, this.cols-cols+1, depth)(Ring[A].zero)
    for
      h <- 0 until depth
      i <- 0 until this.rows-rows+1
      j <- 0 until this.cols-cols+1
    do
      for
        k <- 0 until rows
        l <- 0 until cols
      do
        result(i)(j)(h) += this.underlying(i+k)(j+l)(h)
      result(i)(j)(h) /= size
    Tensor[A, M-P+1, N-Q+1, O](result)

  /**
    * w/ stride
    */
  def apply[S <: Int: ValueOf] = PartiallyAppliedStrideOps[S]

  final class PartiallyAppliedStrideOps[S <: Int: ValueOf]:
    private val stride: S = valueOf[S]
    require(stride > 0)

    /**
      * prime
      */
    object ʹ:

      /**
        * cross-correlation w/ stride
        */
      def ⋆[P <: Int: ValueOf, Q <: Int: ValueOf]: Tensor[A, ((M-P)/S+1)*((N-Q)/S+1), P*Q, O] =
        val rows = valueOf[P]
        val cols = valueOf[Q]
        require(rows <= self.rows && cols <= self.cols)
        require((self.rows-rows)%stride == 0 && (self.cols-cols)%stride == 0)
        val result = Array.fill(((self.rows-rows)/stride+1)*((self.cols-cols)/stride+1), rows*cols, depth)(Ring[A].zero)
        for
          h <- 0 until depth
          i <- 0 until self.rows-rows+1 by stride
          j <- 0 until self.cols-cols+1 by stride
          iʹ = i/stride
          jʹ = j/stride
          m  = iʹ*((self.cols-cols)/stride+1)+jʹ
          k <- 0 until rows
          l <- 0 until cols
          n  = k*cols+l
        do
          result(m)(n)(h) = underlying(i+k)(j+l)(h)
        Tensor[A, ((M-P)/S+1)*((N-Q)/S+1), P*Q, O](result)

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
      * convolution w/ stride
      */
    def ∗[P <: Int, Q <: Int](that: Tensor[A, P, Q, O]): Tensor[A, (M-P)/S+1, (N-Q)/S+1, O] =
      this ⋆ !that

    /**
      * max-pooling w/ stride
      */
    def max[P <: Int: ValueOf, Q <: Int: ValueOf](min: A)(using Ordering[A]): (Tensor[(Int, Int), (M-P)/S+1, (N-Q)/S+1, O],  Tensor[A, (M-P)/S+1, (N-Q)/S+1, O]) =
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

  given [A: Ring: ClassTag, M <: Int, N <: Int, O <: Int]: Conversion[Tensor[A, M, N, O], Vector[A, M*N*O]] with
    def apply(self: Tensor[A, M, N, O]): Vector[A, M*N*O] =
      given ValueOf[M*N*O] = ValueOf(self.size.asInstanceOf[M*N*O])
      Vector[A][M*N*O](self.toSeq.map(_.flatten).flatten*)

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

    def diagonalize[M <: Int, N <: Int, O <: Int](tensors: Tensor[A, M, N, O]*)[D <: Int: ValueOf]: Tensor[A, M*D, N*D, O] =
      require(valueOf[D] > 0)
      require(valueOf[D] == tensors.size)

      val depth = tensors(0).depth
      val rows = tensors(0).rows
      val cols = tensors(0).cols

      val result = Array.fill(rows * valueOf[D], cols * valueOf[D], depth)(Ring.zero)

      for
        h <- 0 until depth
        i <- 0 until rows
        j <- 0 until cols
        k <- 0 until tensors.size
      do
        result(k*rows+i)(k*cols+j)(h) = tensors(k)(i, j, h)

      Tensor[A, M*D, N*D, O](result)
