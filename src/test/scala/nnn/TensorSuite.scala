package nnn

import scala.reflect.ClassTag

import munit.FunSuite

import spire.math.*
import spire.algebra.*
import spire.implicits.*

import real.Util.given
import Matrix.given
import Tensor.*


class TensorSuite extends FunSuite:

  test("tensor ⋅ ~tensor") {
    val m = Tensor[Int][2, 3, 1](1, 2, 3,
                                 4, 5, 6)

    val n = ~m

    val p = Tensor[Int][2, 2, 1](14, 32,
                                 32, 77)

    assertEquals(m ⋅ n, p)
  }

  test("tensor ⋆ tensor") {
    val m = Tensor[Int][3, 4, 1](1, 5, 0, 1,
                                 0, 1, 3, 6,
                                 5, 4, 2, 1)

    val n = Tensor[Int][2, 2, 1](2, 3,
                                 4, 1)

    val p = Tensor[Int][2, 3, 1](18, 17, 21,
                                 27, 29, 33)

    assertEquals(m ⋆ n, p)
  }

  test("!tensor") {
    val m = Tensor[Int][2, 4, 1](1, 2, 3, 4,
                                 5, 6, 7, 8)

    val n = Tensor[Int][2, 4, 1](8, 7, 6, 5,
                                 4, 3, 2, 1)

    assertEquals(!m, n)
  }

  test("tensor ∗ tensor") {
    val m = Tensor[Int][3, 4, 1](1, 5, 0, 1,
                                 0, 1, 3, 6,
                                 5, 4, 2, 1)

    val n = Tensor[Int][2, 2, 1](2, 3,
                                 4, 1)

    val p = Tensor[Int][2, 3, 1](23, 14, 25,
                                 27, 29, 35)

    assertEquals(m ∗ n, p)
  }

  test("tensor ⋆ tensor") {
    val m = Tensor[Int][7, 7, 1](6, 3, 4, 4, 5, 0, 3,
                                 4, 7, 4, 0, 4, 0, 4,
                                 7, 0, 2, 3, 4, 5, 2,
                                 3, 7, 5, 0, 3, 0, 7,
                                 5, 8, 1, 2, 5, 4, 2,
                                 8, 0, 1, 0, 6, 0, 0,
                                 6, 4, 1, 3, 0, 4, 5)

    val n = Tensor[Int][3, 3, 1](1, 0, 1,
                                 1, 0, 0,
                                 0, 0, 2)

    val p = Tensor[Int][5, 5, 1](18, 20, 21, 14, 16,
                                 25,  7, 16,  3, 26,
                                 14, 14, 21, 16, 13,
                                 15, 15, 21,  2, 15,
                                 16, 16,  7, 14, 23)

    assertEquals(m ⋆ n, p)
  }

  test("tensor ⋆ tensor w/ stride 2") {
    val m = Tensor[Int][7, 7, 1](6, 3, 4, 4, 5, 0, 3,
                                 4, 7, 4, 0, 4, 0, 4,
                                 7, 0, 2, 3, 4, 5, 2,
                                 3, 7, 5, 0, 3, 0, 7,
                                 5, 8, 1, 2, 5, 4, 2,
                                 8, 0, 1, 0, 6, 0, 0,
                                 6, 4, 1, 3, 0, 4, 5)

    val n = Tensor[Int][3, 3, 1](1, 0, 1,
                                 1, 0, 0,
                                 0, 0, 2)

    val p = Tensor[Int][3, 3, 1](18, 21, 16,
                                 14, 21, 13,
                                 16,  7, 23)

    assertEquals(m[2] ⋆ n, p)
  }

  test("tensor max-pooling") {
    val m = Tensor[Int][2, 3, 1](23, 14, 25,
                                 27, 29, 35)

    val max = Tensor[(Int, Int)][1, 2, 1]((1, 1), (1, 2))
    val n = Tensor[Int][1, 2, 1](29, 35)

    assertEquals(m.max[2, 2](Int.MinValue), max->n)
  }

  test("tensor max-pooling w/ stride 1") {
    val m = Tensor[Int][7, 7, 1](6, 3, 4, 4, 5, 0, 3,
                                 4, 7, 4, 0, 4, 0, 4,
                                 7, 0, 2, 3, 4, 5, 2,
                                 3, 7, 5, 0, 3, 0, 7,
                                 5, 8, 1, 2, 5, 4, 2,
                                 8, 0, 1, 0, 6, 0, 0,
                                 6, 4, 1, 3, 0, 4, 5)

    val max = Tensor[(Int, Int)][5, 5, 1]((1,1), (1,1), (0,4), (0,4), (0,4),
                                          (1,1), (1,1), (3,2), (2,5), (3,6),
                                          (4,1), (4,1), (3,2), (2,5), (3,6),
                                          (4,1), (4,1), (5,4), (5,4), (3,6),
                                          (4,1), (4,1), (5,4), (5,4), (5,4))

    val n = Tensor[Int][5, 5, 1](7, 7, 5, 5, 5,
                                 7, 7, 5, 5, 7,
                                 8, 8, 5, 5, 7,
                                 8, 8, 6, 6, 7,
                                 8, 8, 6, 6, 6)

    assertEquals(m[1].max[3, 3](Int.MinValue), max->n)
  }

  test("tensor max-pooling w/ stride 2") {
    val m = Tensor[Int][7, 7, 1](6, 3, 4, 4, 5, 0, 3,
                                 4, 7, 4, 0, 4, 0, 4,
                                 7, 0, 2, 3, 4, 5, 2,
                                 3, 7, 5, 0, 3, 0, 7,
                                 5, 8, 1, 2, 5, 4, 2,
                                 8, 0, 1, 0, 6, 0, 0,
                                 6, 4, 1, 3, 0, 4, 5)

    val max = Tensor[(Int, Int)][3, 3, 1]((1,1), (0,4), (0,4),
                                          (4,1), (3,2), (3,6),
                                          (4,1), (5,4), (5,4))

    val n = Tensor[Int][3, 3, 1](7, 5, 5,
                                 8, 5, 7,
                                 8, 6, 6)

    assertEquals(m[2].max[3, 3](Int.MinValue), max->n)
  }

  test("tensor average-pooling w/ stride 2") {
    val m = Tensor[Float][7, 7, 1](6, 3, 4, 4, 5, 0, 3,
                                   4, 7, 4, 0, 4, 0, 4,
                                   7, 0, 2, 3, 4, 5, 2,
                                   3, 7, 5, 0, 3, 0, 7,
                                   5, 8, 1, 2, 5, 4, 2,
                                   8, 0, 1, 0, 6, 0, 0,
                                   6, 4, 1, 3, 0, 4, 5)

    val n = Tensor[Float][3, 3, 1](4.1111110f, 3.3333333f, 3.0000000f,
                                   4.2222223f, 2.7777777f, 3.5555556f,
                                   3.7777777f, 2.1111112f, 2.8888888f)

    assertEquals(m[2].avg[3, 3], n)
  }

  test("tensor flatten") {
    val m = Seq[Int](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    val n = Tensor[Int][2, 3, 2](1, 2,  3,  4,   5,  6,
                                 7, 8,  9, 10,  11, 12)

    assertEquals(n.toSeq.map(_.flatten).flatten, m)
  }

  test("tensor reshape") {
    val m = Vector[Int][12](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    val n = Tensor[Int][2, 3, 2](1, 2,  3,  4,   5,  6,
                                 7, 8,  9, 10,  11, 12)

    assertEquals(m.reshape[2].reshape[2], n)
  }

  test("diagonalize matrices") {
    val m = Tensor[Int][4, 3, 1](6, 3, 4,
                                 4, 7, 4,
                                 7, 0, 2,
                                 3, 7, 5)

    val n = Tensor[Int][4, 3, 1](3, 7, 5,
                                 5, 8, 1,
                                 8, 0, 1,
                                 6, 4, 1)

    val p = Tensor[Int][4, 3, 1](0, 3, 0,
                                 2, 5, 4,
                                 8, 0, 1,
                                 3, 0, 4)

    val r = Tensor[Int][12, 9, 1](6, 3, 4, 0, 0, 0, 0, 0, 0,
                                  4, 7, 4, 0, 0, 0, 0, 0, 0,
                                  7, 0, 2, 0, 0, 0, 0, 0, 0,
                                  3, 7, 5, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 3, 7, 5, 0, 0, 0,
                                  0, 0, 0, 5, 8, 1, 0, 0, 0,
                                  0, 0, 0, 8, 0, 1, 0, 0, 0,
                                  0, 0, 0, 6, 4, 1, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 3, 0,
                                  0, 0, 0, 0, 0, 0, 2, 5, 4,
                                  0, 0, 0, 0, 0, 0, 8, 0, 1,
                                  0, 0, 0, 0, 0, 0, 3, 0, 4)

    assertEquals(Tensor[Int].diagonalize(m, n, p)[3], r)
  }

  test("tensor ʹ.⋆ tensor w/ stride 1") {
    val m = Tensor[Int][4, 4, 1](6, 3, 4, 4,
                                 4, 7, 4, 0,
                                 7, 0, 2, 3,
                                 3, 7, 5, 0)

    val n = Tensor[Int][9, 4, 1](6, 3, 4, 7,
                                 3, 4, 7, 4,
                                 4, 4, 4, 0,
                                 4, 7, 7, 0,
                                 7, 4, 0, 2,
                                 4, 0, 2, 3,
                                 7, 0, 3, 7,
                                 0, 2, 7, 5,
                                 2, 3, 5, 0)

    assertEquals(m[1].ʹ.⋆[2, 2], n)
  }

  test("tensor ʹ.⋆ tensor w/ stride 2") {
    val m = Tensor[Int][7, 7, 1](6, 3, 4, 4, 5, 0, 3,
                                 4, 7, 4, 0, 4, 0, 4,
                                 7, 0, 2, 3, 4, 5, 2,
                                 3, 7, 5, 0, 3, 0, 7,
                                 5, 8, 1, 2, 5, 4, 2,
                                 8, 0, 1, 0, 6, 0, 0,
                                 6, 4, 1, 3, 0, 4, 5)

    val n = Tensor[Int][9, 9, 1](6, 3, 4, 4, 7, 4, 7, 0, 2,
                                 4, 4, 5, 4, 0, 4, 2, 3, 4,
                                 5, 0, 3, 4, 0, 4, 4, 5, 2,
                                 7, 0, 2, 3, 7, 5, 5, 8, 1,
                                 2, 3, 4, 5, 0, 3, 1, 2, 5,
                                 4, 5, 2, 3, 0, 7, 5, 4, 2,
                                 5, 8, 1, 8, 0, 1, 6, 4, 1,
                                 1, 2, 5, 1, 0, 6, 1, 3, 0,
                                 5, 4, 2, 6, 0, 0, 0, 4, 5)

    assertEquals(m[2].ʹ.⋆[3, 3], n)
  }
