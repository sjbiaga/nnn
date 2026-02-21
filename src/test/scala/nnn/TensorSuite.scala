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

  test("dilation tensor w/ stride 2") {
    val m = Tensor[Int][3, 3, 1](18, 21, 16,
                                 14, 21, 13,
                                 16,  7, 23)

    val n = Tensor[Int][5, 5, 1](18, 0, 21, 0, 16,
                                 0,  0, 0 , 0,  0,
                                 14, 0, 21, 0, 13,
                                 0,  0, 0 , 0,  0,
                                 16, 0,  7, 0, 23)

    assertEquals(m[2].dilate, n)
  }

  test("padding and dilation tensor w/ stride 2") {
    val m = Tensor[Int][3, 3, 1](18, 21, 16,
                                 14, 21, 13,
                                 16,  7, 23)

    val n = Tensor[Int][9, 9, 1](0, 0,  0, 0,  0, 0,  0, 0, 0,
                                 0, 0,  0, 0,  0, 0,  0, 0, 0,
                                 0, 0, 18, 0, 21, 0, 16, 0, 0,
                                 0, 0, 0,  0, 0 , 0,  0, 0, 0,
                                 0, 0, 14, 0, 21, 0, 13, 0, 0,
                                 0, 0, 0,  0, 0 , 0,  0, 0, 0,
                                 0, 0, 16, 0,  7, 0, 23, 0, 0,
                                 0, 0,  0, 0,  0, 0,  0, 0, 0,
                                 0, 0,  0, 0,  0, 0,  0, 0, 0)

    assertEquals(m[2].pad_and_dilate[3, 3], n)
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

  test("tensor stack matrices") {
    val m = Matrix[Int][2, 3](1, 3,  5,
                              7, 9, 11)

    val n = Matrix[Int][2, 3](2,  4,  6,
                              8, 10, 12)

    val p = Tensor[Int][2, 3, 2](1, 2,  3,  4,   5,  6,
                                 7, 8,  9, 10,  11, 12)

    assertEquals(Tensor[Int].stack(m, n)[2], p)
  }
