package nnn

import scala.reflect.ClassTag

import munit.FunSuite

import spire.math.*
import spire.algebra.*
import spire.implicits.*

import real.Util.given
import Vector.given
import Matrix.*


class MatrixSuite extends FunSuite:

  test("matrix ⋅ ~matrix") {
    val m = Matrix[Int][2, 3](1, 2, 3,
                              4, 5, 6)

    val n = ~m

    val p = Matrix[Int][2, 2](14, 32,
                              32, 77)

    assertEquals(m ⋅ n, p)
  }

  test("matrix ⊙ matrix") {
    val m = Matrix[Int].constant[2, 3](2)
    val n = Matrix[Int].constant[2, 3](3)
    val p = Matrix[Int].constant[2, 3](6)

    assertEquals(m ⊙ n, p)
  }

  test("matrix ⋅ vector") {
    val m = Matrix[Int][2, 3](1, 2, 3,
                              4, 5, 6)

    val n = Vector[Int][3](7, 8, 9)

    val p = Vector[Int][2](50, 122)

    assertEquals(m ⋅ n, p)
  }

  test("vector ⋅ matrix") {
    val n = Vector[Int][3](7, 8, 9)

    val m = Matrix[Int][3, 2](1, 2,
                              3, 4,
                              5, 6)

    val p = n ⋅ m

    assertEquals(p.to[Real] ⋅ ~p.to[Real], Matrix[Real][1, 1](76*76+10000))
    assertEquals(p: Vector[Int, 2], Vector[Int][2](76, 100))
  }

  test("matrix ⋆ matrix") {
    val m = Matrix[Int][3, 4](1, 5, 0, 1,
                              0, 1, 3, 6,
                              5, 4, 2, 1)

    val n = Matrix[Int][2, 2](2, 3,
                              4, 1)

    val p = Matrix[Int][2, 3](18, 17, 21,
                              27, 29, 33)

    assertEquals(m ⋆ n, p)
  }

  test("!matrix") {
    val m = Matrix[Int][2, 4](1, 2, 3, 4,
                              5, 6, 7, 8)

    val n = Matrix[Int][2, 4](8, 7, 6, 5,
                              4, 3, 2, 1)

    assertEquals(!m, n)
  }

  test("matrix ∗ matrix") {
    val m = Matrix[Int][3, 4](1, 5, 0, 1,
                              0, 1, 3, 6,
                              5, 4, 2, 1)

    val n = Matrix[Int][2, 2](2, 3,
                              4, 1)

    val p = Matrix[Int][2, 3](23, 14, 25,
                              27, 29, 35)

    assertEquals(m ∗ n, p)
  }

  test("matrix ⋆ matrix") {
    val m = Matrix[Int][7, 7](6, 3, 4, 4, 5, 0, 3,
                              4, 7, 4, 0, 4, 0, 4,
                              7, 0, 2, 3, 4, 5, 2,
                              3, 7, 5, 0, 3, 0, 7,
                              5, 8, 1, 2, 5, 4, 2,
                              8, 0, 1, 0, 6, 0, 0,
                              6, 4, 1, 3, 0, 4, 5)

    val n = Matrix[Int][3, 3](1, 0, 1,
                              1, 0, 0,
                              0, 0, 2)

    val p = Matrix[Int][5, 5](18, 20, 21, 14, 16,
                              25,  7, 16,  3, 26,
                              14, 14, 21, 16, 13,
                              15, 15, 21,  2, 15,
                              16, 16,  7, 14, 23)

    assertEquals(m ⋆ n, p)
  }

  test("matrix ⋆ matrix w/ stride 2") {
    val m = Matrix[Int][7, 7](6, 3, 4, 4, 5, 0, 3,
                              4, 7, 4, 0, 4, 0, 4,
                              7, 0, 2, 3, 4, 5, 2,
                              3, 7, 5, 0, 3, 0, 7,
                              5, 8, 1, 2, 5, 4, 2,
                              8, 0, 1, 0, 6, 0, 0,
                              6, 4, 1, 3, 0, 4, 5)

    val n = Matrix[Int][3, 3](1, 0, 1,
                              1, 0, 0,
                              0, 0, 2)

    val p = Matrix[Int][3, 3](18, 21, 16,
                              14, 21, 13,
                              16,  7, 23)

    assertEquals(m[2] ⋆ n, p)
  }

  test("matrix max-pooling") {
    val m = Matrix[Int][2, 3](23, 14, 25,
                              27, 29, 35)

    val n = Matrix[Int][1, 2](29, 35)

    assertEquals(m.max[2, 2](Int.MinValue), n)
  }

  test("matrix max-pooling w/ stride 1") {
    val m = Matrix[Int][7, 7](6, 3, 4, 4, 5, 0, 3,
                              4, 7, 4, 0, 4, 0, 4,
                              7, 0, 2, 3, 4, 5, 2,
                              3, 7, 5, 0, 3, 0, 7,
                              5, 8, 1, 2, 5, 4, 2,
                              8, 0, 1, 0, 6, 0, 0,
                              6, 4, 1, 3, 0, 4, 5)

    val n = Matrix[Int][5, 5](7, 7, 5, 5, 5,
                              7, 7, 5, 5, 7,
                              8, 8, 5, 5, 7,
                              8, 8, 6, 6, 7,
                              8, 8, 6, 6, 6)

    assertEquals(m[1].max[3, 3](Int.MinValue), n)
  }

  test("matrix max-pooling w/ stride 2") {
    val m = Matrix[Int][7, 7](6, 3, 4, 4, 5, 0, 3,
                              4, 7, 4, 0, 4, 0, 4,
                              7, 0, 2, 3, 4, 5, 2,
                              3, 7, 5, 0, 3, 0, 7,
                              5, 8, 1, 2, 5, 4, 2,
                              8, 0, 1, 0, 6, 0, 0,
                              6, 4, 1, 3, 0, 4, 5)

    val n = Matrix[Int][3, 3](7, 5, 5,
                              8, 5, 7,
                              8, 6, 6)

    assertEquals(m[2].max[3, 3](Int.MinValue), n)
  }

  test("matrix average-pooling w/ stride 2") {
    val m = Matrix[Float][7, 7](6, 3, 4, 4, 5, 0, 3,
                                4, 7, 4, 0, 4, 0, 4,
                                7, 0, 2, 3, 4, 5, 2,
                                3, 7, 5, 0, 3, 0, 7,
                                5, 8, 1, 2, 5, 4, 2,
                                8, 0, 1, 0, 6, 0, 0,
                                6, 4, 1, 3, 0, 4, 5)

    val n = Matrix[Float][3, 3](4.1111110f, 3.3333333f, 3.0000000f,
                                4.2222223f, 2.7777777f, 3.5555556f,
                                3.7777777f, 2.1111112f, 2.8888888f)

    assertEquals(m[2].avg[3, 3], n)
  }

  test("matrix flatten") {
    val m = Matrix[Int][3, 3]( 1, 3,  5,
                              -4, 0, -3,
                               5, 7, -9)

    val n = Vector[Int][9](1, 3, 5, -4, 0, -3, 5, 7, -9)

    assertEquals(-m, n)
  }
