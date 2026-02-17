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

  test("matrix flatten") {
    val m = Matrix[Int][3, 3]( 1, 3,  5,
                              -4, 0, -3,
                               5, 7, -9)

    val n = Vector[Int][9](1, 3, 5, -4, 0, -3, 5, 7, -9)

    assertEquals(-m, n)
  }

  test("vector reshape") {
    val n = Vector[Int][9](1, 3, 5, -4, 0, -3, 5, 7, -9)

    val m = Matrix[Int][3, 3]( 1, 3,  5,
                              -4, 0, -3,
                               5, 7, -9)

    assertEquals(n.reshape[3], m)
  }

  test("matrix reshape") {
    val m = Matrix[Int][2, 6](1, 2,  3,  4,   5,  6,
                              7, 8,  9, 10,  11, 12)

    val n = Tensor[Int][2, 3, 2](1, 2,  3,  4,   5,  6,
                                 7, 8,  9, 10,  11, 12)

    assertEquals(m.reshape[2], n)
  }
