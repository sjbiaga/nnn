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
    val m = Matrix[Int, 2, 3](1, 2, 3,
                              4, 5, 6)

    val n = ~m

    val p = Matrix[Int, 2, 2](14, 32,
                              32, 77)

    assertEquals(m ⋅ n, p)
  }

  test("matrix ⊙ matrix") {
    val m = Matrix.constant[Int, 2, 3](2)
    val n = Matrix.constant[Int, 2, 3](3)
    val p = Matrix.constant[Int, 2, 3](6)

    assertEquals(m ⊙ n, p)
  }

  test("matrix ⋅ vector") {
    val m = Matrix[Int, 2, 3](1, 2, 3,
                              4, 5, 6)

    val n = Vector[Int, 3](7, 8, 9)

    val p = Vector[Int, 2](50, 122)

    assertEquals(m ⋅ n, p)
  }

  test("vector ⋅ matrix") {
    val n = Vector[Int, 3](7, 8, 9)

    val m = Matrix[Int, 3, 2](1, 2,
                              3, 4,
                              5, 6)

    val p = n ⋅ m

    assertEquals(p.to[Real] ⋅ ~p.to[Real], Matrix[Real, 1, 1](76*76+10000))
    assertEquals(p: Vector[Int, 2], Vector[Int, 2](76, 100))
  }
