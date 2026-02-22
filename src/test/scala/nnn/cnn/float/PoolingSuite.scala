package nnn
package cnn
package float

import spire.implicits.*

import munit.FunSuite

import Image.*
import Pooling.*


class PoolingSuite extends FunSuite:

  test("subsampling gradient distribution w/ stride 3") {
    val h = Image[Float, 6, 6, 1](null, Tensor[Float][6, 6, 1](6, 3, 4, 4, 5, 0,
                                                               4, 7, 4, 0, 4, 0,
                                                               7, 0, 2, 3, 4, 5,
                                                               3, 7, 5, 0, 3, 0,
                                                               5, 8, 1, 2, 5, 4,
                                                               6, 4, 1, 3, 0, 4))

    val p = subsampling[Float, 3, 3, 3](Vector[Float][1](9), Vector[Float][1](0))

    val a = p.apply(h)

    val δ = Tensor[Float][2, 2, 1](1, 2,
                                   4, 5)

    val r = Tensor[Float][6, 6, 1](1, 1, 1, 2, 2, 2,
                                   1, 1, 1, 2, 2, 2,
                                   1, 1, 1, 2, 2, 2,
                                   4, 4, 4, 5, 5, 5,
                                   4, 4, 4, 5, 5, 5,
                                   4, 4, 4, 5, 5, 5)

    assertEquals(p(h, a, δ), r)
  }

  test("max gradient distribution w/ stride 3") {
    val h = Image[Float, 6, 6, 1](null, Tensor[Float][6, 6, 1](6, 3, 4, 4, 5, 0,
                                                               4, 7, 4, 0, 4, 0,
                                                               7, 0, 2, 3, 4, 5,
                                                               3, 7, 5, 0, 3, 0,
                                                               5, 8, 1, 2, 5, 4,
                                                               6, 4, 1, 3, 0, 4))

    val p = max[Float, 3, 3, 3](Float.MinValue)

    val a = p.apply(h)

    val δ = Tensor[Float][2, 2, 1](1, 2,
                                   4, 5)

    val r = Tensor[Float][6, 6, 1](0, 0, 0, 0, 2, 0,
                                   0, 1, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 4, 0, 0, 5, 0,
                                   0, 0, 0, 0, 0, 0)

    assertEquals(p(h, a, δ), r)
  }
