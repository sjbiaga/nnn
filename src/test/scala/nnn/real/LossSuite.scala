package nnn
package real

import munit.FunSuite

import spire.math.Real

import Loss.*


class LossSuite extends FunSuite:

  test("MSE https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example") {
    val ideal = Vector[Real][2](0.01, 0.99)
    val real = Vector[Real][2](0.75136507, 0.772928465)

    assertEquals(MSE[2]().apply(ideal, real).toDouble, 0.2983711091616805)
  }
