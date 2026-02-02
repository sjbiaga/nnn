package nnn
package real

import munit.FunSuite

import spire.math.Real
import spire.implicits.*

import Activation.*
import Loss.*
import Util.given
import Network.*


class NetworkSuite extends FunSuite:

  test("NN https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example") {

    val nn = Network[2](
      loss = MSE[2](),
      learningRate = 0.5,
      Layer[2](
        Neuron[2](Vector[Real, 2](.15, .20), .35, Sigmoid),
        Neuron[2](Vector[Real, 2](.25, .30), .35, Sigmoid),
      ),
      Layer[2](
        Neuron[2](Vector[Real, 2](.40, .45), .60, Sigmoid),
        Neuron[2](Vector[Real, 2](.50, .55), .60, Sigmoid),
      )
    )

    assertEquals(nn().map(_.to[Double]), List(Matrix[Double, 2, 3](.35, .15, .20, .35, .25, .30), Matrix[Double, 2, 3](.60, .40, .45, .60, .50, .55)))

    val data = Data[2](Input(Vector[Real, 2](.05, .10)) -> Output(Vector[Real, 2](.01, .99)))

    val (epochs, error) = nn(data, 10)

    println(s"Training took #$epochs epochs and ended with error ${error.toDouble}")
    println(nn(Input(Vector[Real, 2](.05, .10))).answer.to[Double])

  }
