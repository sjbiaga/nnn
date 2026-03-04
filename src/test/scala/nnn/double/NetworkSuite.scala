package nnn
package double

import munit.FunSuite

import spire.implicits.*

import Activation.*
import Initialization.Gaussian
import Loss.*
import Util.{ truncate, given }
import Network.*


class NetworkSuite extends FunSuite:

  test("NN https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example") {

    val nn = Network[2](
      loss = MSE[2](),
      learningRate = 0.5,
      Layer[2](
        Neuron(Vector[Double](.15, .20), .35, Sigmoid),
        Neuron(Vector[Double](.25, .30), .35, Sigmoid),
      ),
      Layer[2](
        Neuron(Vector[Double](.40, .45), .60, Sigmoid),
        Neuron(Vector[Double](.50, .55), .60, Sigmoid),
      )
    )

    assertEquals(nn(), List(Matrix[Double][2, 3](.35, .15, .20, .35, .25, .30), Matrix[Double][2, 3](.60, .40, .45, .60, .50, .55)))

    val data = Data[2](Input(Vector[Double][2](.05, .10)) -> Output(Vector[Double][2](.01, .99)))

    val (epochs, error) = nn.train(data, 1, 10000)

    assert(epochs == 10000 && error < 1E-5)

    val answer = nn.test(Input(Vector[Double][2](.05, .10))).answer

    assertEquals(answer.truncate(2), Vector[Double][2](.01, .98))

  }

  test("XOR https://github.com/lexesj/java-toy-neural-network") {

    val xor = Network[10](
      loss = MSE[10](),
      learningRate = 3,
      Layer[10](Neuron[10, 10](Gaussian(), Sigmoid)*),
      Layer[10]((Neuron[10, 1](Gaussian(), Sigmoid) ++ Neuron[10, 9](Gaussian(), Sign))*)
    )

    val data = Data[10](Input(Vector[Double][10](0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) -> Output(Vector[Double][10](0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
                        Input(Vector[Double][10](0, 1, 0, 0, 0, 0, 0, 0, 0, 0)) -> Output(Vector[Double][10](1, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
                        Input(Vector[Double][10](1, 0, 0, 0, 0, 0, 0, 0, 0, 0)) -> Output(Vector[Double][10](1, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
                        Input(Vector[Double][10](1, 1, 0, 0, 0, 0, 0, 0, 0, 0)) -> Output(Vector[Double][10](0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    )

    xor.train(data, 1, 10000)

    val weights = xor()

    val answer00 = xor.test(Input(Vector[Double][10](0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), weights).answer
    val answer01 = xor.test(Input(Vector[Double][10](0, 1, 0, 0, 0, 0, 0, 0, 0, 0)), weights).answer
    val answer10 = xor.test(Input(Vector[Double][10](1, 0, 0, 0, 0, 0, 0, 0, 0, 0)), weights).answer
    val answer11 = xor.test(Input(Vector[Double][10](1, 1, 0, 0, 0, 0, 0, 0, 0, 0)), weights).answer

    assertEquals(answer00.truncate(1)(0), .0d)
    assertEquals(answer01.truncate(1)(0), .9d)
    assertEquals(answer10.truncate(1)(0), .9d)
    assertEquals(answer11.truncate(1)(0), .0d)

  }
