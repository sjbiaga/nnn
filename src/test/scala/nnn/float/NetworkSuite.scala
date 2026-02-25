package nnn
package float

import scala.compiletime.ops.int.-

import munit.FunSuite

import spire.implicits.*

import Activation.*
import Initialization.Gaussian
import Loss.*
import Util.{ truncate, given }
import Network.*


class NetworkSuite extends FunSuite:

  test("NN https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example") {

    def D[L <: Int](layer: Layer[?, ?])
                   (using p: ValueOf[N[L-1]], n: ValueOf[N[L]]): layer.type =
      require(n.value == layer.neurons.size)
      require(layer.neurons.forall { it => it.weights.size == p.value })
      layer

    type N[L <: Int] = L match { case 0 => 2 case 1 => 2 case 2 => 2 }

    given List[Int] = 2 :: 2 :: 2 :: Nil

    val nn = Network[N, 2](
      loss = MSE[2](),
      learningRate = 0.5,
      D[1] {
        Layer[2, 2](
          Neuron(Vector[Float](.15, .20), .35, Sigmoid),
          Neuron(Vector[Float](.25, .30), .35, Sigmoid),
        )
      },
      D[2] {
        Layer[2, 2](
          Neuron(Vector[Float](.40, .45), .60, Sigmoid),
          Neuron(Vector[Float](.50, .55), .60, Sigmoid),
        )
      }
    )

    assertEquals(nn(), List(Matrix[Float][2, 3](.35, .15, .20, .35, .25, .30), Matrix[Float][2, 3](.60, .40, .45, .60, .50, .55)))

    val data = Data[2, 2](Input(Vector[Float][2](.05, .10)) -> Output(Vector[Float][2](.01, .99)))

    val (epochs, error) = nn(data, 1, 10000)

    assert(epochs == 10000 && error < 1E-5)

    val answer = nn(Input[2](Vector[Float](.05, .10))).answer

    assertEquals(answer.truncate(2), Vector[Float][2](.01, .98))

  }

  test("XOR https://github.com/lexesj/java-toy-neural-network") {

    def D[L <: Int](layer: Layer[?, ?])
                   (using p: ValueOf[N[L-1]], n: ValueOf[N[L]]): layer.type =
      require(n.value == layer.neurons.size)
      require(layer.neurons.forall { it => it.weights.size == p.value })
      layer

    type N[L <: Int] = L match { case 0 => 2 case 1 => 10 case 2 => 1 }

    given List[Int] = 2 :: 10 :: 1 :: Nil

    val xor = Network[N, 2](
      loss = MSE[1](),
      learningRate = 3,
      D[1](Layer[2, 10](Neuron[2, 10](Gaussian(), Sigmoid)*)),
      D[2](Layer[10, 1](Neuron[10, 1](Gaussian(), Sigmoid)*))
    )

    val data = Data[2, 1](Input(Vector[Float][2](0, 0)) -> Output(Vector[Float][1](0)),
                          Input(Vector[Float][2](0, 1)) -> Output(Vector[Float][1](1)),
                          Input(Vector[Float][2](1, 0)) -> Output(Vector[Float][1](1)),
                          Input(Vector[Float][2](1, 1)) -> Output(Vector[Float][1](0)),
    )

    xor(data, 1, 10000)

    val answer00 = xor(Input(Vector[Float][2](0, 0))).answer
    val answer01 = xor(Input(Vector[Float][2](0, 1))).answer
    val answer10 = xor(Input(Vector[Float][2](1, 0))).answer
    val answer11 = xor(Input(Vector[Float][2](1, 1))).answer

    assertEquals(answer00.truncate(2)(0), .00f)
    assertEquals(answer01.truncate(2)(0), .99f)
    assertEquals(answer10.truncate(2)(0), .99f)
    assertEquals(answer11.truncate(2)(0), .00f)

  }
