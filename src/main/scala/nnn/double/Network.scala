package nnn
package double

import spire.implicits.*

import Util.given
import Network.*


case class Network[
  N <: Int: ValueOf,
  N1 <: Int: ValueOf
](loss: Loss[N],
  learningRate: Double,
  layers: HiddenLayer[N]*):
  require(valueOf[N1] == 1+valueOf[N])

  val N = valueOf[N]
  val L = layers.size

  def rows = 0 until N
  def cols = 0 until L

  /**
    * (bias and) weights matrices
    */
  def apply(): List[Matrix[Double, N, N1]] =
    var r = List[Matrix[Double, N, N1]]()
    for
      l <- cols.reverse
    do
      r ::= Matrix[Double, N, N1](layers(l).neurons.flatMap { it => it.bias +: it.weights.toSeq }*)
    r

  /**
    * applies each neuron's activation function to each net output
    */
  def apply(l: Int, net: Vector[Double, N]): Vector[Double, N] =
    Vector[Double, N]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int, net: Vector[Double, N]): Vector[Double, N] =
    Vector[Double, N]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

  /**
    * train
    */
  def apply(data: Data[N], epochs: Long = Long.MaxValue, error: Option[Double] = None): (Long, Double) =
    require(epochs >= 0 && error.map(_ > 0).getOrElse(true) && (!error.isDefined || data.io.size == 1))

    var count = 0L
    var total: Double = Double.MaxValue

    for
      _ <- 1L to epochs
      if error.map(total > _).getOrElse(true)
    do
      count += 1

      val weights = this()

      var nabla = List.fill(cols.size)(Matrix.zero[Double, N, N1])

      for
        io <- data.io
      do
        var net = List[Vector[Double, N]]()
        var out = List(io._1.data.++[N1](1))

        // FORWARD PASS

        for
          l <- cols
        do
          net ::= weights(l) ⋅ out.head
          out ::= apply(l, net.head).++(1)

        net = net.reverse
        out = out.reverse

        total = total min loss.apply(io._2.answer, out(L).--)

        var delta = Vector[Double, N](rows.map(loss.partial(io._2.answer, out(L).--)(_))*)
                  ⊙ prime(L-1, net(L-1))

        // BACKPROPAGATION

        for
          l <- cols.tail.reverse
        do
          nabla(l) += delta ⋅ out(l)

          delta = (~weights(l) ⋅ delta).-- ⊙ prime(l-1, net(l-1))

        nabla(0) += delta ⋅ out(0)

      // UPDATE

      for
        l <- cols
      do
        val update = nabla(l).op(_ / data.io.size).op(-learningRate * _)

        for
          i <- rows
        do
          val (update0, update1) = (update(i)(0), update(i).--[N])
          layers(l).neurons(i).bias += update0
          layers(l).neurons(i).weights += update1

    count -> total

  /**
    * predict
    */
  def apply(input: Input[N]): Output[N] =
    val weights = this()

    var out = input.data.++[N1](1)

    for
      l <- cols
    do
      out = apply(l, weights(l) ⋅ out).++[N1](1)

    Output(out.--[N])


object Network:

  case class Input[
    N <: Int: ValueOf
  ](data: Vector[Double, N])

  case class Output[
    N <: Int: ValueOf
  ](answer: Vector[Double, N])

  case class Data[
    N <: Int: ValueOf
  ](io: (Input[N], Output[N])*)

  case class Neuron[
    N <: Int: ValueOf
  ](weights: Vector[Double, N],
    var bias: Double,
    activation: Activation)

  case class HiddenLayer[
    N <: Int: ValueOf
  ](neurons: Neuron[N]*):
    require(neurons.nonEmpty)
