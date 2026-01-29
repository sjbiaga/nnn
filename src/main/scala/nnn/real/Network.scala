package nnn
package real

import spire.math.Real
import spire.implicits.*

import Util.given
import Network.*


case class Network[
  N <: Int: ValueOf,
  N1 <: Int: ValueOf
](loss: Loss[N],
  learningRate: Real,
  layers: HiddenLayer[N]*):
  require(valueOf[N1] == 1+valueOf[N])

  val N = valueOf[N]
  val L = layers.size

  def rows = 0 until N
  def cols = 0 until L

  /**
    * (bias and) weights matrices
    */
  def apply(): List[Matrix[Real, N, N1]] =
    var r = List[Matrix[Real, N, N1]]()
    for
      l <- cols.reverse
    do
      r ::= Matrix[Real, N, N1](layers(l).neurons.flatMap { it => Real(it.bias.toDouble) +: it.weights.to[Double].to[Real].toSeq }*)
    r

  /**
    * applies each neuron's activation function to each net output
    */
  def apply(l: Int, net: Vector[Real, N]): Vector[Real, N] =
    Vector[Real, N]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int, net: Vector[Real, N]): Vector[Real, N] =
    Vector[Real, N]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

  /**
    * train
    */
  def apply(data: Data[N], epochs: Long = Long.MaxValue, error: Option[Double] = None): (Long, Real) =
    require(epochs >= 0 && error.map(_ > 0).getOrElse(true) && (!error.isDefined || data.io.size == 1))

    var count = 0L
    var total: Real = Double.MaxValue

    for
      _ <- 1L to epochs
      if error.map(total.toDouble > _).getOrElse(true)
    do
      count += 1

      val weights = this()

      var nabla = List.fill(cols.size)(Matrix.zero[Real, N, N1])

      for
        io <- data.io
      do
        var net = List[Vector[Real, N]]()
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

        var delta = Vector[Real, N](rows.map(loss.partial(io._2.answer, out(L).--)(_))*)
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
  ](data: Vector[Real, N])

  case class Output[
    N <: Int: ValueOf
  ](answer: Vector[Real, N])

  case class Data[
    N <: Int: ValueOf
  ](io: (Input[N], Output[N])*)

  case class Neuron[
    N <: Int: ValueOf
  ](weights: Vector[Real, N],
    var bias: Real,
    activation: Activation)

  case class HiddenLayer[
    N <: Int: ValueOf
  ](neurons: Neuron[N]*):
    require(neurons.nonEmpty)
