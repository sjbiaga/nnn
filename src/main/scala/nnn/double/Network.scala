package nnn
package double

import scala.compiletime.ops.int.+

import spire.implicits.DoubleAlgebra

import Util.given
import Network.*


case class Network[
  N <: Int: ValueOf
](loss: Loss[N],
  learningRate: Double,
  layers: Layer[N]*):

  protected given ValueOf[N+1] = ValueOf[N+1]((valueOf[N]+1).asInstanceOf[N+1])

  val N = valueOf[N]
  val M = layers.size

  val rows = 0 until N
  val cols = 0 until M

  val softmax_cross_entropy = layers(M-1).isInstanceOf[Softmax[?]]

  if softmax_cross_entropy then require(loss.isInstanceOf[Loss.CCE[N]])
  if softmax_cross_entropy then require {
    layers(M-1).neurons(0).activation match
      case Activation.Softmax | Activation.LogSoftmax => true
      case _ => false
  }

  /**
    * (bias and) weights matrices
    */
  def apply(): List[Matrix[Double, N, N+1]] =
    var r = List[Matrix[Double, N, N+1]]()
    for
      l <- cols.reverse
    do
      r ::= Matrix[Double](layers(l).neurons.flatMap { it => it.bias +: it.weights.toSeq }*)
    r

  /**
    * applies each neuron's activation function to each net output
    */
  def apply(l: Int, net: Vector[Double, N]): Vector[Double, N] =
    if softmax_cross_entropy && l == M-1
    then
      layers(M-1).neurons(0).activation.apply(net)
    else
      Vector[Double]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int, net: Vector[Double, N]): Vector[Double, N] =
    Vector[Double]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

  /**
    * train
    */
  def apply(data: Data[N], batch: Int, epochs: Int = Int.MaxValue, error: Option[Double] = None): (Int, Double) =
    require(epochs >= 0 && error.map(_ > 0).getOrElse(true) && (!error.isDefined || data.io.size == 1))

    var count = 0
    var total: Double = Double.MaxValue

    val nabla =
      for
        _ <- cols
      yield
        Matrix[Double].zero[N, N+1]

    for
      _ <- 1 to epochs
      if error.map(total > _).getOrElse(true)
      _ = count += 1
      done <- 0 until data.io.size by batch
      weights = this()
    do
      for
        (input, output) <- data.io.drop(done).take(batch)
      do
        var net = List[Vector[Double, N]]()
        var out = List(input.data.++(1))

        // FORWARD PASS

        for
          l <- cols
        do
          net ::= weights(l) ⋅ out.head
          out ::= apply(l, net.head).++(1)

        net = net.reverse
        out = out.reverse

        val y = output.answer
        val ŷ = out(M).--

        total = total min loss.apply(y, ŷ)

        // BACKPROPAGATION

        var delta =
          if softmax_cross_entropy
          then
            ŷ - y
          else
            Vector[Double](rows.map(loss.partial(y, ŷ)(_))*)
          ⊙ prime(M-1, net(M-1))

        for
          l <- cols.reverse
        do
          nabla(l) += delta ⋅ out(l)

          if l > 0
          then
            delta = (~weights(l) ⋅ delta).-- ⊙ prime(l-1, net(l-1))

      // STOCHASTIC GRADIENT DESCENT

      for
        l <- cols
      do
        val update = nabla(l).op(-learningRate * _ / data.io.size)

        for
          n <- rows
          update0 = update(n, 0)
          update1 = update(n).--
        do
          layers(l).neurons(n).bias += update0
          layers(l).neurons(n).weights += update1

        nabla(l).reset

    count -> total

  /**
    * predict
    */
  def apply(input: Input[N]): Output[N] =
    val weights = this()

    var out = input.data.++(1)

    for
      l <- cols
    do
      out = apply(l, weights(l) ⋅ out).++(1)

    Output(out.--)


object Network:

  case class Input[N <: Int](data: Vector[Double, N])

  case class Output[N <: Int](answer: Vector[Double, N])

  case class Data[N <: Int](io: (Input[N], Output[N])*)

  case class Neuron[N <: Int](weights: Vector[Double, N],
                              var bias: Double,
                              activation: Activation)

  object Neuron:

    import Initialization.*

    def apply[I <: Int: ValueOf,
              O <: Int: ValueOf](initialization: Initialization,
                                 activation: Activation): Seq[Neuron[I]] =
      (1 to valueOf[O]).map(_ => Neuron(Vector[Double][I](initialization), initialization(), activation))

    def xavier[I <: Int: ValueOf,
               O <: Int: ValueOf](activation: Activation): Seq[Neuron[I]] =
      val initialization = Xavier(valueOf[I], valueOf[O])
      (1 to valueOf[O]).map(_ => Neuron(Vector[Double][I](initialization), initialization(), activation))

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: Activation = Activation.ReLU): Seq[Neuron[I]] =
      val initialization = Kaiming(valueOf[I])
      (1 to valueOf[O]).map(_ => Neuron(Vector[Double][I](initialization), initialization(), activation))

  case class Layer[N <: Int: ValueOf](neurons: Neuron[N]*):
    require(valueOf[N] == neurons.size)

  class Softmax[N <: Int: ValueOf](initialization: Initialization,
                                   activation: Activation = Activation.Softmax)
      extends Layer[N](Neuron[N, N](initialization, activation)*)
