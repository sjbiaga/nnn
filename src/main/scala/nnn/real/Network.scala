package nnn
package real

import scala.compiletime.ops.int.+

import spire.math.Real
import spire.implicits.*

import Util.given
import Network.*


case class Network[
  N <: Int: ValueOf
](loss: Loss[N],
  learningRate: Real,
  layers: Layer[N]*):

  protected given ValueOf[N+1] = ValueOf[N+1]((valueOf[N]+1).asInstanceOf[N+1])

  val N = valueOf[N]
  val L = layers.size

  val rows = 0 until N
  val cols = 0 until L

  /**
    * (bias and) weights matrices
    */
  def apply(): List[Matrix[Real, N, N+1]] =
    var r = List[Matrix[Real, N, N+1]]()
    for
      l <- cols.reverse
    do
      r ::= Matrix[Real](layers(l).neurons.flatMap { it => Real(it.bias.toDouble) +: it.weights.to[Double].to[Real].toSeq }*)
    r

  /**
    * applies each neuron's activation function to each net output
    */
  def apply(l: Int, net: Vector[Real, N]): Vector[Real, N] =
    Vector[Real]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int, net: Vector[Real, N]): Vector[Real, N] =
    Vector[Real]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

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

      var nabla = List.fill(cols.size)(Matrix[Real].zero[N, N+1])

      for
        (input, output) <- data.io
      do
        var net = List[Vector[Real, N]]()
        var out = List(input.data.++(1))

        // FORWARD PASS

        for
          l <- cols
        do
          net ::= weights(l) ⋅ out.head
          out ::= apply(l, net.head).++(1)

        net = net.reverse
        out = out.reverse

        val y = out(L).--

        total = total min loss.apply(output.answer, y)

        // BACKPROPAGATION

        var delta = Vector[Real](rows.map(loss.partial(output.answer, y)(_))*)
                  ⊙ prime(L-1, net(L-1))

        for
          l <- cols.tail.reverse
        do
          nabla(l) += delta ⋅ out(l)

          delta = (~weights(l) ⋅ delta).-- ⊙ prime(l-1, net(l-1))

        nabla(0) += delta ⋅ out(0)

      // GRADIENT DESCENT

      for
        l <- cols
      do
        val update = nabla(l).op(-learningRate * _ / data.io.size)

        for
          n <- rows
          update0 = update(n)(0)
          update1 = update(n).--
        do
          layers(l).neurons(n).bias += update0
          layers(l).neurons(n).weights += update1

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

  case class Input[N <: Int](data: Vector[Real, N])

  case class Output[N <: Int](answer: Vector[Real, N])

  case class Data[N <: Int](io: (Input[N], Output[N])*)

  case class Neuron[N <: Int](weights: Vector[Real, N],
                              var bias: Real,
                              activation: Activation)

  object Neuron:

    import Initialization.*

    def apply[I <: Int: ValueOf,
              O <: Int: ValueOf](initialization: Initialization,
                                 activation: Activation): Seq[Neuron[I]] =
      (1 to valueOf[O]).map(_ => Neuron(Vector[Real][I](initialization), initialization(), activation))

    def xavier[I <: Int: ValueOf,
               O <: Int: ValueOf](activation: Activation): Seq[Neuron[I]] =
      val initialization = Xavier(valueOf[I], valueOf[O])
      (1 to valueOf[O]).map(_ => Neuron(Vector[Real][I](initialization), initialization(), activation))

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: Activation = Activation.ReLU): Seq[Neuron[I]] =
      val initialization = Kaiming(valueOf[I])
      (1 to valueOf[O]).map(_ => Neuron(Vector[Real][I](initialization), initialization(), activation))

  case class Layer[N <: Int](neurons: Neuron[N]*):
    require(neurons.nonEmpty)
