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
    if softmax_cross_entropy && l == M-1
    then
      layers(M-1).neurons(0).activation.apply(net)
    else
      Vector[Real]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int, net: Vector[Real, N]): Vector[Real, N] =
    Vector[Real]((layers(l).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

  def train(data: Data[N], batch: Int, epochs: Int = Int.MaxValue, error: Option[Double] = None): (Int, Real) =
    require(epochs >= 0 && error.map(_ > 0).getOrElse(true) && (!error.isDefined || data.io.size == 1))

    var count = 0
    var total: Real = Double.MaxValue

    val nabla =
      for
        _ <- cols
      yield
        Matrix[Real].zero[N, N+1]

    for
      _ <- 1 to epochs
      if error.map(total.toDouble > _).getOrElse(true)
      _ = count += 1
      done <- 0 until data.io.size by batch
      weights = this()
    do
      for
        (input, output) <- data.io.drop(done).take(batch)
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

        val y = output.answer
        val ŷ = out(M).--

        total = total min loss.apply(y, ŷ)

        // BACKPROPAGATION

        var delta =
          if softmax_cross_entropy
          then
            ŷ - y
          else
            Vector[Real](rows.map(loss.partial(y, ŷ)(_))*)
          ⊙ prime(M-1, net(M-1))

        for
          l <- cols.tail.reverse
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

  def test(input: Input[N], weights: List[Matrix[Real, N, N+1]] = this()): Output[N] =
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
      this[I, O](Xavier(valueOf[I], valueOf[O]), activation)

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: Activation = Activation.ReLU): Seq[Neuron[I]] =
      this[I, O](Kaiming(valueOf[I]), activation)

  case class Layer[N <: Int: ValueOf](neurons: Neuron[N]*):
    require(valueOf[N] == neurons.size)

  class Softmax[N <: Int: ValueOf](initialization: Initialization,
                                   activation: Activation = Activation.Softmax)
      extends Layer[N](Neuron[N, N](initialization, activation)*)
