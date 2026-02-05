package nnn
package float

import scala.compiletime.ops.int.+

import spire.implicits.FloatAlgebra

import Util.given
import Network.*


case class Network[
  N[_ <: Int] <: Int,
  M <: Int: ValueOf
](loss: Loss[N[M]],
  learningRate: Float,
  layers: Layer[?]*)
  (using shape: List[Int]):
  require(valueOf[M] == layers.size)

  // given_Int is the current layer (L) in which each neuron has shape(given_Long.toInt) weights (plus bias)
  protected implicit def _valueOf[L <: Int](using l: Int): ValueOf[N[L]] = ValueOf(shape(l).asInstanceOf[N[L]])
  // shape(given_Long.toInt) is the number of neurons in the previous layer (L)
  protected implicit def _valueOf1[L <: Int](using l: Long): ValueOf[N[L]+1] = ValueOf((shape(l.toInt)+1).asInstanceOf[N[L]+1])
  // given_Long.toInt == given_Int-1, although given_Int comes first and given_Long comes second

  val M = valueOf[M]

  def rows(using l: Int) = 0 until layers(l-1).neurons.size
  val cols = 1 to M

  /**
    * (bias and) weights matrices
    */
  def apply(): List[Matrix[Float, ?, ?]] =
    var r = List[Matrix[Float, ?, ?]]()
    for
      given Int <- cols.reverse
      l = given_Int-1
      given Long = l.toLong
    do
      val ws = layers(l).neurons.flatMap { it => it.bias +: it.weights.toSeq }
      r ::= Matrix[Float][N[given_Int.type], N[l.type]+1](ws*)
    r

  /**
    * applies each neuron's activation function to each net output
    */
  def apply(using l: Int)(net: Vector[Float, N[l.type]]): Vector[Float, N[l.type]] =
    Vector[Float]((layers(l-1).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int)(net: Vector[Float, N[l.type]]): Vector[Float, N[l.type]] =
    given Int = l
    Vector[Float]((layers(l-1).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

  /**
    * train
    */
  def apply(data: Data[N[0], N[M]], epochs: Long = Long.MaxValue, error: Option[Float] = None): (Long, Float) =
    require(epochs >= 0 && error.map(_ > 0).getOrElse(true) && (!error.isDefined || data.io.size == 1))

    var count = 0L
    var total: Float = Float.MaxValue

    for
      _ <- 1L to epochs
      if error.map(total > _).getOrElse(true)
    do
      count += 1

      val weights = this()

      var nabla: List[Matrix[Float, ?, ?]] = Nil

      for
        given Int <- cols.reverse
        l = given_Int-1
        given Long = l.toLong
      do
        nabla ::= Matrix[Float].zero[N[given_Int.type], N[l.type]+1]

      for
        (input, output) <- data.io
      do
        var net: List[Vector[Float, ?]] = Nil
        var out: List[Vector[Float, ?]] = List(input.data.++(1))

        // FORWARD PASS

        for
          given Int <- cols
          l = given_Int-1
          given Long = l.toLong
        do
          val w = weights(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
          val x = out.head.asInstanceOf[Vector[Float, N[l.type]+1]]
          val a = w ⋅ x
          net ::= a
          out ::= apply(a).++(1)

        net = net.reverse
        out = out.reverse

        val y = out(M).asInstanceOf[Vector[Float, N[M]+1]].--

        total = total min loss.apply(output.answer, y)

        // BACKPROPAGATION

        var delta: Vector[Float, ?] = {
          given Int = M
          Vector[Float][N[given_Int.type]](rows.map(loss.partial(output.answer, y)(_))*)
        ⊙ prime(given_Int)(net(M-1).asInstanceOf[Vector[Float, N[given_Int.type]]])
        }

        for
          given Int <- cols.tail.reverse
          l = given_Int-1
          given Long = l.toLong
        do
          val ∇ = nabla(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
          val δ = delta.asInstanceOf[Vector[Float, N[given_Int.type]]]
          val h = out(l).asInstanceOf[Vector[Float, N[l.type]+1]]

          nabla(l) := ∇ + δ ⋅ h

          val w = weights(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
          delta = (~w ⋅ δ).-- ⊙ prime(l)(net(l-1).asInstanceOf[Vector[Float, N[l.type]]])

        {
          given Int = 1
          given Long = 0L

          val ∇ = nabla(0).asInstanceOf[Matrix[Float, N[given_Int.type], N[0]+1]]
          val δ = delta.asInstanceOf[Vector[Float, N[given_Int.type]]]
          val x = out(0).asInstanceOf[Vector[Float, N[0]+1]]

          nabla(0) := ∇ + δ ⋅ x
        }

      // GRADIENT DESCENT

      for
        given Int <- cols
        l = given_Int-1
        given Long = l.toLong
      do
        val ∇ = nabla(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
        val update = ∇.op(-learningRate * _ / data.io.size)

        for
          n <- rows
          update0 = update(n)(0)
          update1 = update(n).--
        do
          val w = layers(l).neurons(n).weights.asInstanceOf[Vector[Float, N[l.type]]]
          layers(l).neurons(n).bias += update0
          layers(l).neurons(n).weights := w + update1

    count -> total

  /**
    * predict
    */
  def apply(input: Input[N[0]]): Output[N[M]] =
    val weights = this()

    var out: Vector[Float, ?] = input.data.++(1)

    for
      given Int <- cols
      l = given_Int-1
      given Long = l.toLong
    do
      val w = weights(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
      val x = out.asInstanceOf[Vector[Float, N[l.type]+1]]
      out = apply(w ⋅ x).++(1)

    Output(out.asInstanceOf[Vector[Float, N[M]+1]].--)


object Network:

  case class Input[N <: Int](data: Vector[Float, N])

  case class Output[N <: Int](answer: Vector[Float, N])

  case class Data[X <: Int, Y <: Int](io: (Input[X], Output[Y])*)

  case class Neuron[N <: Int](weights: Vector[Float, N],
                              var bias: Float,
                              activation: Activation)

  object Neuron:

    import Initialization.*

    def apply[I <: Int: ValueOf,
              O <: Int: ValueOf](initialization: Initialization,
                                 activation: Activation): Seq[Neuron[I]] =
      (1 to valueOf[O]).map(_ => Neuron(Vector[Float][I](initialization), initialization(), activation))

    def xavier[I <: Int: ValueOf,
               O <: Int: ValueOf](activation: Activation): Seq[Neuron[I]] =
      val initialization = Xavier(valueOf[I], valueOf[O])
      (1 to valueOf[O]).map(_ => Neuron(Vector[Float][I](initialization), initialization(), activation))

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: Activation = Activation.ReLU): Seq[Neuron[I]] =
      val initialization = Kaiming(valueOf[I])
      (1 to valueOf[O]).map(_ => Neuron(Vector[Float][I](initialization), initialization(), activation))

  case class Layer[N <: Int](neurons: Neuron[N]*):
    require(neurons.nonEmpty)
