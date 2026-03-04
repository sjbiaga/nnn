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
  parametersSGD: (Float, Float, Float),
  layers: Layer[?, ?]*
)(using
  shape: List[Int]
):
  val (learningRate, momentumCoefficient, decayFactor) = parametersSGD

  // given_Int is the current layer (L) in which each neuron has shape(given_Long.toInt) weights (plus bias)
  protected implicit def valueOfN[L <: Int](using l: Int): ValueOf[N[L]] = ValueOf(shape(l).asInstanceOf[N[L]])
  // shape(given_Int-1) is the number of neurons in the previous layer (L)
  protected implicit def valueOfN1[L <: Int](using l: Int): ValueOf[N[L]+1] = ValueOf((shape(l-1)+1).asInstanceOf[N[L]+1])

  val M = valueOf[M]

  def rows(using l: Int) = 0 until layers(l-1).neurons.size
  val cols = 1 to M

  require(valueOf[M] == layers.size)

  val softmax_cross_entropy = layers(M-1).isInstanceOf[Softmax[?]]

  if softmax_cross_entropy then require(loss.isInstanceOf[Loss.CCE[N[M]]])
  if softmax_cross_entropy then require {
    layers(M-1).neurons(0).activation match
      case Activation.Softmax | Activation.LogSoftmax => true
      case _ => false
  }

  /**
    * (bias and) weights matrices
    */
  def apply(): List[Matrix[Float, ?, ?]] =
    var r = List[Matrix[Float, ?, ?]]()
    for
      given Int <- cols.reverse
      l = given_Int-1
    do
      val ws = layers(l).neurons.flatMap { it => it.bias +: it.weights.toSeq }
      r ::= Matrix[Float][N[given_Int.type], N[l.type]+1](ws*)
    r

  /**
    * applies each neuron's activation function to each net output
    */
  def apply(using l: Int)(net: Vector[Float, N[l.type]]): Vector[Float, N[l.type]] =
    if softmax_cross_entropy && l == M
    then
      layers(M-1).neurons(0).activation.apply(net)
    else
      Vector[Float]((layers(l-1).neurons.map(_.activation) zip net.toSeq).map(_.apply(_))*)

  /**
    * applies each neuron's activation derivative function to each net output
    */
  def prime(l: Int)(net: Vector[Float, N[l.type]]): Vector[Float, N[l.type]] =
    given Int = l
    Vector[Float]((layers(l-1).neurons.map(_.activation) zip net.toSeq).map(_.prime(_))*)

  def train(data: Data[N[0], N[M]], batch: Int, epochs: Int = Int.MaxValue, error: Option[Float] = None): (Int, Float) =
    require(epochs >= 0 && error.map(_ > 0).getOrElse(true) && (!error.isDefined || data.io.size == 1))

    var count = 0
    var total: Float = Float.MaxValue

    val nabla =
      for
        given Int <- cols
        l = given_Int-1
      yield
        Matrix[Float].zero[N[given_Int.type], N[l.type]+1]

    val velocity =
      if decayFactor > 0
      then
        for
          given Int <- cols.reverse
          l = given_Int-1
        yield
          Matrix[Float].zero[N[given_Int.type], N[l.type]+1]
      else
        Nil

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
        var net: List[Vector[Float, ?]] = Nil
        var out: List[Vector[Float, ?]] = List(input.data.++(1))

        // FORWARD PASS

        for
          given Int <- cols
          l = given_Int-1
        do
          val w = weights(l).shaped[N[given_Int.type], N[l.type]+1]
          val x = out.head.shaped[N[l.type]+1]
          val a = w ⋅ x
          net ::= a
          out ::= apply(a).++(1)

        net = net.reverse
        out = out.reverse

        val y = output.answer
        val ŷ = out(M).shaped[N[M]+1].--

        total = total min loss.apply(y, ŷ)

        // BACKPROPAGATION

        var delta: Vector[Float, ?] =
          if softmax_cross_entropy
          then
            ŷ - y
          else {
            given Int = M
            Vector[Float][N[given_Int.type]](rows.map(loss.partial(y, ŷ)(_))*)
          ⊙ prime(given_Int)(net(M-1).shaped[N])
          }

        for
          given Int <- cols.reverse
          l = given_Int-1
        do
          val ∇ = nabla(l).shaped[N[given_Int.type], N[l.type]+1]
          val δ = delta.shaped[N]
          val h = out(l).shaped[N[l.type]+1]

          nabla(l) := ∇ + δ ⋅ h

          if l > 0
          then
            val w = weights(l).shaped[N[given_Int.type], N[l.type]+1]
            delta = (~w ⋅ δ).--
                  ⊙ prime(l)(net(l-1).shaped[N](using l))

      if decayFactor > 0
      then // MOMENTUM GRADIENT DESCENT

        for
          given Int <- cols
          l = given_Int-1
        do
          val ∇ = nabla(l).shaped[N[given_Int.type], N[l.type]+1]
          val v = velocity(l).shaped[N[given_Int.type], N[l.type]+1]
          val update = ∇.op(-learningRate * _ / batch)
          for
            n <- rows
          do
            var weights = layers(l).neurons(n).weights.shaped[N[l.type]].++(layers(l).neurons(n).bias)
            v.velocityUpdate(n)(momentumCoefficient, decayFactor, learningRate)(weights, update)
            weights = weights + v(n)
            layers(l).neurons(n).bias = weights(0)
            layers(l).neurons(n).weights := weights.--

          nabla(l).reset

      else // STOCHASTIC GRADIENT DESCENT

        for
          given Int <- cols
          l = given_Int-1
        do
          val ∇ = nabla(l).shaped[N[given_Int.type], N[l.type]+1]
          val update = ∇.op(-learningRate * _ / data.io.size)

          for
            n <- rows
            update0 = update(n, 0)
            update1 = update(n).--
          do
            val weights = layers(l).neurons(n).weights.shaped[N](using l)
            layers(l).neurons(n).bias += update0
            layers(l).neurons(n).weights := weights + update1

          nabla(l).reset

    count -> total

  def test(input: Input[N[0]], weights: List[Matrix[Float, ?, ?]] = this()): Output[N[M]] =
    var out: Vector[Float, ?] = input.data.++(1)

    for
      given Int <- cols
      l = given_Int-1
    do
      val w = weights(l).shaped[N[given_Int.type], N[l.type]+1]
      val x = out.shaped[N[l.type]+1]
      out = apply(w ⋅ x).++(1)

    Output(out.shaped[N[M]+1].--)


object Network:

  extension [M <: Int, N <: Int](self: Matrix[Float, M, N])
    def velocityUpdate(m: Int)(momentumCoefficient: Float, decayFactor: Float, learningRate: Float)(weights: Vector[Float, N], update: Matrix[Float, M, N]): Unit =
      for
        n <- 0 until self.cols
      do
        self(m->n) = momentumCoefficient*self(m, n) - decayFactor*learningRate*weights(n) - update(m, n)

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
      this[I, O](Xavier(valueOf[I], valueOf[O]), activation)

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: Activation = Activation.ReLU): Seq[Neuron[I]] =
      this[I, O](Kaiming(valueOf[I]), activation)

  case class Layer[N <: Int, O <: Int: ValueOf](neurons: Neuron[N]*):
    require(valueOf[O] == neurons.size)

  class Softmax[N <: Int: ValueOf](initialization: Initialization,
                                   activation: Activation = Activation.Softmax)
      extends Layer[N, N](Neuron[N, N](initialization, activation)*)
