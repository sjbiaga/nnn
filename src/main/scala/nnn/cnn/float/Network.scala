package nnn
package cnn
package float

import scala.compiletime.ops.int.{ +, -, /, * }

import spire.implicits.FloatAlgebra

import nnn.float.Util.{ softmax, given }
import Image.*
import Pooling.*
import Network.*
import Layer.*


case class Network[
  FL[_ <: Int] <: Int, // Feature Length
  FB[_ <: Int] <: Int, // Feature Breadth
  KL[_ <: Int] <: Int, // Kernel Length
  KB[_ <: Int] <: Int, // Kernel Breadth
  KS[_ <: Int] <: Int, // Kernel Stride
  D[_ <: Int] <: Int, // feature/kernel Depth
  PL[_ <: Int] <: Int, // Pooling Length
  PB[_ <: Int] <: Int, // Pooling Breadth
  PS[_ <: Int] <: Int, // Pooling Stride
  K <: Int: ValueOf, // convolution/pooling layers
  N[_ <: Int] <: Int, // dense
  M <: Int: ValueOf // total layers
](loss: nnn.float.Loss[N[M]],
  learningRate: Float,
  layers: Layer*
)(using
  pattern: List[Boolean],
  volume: List[(Int, Int, Int, Int, Int, Int, Int)],
  shape: List[Int]):

  protected implicit def _valueOfFL[C <: Int](using k: Int): ValueOf[FL[C]] = ValueOf(volume(k)._1.asInstanceOf[FL[C]])
  protected implicit def _valueOfFB[C <: Int](using k: Int): ValueOf[FB[C]] = ValueOf(volume(k)._2.asInstanceOf[FB[C]])
  protected implicit def _valueOfKL[C <: Int](using k: Int): ValueOf[KL[C]] = ValueOf(volume(k)._3.asInstanceOf[KL[C]])
  protected implicit def _valueOfKB[C <: Int](using k: Int): ValueOf[KB[C]] = ValueOf(volume(k)._4.asInstanceOf[KB[C]])
  protected implicit def _valueOfKS[C <: Int](using k: Int): ValueOf[KS[C]] = ValueOf(volume(k)._5.asInstanceOf[KS[C]])
  protected implicit def _valueOfD[C <: Int](using k: Int): ValueOf[D[C]] = ValueOf(volume(k)._7.asInstanceOf[D[C]])
  protected implicit def _valueOfPL[C <: Int](using k: Int): ValueOf[PL[C]] = ValueOf(volume(k)._3.asInstanceOf[PL[C]])
  protected implicit def _valueOfPB[C <: Int](using k: Int): ValueOf[PB[C]] = ValueOf(volume(k)._4.asInstanceOf[PB[C]])
  protected implicit def _valueOfPS[C <: Int](using k: Int): ValueOf[PS[C]] = ValueOf(volume(k)._5.asInstanceOf[PS[C]])

  protected implicit def _valueOfN[L <: Int](using l: Int): ValueOf[N[L]] = ValueOf(shape(l-K).asInstanceOf[N[L]])
  protected implicit def _valueOfN1[L <: Int](using l: Long): ValueOf[N[L]+1] = ValueOf((shape(l.toInt-K)+1).asInstanceOf[N[L]+1])

  val K = valueOf[K]
  val M = valueOf[M]

  val maps = 1 to K
  def rows(using l: Int) = 0 until layers(l-1).neurons.size
  val cols = K+1 to M

  require(K+1 == pattern.size)
  require(M == layers.size)

  val softmax_cross_entropy = layers(M-1).isInstanceOf[Softmax[?]]

  if softmax_cross_entropy then require(loss.isInstanceOf[nnn.float.Loss.CCE[N[M]]])
  if softmax_cross_entropy then require {
    layers(M-1).neurons(0).activation match
      case nnn.float.Activation.Softmax | nnn.float.Activation.LogSoftmax => true
      case _ => false
  }

  for
    given Int <- maps
    l = given_Int-1
  do
    if pattern(given_Int)
    then
      {
        val P = valueOf[KL[given_Int.type]]
        val Q = valueOf[KB[given_Int.type]]
        val S = valueOf[KS[given_Int.type]]
        val Z = volume(given_Int)._6
        val MN = {
          given Int = l
          val M = valueOf[FL[given_Int.type]]
          val N = valueOf[FB[given_Int.type]]
          ((M+2*Z-P)/S+1, (N+2*Z-Q)/S+1)
        }
        require(MN._1 == valueOf[FL[given_Int.type]] && MN._2 == valueOf[FB[given_Int.type]],
                s"mismatch in the convolutional layer ${given_Int} with kernel size ${P}x${Q} stride ${S}")
      }
    else
      {
        val P = valueOf[PL[given_Int.type]]
        val Q = valueOf[PB[given_Int.type]]
        val S = valueOf[PS[given_Int.type]]
        val MND = {
          given Int = l
          val M = valueOf[FL[given_Int.type]]
          val N = valueOf[FB[given_Int.type]]
          ((M-P)/S+1, (N-Q)/S+1, valueOf[D[given_Int.type]])
        }
        require(MND._1 == valueOf[FL[given_Int.type]] && MND._2 == valueOf[FB[given_Int.type]],
                s"mismatch in the pooling layer ${given_Int} with pool size ${P}x${Q} stride ${S}")
        require(MND._3 == valueOf[D[given_Int.type]],
                s"depth mismatch in the pooling layer ${given_Int} with pool size ${P}x${Q} stride ${S}")
      }

  {
    given Int = K
    require(shape(0) == valueOf[FL[K]] * valueOf[FB[K]] * valueOf[D[K]],
            s"layer ${given_Int} flattens to ${valueOf[FL[K]] * valueOf[FB[K]] * valueOf[D[K]]} and not to ${shape(0)}")
    require(volume(K)._6 == 0,
            s"padding on last ${if pattern(given_Int) then "convolutional" else "pooling"} layer ${given_Int} must be zero")
  }

  extension [L <: Int, B <: Int, D <: Int: ValueOf](self: FeatureMap[Float, L, B, D])
    /**
      * applies an activation function to a feature map
      */
    def -->(activation: nnn.float.Activation): FeatureMap[Float, L, B, D] =
      Image[Float, L, B, D](null, self.volume.data.op(activation.apply))
    /**
      * applies the activation derivative function to a feature map
      */
    def -->>(activation: nnn.float.Activation): FeatureMap[Float, L, B, D] =
      Image[Float, L, B, D](null, self.volume.data.op(activation.prime))

  /**
    * dense (bias and) weights matrices
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

  /**
    * train
    */
  def apply(data: Data[FL[0], FB[0], D[0], N[M]], batch: Int, epochs: Int = Int.MaxValue)(blink: (Int, Int) => Unit): Unit =
    // INITIALIZATION

    var NABLA: List[Seq[Tensor[Float, ?, ?, ?]]] = Nil // C
    var NBETA: List[Vector[Float, ?]] = Nil // C|P
    var NBIAS: List[Vector[Float, ?]] = Nil // C|P
    var nabla: List[Matrix[Float, ?, ?]] = Nil // FC

    for // FULLY CONNECTED
      given Int <- cols.reverse
      l = given_Int-1
      given Long = l.toLong
    do
      nabla ::= Matrix[Float].zero[N[given_Int.type], N[l.type]+1]

    for // CONVOLUTION|POOLING
      given Int <- maps.reverse
      l = given_Int-1
      given Boolean = pattern(given_Int)
    do
      if given_Boolean
      then // CONVOLUTION
        NBETA ::= null
        NBIAS ::= Vector[Float].zero[D[given_Int.type]]
        nabla ::= null
        NABLA ::= (1 to valueOf[D[given_Int.type]]).map { _ =>
                    given ValueOf[D[l.type]] = ValueOf(volume(l)._7.asInstanceOf[D[l.type]])
                    Tensor[Float].zero[KL[given_Int.type], KB[given_Int.type], D[l.type]]
                  }
      else // POOLING
        layers(l).pooling match
          case subsampling(_, _) =>
            NBETA ::= Vector[Float].zero[D[given_Int.type]]
            NBIAS ::= Vector[Float].zero[D[given_Int.type]]
          case _ =>
            NBETA ::= null
            NBIAS ::= null
        nabla ::= null
        NABLA ::= null

    // TRAINING

    for
      count <- 0 until epochs
      done <- 0 until data.io.size by batch
      _ = blink(count, done)
      weights = this()
    do
      for
        (input, output) <- data.io.drop(done).take(batch)
      do
        // FORWARD PASS

        var NET: List[FeatureMap[Float, ?, ?, ?]] = Nil
        var OUT: List[FeatureMap[Float, ?, ?, ?]] = List(input.image)

        for // CONVOLUTION|POOLING
          given Int <- maps
          l = given_Int-1
          given Boolean = pattern(given_Int)
          padding = volume(given_Int)._6
        do
          val featureMapIn = OUT.head.asInstanceOf[FeatureMap[Float, FL[l.type], FB[l.type], D[l.type]]]
          val featureMapOut = {
            if given_Boolean
            then // CONVOLUTION
              val kernels = layers(l).kernels[KL[given_Int.type], KB[given_Int.type], D[l.type]]
              featureMapIn[KS[given_Int.type]].⋆(kernels*)[D[given_Int.type]]
            else // POOLING
              val pooling = layers(l).pooling[PL[given_Int.type], PB[given_Int.type], PS[given_Int.type]]
              pooling(featureMapIn)
          }.pad(padding).asInstanceOf[FeatureMap[Float, FL[given_Int.type], FB[given_Int.type], D[given_Int.type]]]
          NET ::= featureMapOut
          OUT ::= featureMapOut --> layers(l).activation

        NET = NET.reverse
        OUT = OUT.reverse

        val flattened = {
          given Int = K
          Vector[Float][N[K]](OUT(K).toSeq*)
        }

        var net: List[Vector[Float, ?]] = List.fill(maps.size)(null)
        var out: List[Vector[Float, ?]] = List.fill(maps.size)(null)

        out ::= flattened.++(1)

        for // FULLY CONNECTED
          given Int <- cols
          l = given_Int-1
        do
          val w = weights(l-K).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
          val x = out.head.asInstanceOf[Vector[Float, N[l.type]+1]]
          val a = w ⋅ x
          net ::= a
          out ::= apply(a).++(1)

        net = net.reverse
        out = out.reverse

        val y = output.answer
        val ŷ = out(M).asInstanceOf[Vector[Float, N[M]+1]].--

        // BACKPROPAGATION

        var DELTA: Tensor[Float, ?, ?, ?] = null
        var delta: Vector[Float, ?] =
          if softmax_cross_entropy
          then
            ŷ - y
          else {
            given Int = M
            Vector[Float][N[given_Int.type]](rows.map(loss.partial(y, ŷ)(_))*)
          ⊙ prime(given_Int)(net(M-1).asInstanceOf[Vector[Float, N[given_Int.type]]])
          }

        for // FULLY CONNECTED
          given Int <- cols.reverse
          l = given_Int-1
        do
          val ∇ = nabla(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
          val δ = delta.asInstanceOf[Vector[Float, N[given_Int.type]]]
          val h = out(l).asInstanceOf[Vector[Float, N[l.type]+1]]

          nabla(l) := ∇ + δ ⋅ h

          val w = weights(l-K).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
          if l > K
          then
            delta = (~w ⋅ δ).--
                  ⊙ prime(l)(net(l-1).asInstanceOf[Vector[Float, N[l.type]]])
          else
            given Int = K
            DELTA = (~w ⋅ δ).--.reshape[FL[given_Int.type]].reshape[D[given_Int.type]]

        for // CONVOLUTION|POOLING
          given Int <- maps.reverse
          l = given_Int-1
          given Boolean = pattern(given_Int)
          padding = volume(l)._6
        do
          val h = OUT(l).asInstanceOf[FeatureMap[Float, FL[l.type], FB[l.type], D[l.type]]]
          val a = NET(l).asInstanceOf[FeatureMap[Float, FL[given_Int.type], FB[given_Int.type], D[given_Int.type]]]
          val δ = DELTA.asInstanceOf[Tensor[Float, FL[given_Int.type], FB[given_Int.type], D[given_Int.type]]]
                ⊙ (a -->> layers(l).activation)
          if given_Boolean
          then // CONVOLUTION
            val kernels = layers(l).kernels[KL[given_Int.type], KB[given_Int.type], D[l.type]]
            { for
                d <- 0 until kernels(0).depth
              yield
                h(d)
            } match
              case h =>
                val dδ = δ[KS[given_Int.type]].dilate
                { for
                    k <- 0 until kernels.size
                  yield
                    dδ(k)
                } match
                  case δ => // weights gradient
                    for
                      k <- 0 until kernels.size
                    do
                      val ms =
                        for
                          d <- 0 until kernels(0).depth
                        yield
                          type P = FL[given_Int.type] + (FL[given_Int.type] - 1) * (KS[given_Int.type] - 1)
                          type Q = FB[given_Int.type] + (FB[given_Int.type] - 1) * (KS[given_Int.type] - 1)
                          h(d).⋆[P, Q](δ(k))
                      { val ∇ = NABLA(l)(k).asInstanceOf[Tensor[Float, KL[given_Int.type], KB[given_Int.type], D[l.type]]]
                        val ∂ = { given Int = l; Tensor[Float].stack(ms*)[D[given_Int.type]] }.asInstanceOf[Tensor[Float, KL[given_Int.type], KB[given_Int.type], D[l.type]]]
                        NABLA(l)(k) := ∇ + ∂
                      }
                      { val ∇ = NBIAS(l).asInstanceOf[Vector[Float, D[given_Int.type]]]
                        NBIAS(l)(k) = ∇(k) + δ(k).sum
                      }
            if l > 0
            then
              val pdδ = δ[KS[given_Int.type]].pad_and_dilate[KL[given_Int.type], KB[given_Int.type]]
              { for
                  k <- 0 until kernels.size
                yield
                  pdδ(k)
              } match
                case δ => // input gradient
                  val ms =
                    for
                      d <- 0 until kernels(0).depth
                    yield {
                      for
                        k <- 0 until kernels.size
                      yield
                        δ(k).∗[KL[given_Int.type], KB[given_Int.type]](kernels(k)(d))
                    }.reduce(_ + _)
                  if padding == 0
                  then
                    DELTA = { given Int = l; Tensor[Float].stack(ms*)[D[given_Int.type]] }
                  else
                    type P = padding.type
                    given ValueOf[P] = ValueOf(padding)
                    DELTA = { given Int = l; Tensor[Float].stack(ms*)[D[given_Int.type]] }.crop[P, P]
          else // POOLING
            type P = (FL[l.type] - PL[given_Int.type]) / PS[given_Int.type] + 1
            type Q = (FB[l.type] - PB[given_Int.type]) / PS[given_Int.type] + 1
            val pa = a.asInstanceOf[FeatureMap[Float, P, Q, D[l.type]]]
            val pδ = δ.asInstanceOf[Tensor[Float, P, Q, D[l.type]]]
            val pooling = layers(l).pooling[PL[given_Int.type], PB[given_Int.type], PS[given_Int.type]]
            pooling match
              case subsampling(_, _) =>
                { for
                    k <- 0 until valueOf[D[given_Int.type]]
                  yield
                    pδ(k)
                } match
                  case δ =>
                    { val ∇ = NBETA(l).asInstanceOf[Vector[Float, D[given_Int.type]]]
                      val avg = pa.volume.asInstanceOf[SubsamplingVolume[Float, FL[l.type], FB[l.type], D[l.type], PL[given_Int.type], PB[given_Int.type], PS[given_Int.type]]].avg
                      for
                        k <- 0 until valueOf[D[given_Int.type]]
                      do
                        NBETA(l)(k) = ∇(k) + (δ(k) ⊙ avg(k)).sum
                    }
                    { val ∇ = NBIAS(l).asInstanceOf[Vector[Float, D[given_Int.type]]]
                      for
                        k <- 0 until valueOf[D[given_Int.type]]
                      do
                        NBIAS(l)(k) = ∇(k) + δ(k).sum
                    }
              case _ =>
            if l > 0
            then
              if padding == 0
              then
                DELTA = pooling(h, pa, pδ)
              else
                type P = padding.type
                given ValueOf[P] = ValueOf(padding)
                DELTA = pooling(h, pa, pδ).crop[P, P]

      // GRADIENT DESCENT

      for // CONVOLUTION|POOLING
        given Int <- maps
        l = given_Int-1
        given Boolean = pattern(given_Int)
      do
        if given_Boolean
        then // CONVOLUTION
          for
            k <- 0 until layers(l).kernels.size
          do
            val ∇ = NABLA(l)(k).asInstanceOf[Tensor[Float, KL[given_Int.type], KB[given_Int.type], D[l.type]]]
            val update = ∇.op(-learningRate * _ / batch)
            val weights = layers(l).kernels[KL[given_Int.type], KB[given_Int.type], D[l.type]](k).volume.data
            layers(l).kernels(k).volume.data := weights + update
          val ∇ = NBIAS(l).asInstanceOf[Vector[Float, D[given_Int.type]]]
          val update = ∇.op(-learningRate * _ / batch)
          for
            k <- 0 until layers(l).kernels.size
          do
            layers(l).kernels(k).bias += update(k)
          NBIAS(l).reset
          NABLA(l).foreach(_.reset)
        else // POOLING
          layers(l).pooling match
            case ss @ subsampling(β: Vector[Float, D[given_Int.type]], b: Vector[Float, D[given_Int.type]]) =>
              {
                val ∇ = NBETA(l).asInstanceOf[Vector[Float, D[given_Int.type]]]
                val update = ∇.op(-learningRate * _ / batch)
                ss.beta := β + update
              }
              {
                val ∇ = NBIAS(l).asInstanceOf[Vector[Float, D[given_Int.type]]]
                val update = ∇.op(-learningRate * _ / batch)
                ss.bias := b + update
              }
              NBETA(l).reset
              NBIAS(l).reset
            case _ =>

      for // FULLY CONNECTED
        given Int <- cols
        l = given_Int-1
      do
        val ∇ = nabla(l).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
        val update = ∇.op(-learningRate * _ / batch)
        for
          n <- rows
          update0 = update(n)(0)
          update1 = update(n).--
        do
          val weights = layers(l).neurons(n).weights.asInstanceOf[Vector[Float, N[l.type]]]
          layers(l).neurons(n).bias += update0
          layers(l).neurons(n).weights := weights + update1

        nabla(l).reset

  /**
    * predict
    */
  def apply(input: Input[FL[0], FB[0], D[0]]): Output[N[M]] =
    var OUT: FeatureMap[Float, ?, ?, ?] = input.image

    for // CONVOLUTION|POOLING
      given Int <- maps
      l = given_Int-1
      given Boolean = pattern(given_Int)
      padding = volume(given_Int)._6
    do
      val featureMapIn = OUT.asInstanceOf[FeatureMap[Float, FL[l.type], FB[l.type], D[l.type]]]
      val featureMapOut = {
        if given_Boolean
        then // CONVOLUTION
          val kernels = layers(l).kernels[KL[given_Int.type], KB[given_Int.type], D[l.type]]
          featureMapIn[KS[given_Int.type]].⋆(kernels*)[D[given_Int.type]]
        else // POOLING
          val pooling = layers(l).pooling[PL[given_Int.type], PB[given_Int.type], PS[given_Int.type]]
          pooling(featureMapIn)
      }.pad(padding).asInstanceOf[FeatureMap[Float, FL[given_Int.type], FB[given_Int.type], D[given_Int.type]]]
      OUT = featureMapOut --> layers(l).activation

    val flattened = {
      given Int = K
      Vector[Float][N[K]](OUT.toSeq*)
    }

    val weights = this()

    var out: Vector[Float, ?] = flattened.++(1)

    for // FULLY CONNECTED
      given Int <- cols
      l = given_Int-1
    do
      val w = weights(l-K).asInstanceOf[Matrix[Float, N[given_Int.type], N[l.type]+1]]
      val x = out.asInstanceOf[Vector[Float, N[l.type]+1]]
      out = apply(w ⋅ x).++(1)

    Output(out.asInstanceOf[Vector[Float, N[M]+1]].--)


object Network:

  case class Input[
    L <: Int,
    B <: Int,
    D <: Int
  ](image: Image[Float, L, B, D])

  case class Output[N <: Int](answer: Vector[Float, N])

  class OneHotOutput[N <: Int: ValueOf](label: Int)
      extends Output(Vector[Float] { case `label` => 1 case _ => 0 })

  case class Data[
    L <: Int,
    B <: Int,
    D <: Int,
    Y <: Int
  ](io: (Input[L, B, D], Output[Y])*)

  case class Neuron[N <: Int](weights: Vector[Float, N],
                              var bias: Float,
                              activation: nnn.float.Activation)

  object Neuron:

    import Initialization.*

    case class bias(bias: Float = 0):

      def apply[I <: Int: ValueOf,
                O <: Int: ValueOf](initialization: Initialization,
                                   activation: nnn.float.Activation): Seq[Neuron[I]] =
        (1 to valueOf[O]).map(_ => Neuron(Vector[Float][I](initialization), bias, activation))

    def apply[I <: Int: ValueOf,
              O <: Int: ValueOf](initialization: Initialization,
                                 activation: nnn.float.Activation): Seq[Neuron[I]] =
      (1 to valueOf[O]).map(_ => Neuron(Vector[Float][I](initialization), initialization(), activation))

    def xavier[I <: Int: ValueOf,
               O <: Int: ValueOf](activation: nnn.float.Activation): Seq[Neuron[I]] =
      this[I, O](Xavier(valueOf[I], valueOf[O]), activation)

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: nnn.float.Activation = nnn.float.Activation.ReLU): Seq[Neuron[I]] =
      this[I, O](Kaiming(valueOf[I]), activation)

  sealed trait Layer

  object Layer:

    case class Convolution[
      L <: Int,
      B <: Int,
      D <: Int,
      O <: Int: ValueOf
    ](activation: nnn.float.Activation,
      kernels: Kernel[Float, L, B, D]*) extends Layer:
      require(valueOf[O] == kernels.size)

    case class Pool[
      P <: Int,
      Q <: Int,
      S <: Int: ValueOf
    ](pooling: Pooling[Float, P, Q, S],
      activation: nnn.float.Activation = nnn.float.Activation.Linear()) extends Layer

    case class Dense[N <: Int, O <: Int: ValueOf](neurons: Neuron[N]*) extends Layer:
      require(valueOf[O] == neurons.size)

    class Softmax[N <: Int: ValueOf](initialization: Initialization,
                                     activation: nnn.float.Activation = nnn.float.Activation.Softmax)
        extends Dense[N, N](Neuron[N, N](initialization, activation)*):
      this.neurons.foreach(_.bias = 0)

    extension (self: Layer)
      def pooling[P <: Int, Q <: Int, S <: Int]: Pooling[Float, P, Q, S] = self.asInstanceOf[Pool[P, Q, S]].pooling
      def activation(using p: Boolean): nnn.float.Activation =
        if p
        then
          self.asInstanceOf[Convolution[?, ?, ?, ?]].activation
        else
          self.asInstanceOf[Pool[?, ?, ?]].activation
      def kernels[L <: Int, B <: Int, D <: Int]: Seq[Kernel[Float, L, B, D]] = self.asInstanceOf[Convolution[L, B, D, ?]].kernels
      def neurons: Seq[Neuron[?]] = self.asInstanceOf[Dense[?, ?]].neurons
