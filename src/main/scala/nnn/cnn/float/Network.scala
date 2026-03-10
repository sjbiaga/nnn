package nnn
package cnn
package float

import scala.compiletime.ops.int.{ +, -, /, * }

import spire.math.pow
import spire.implicits.FloatAlgebra

import nnn.float.Util.{ dropout, softmax, given }
import Image.*
import Volume.{ *, given }
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
  parametersSGD: (Float, Float, Float),
  layers: Layer*
)(using
  volume: List[(Int, Int, Int, Int, Int, Int, Int)],
  shape: List[Int]
):
  val (learningRate, momentumCoefficient, decayFactor) = parametersSGD

  protected implicit def valueOfFL[C <: Int](using c: Int): ValueOf[FL[C]] = ValueOf(volume(c)._1.asInstanceOf[FL[C]])
  protected implicit def valueOfFB[C <: Int](using c: Int): ValueOf[FB[C]] = ValueOf(volume(c)._2.asInstanceOf[FB[C]])
  protected implicit def valueOfKL[C <: Int](using c: Int): ValueOf[KL[C]] = ValueOf(volume(c)._3.asInstanceOf[KL[C]])
  protected implicit def valueOfKB[C <: Int](using c: Int): ValueOf[KB[C]] = ValueOf(volume(c)._4.asInstanceOf[KB[C]])
  protected implicit def valueOfKS[C <: Int](using c: Int): ValueOf[KS[C]] = ValueOf(volume(c)._5.asInstanceOf[KS[C]])
  protected implicit def valueOfD[C <: Int](using c: Int): ValueOf[D[C]] = ValueOf(volume(c)._7.asInstanceOf[D[C]])
  protected implicit def valueOfPL[C <: Int](using c: Int): ValueOf[PL[C]] = ValueOf(volume(c)._3.asInstanceOf[PL[C]])
  protected implicit def valueOfPB[C <: Int](using c: Int): ValueOf[PB[C]] = ValueOf(volume(c)._4.asInstanceOf[PB[C]])
  protected implicit def valueOfPS[C <: Int](using c: Int): ValueOf[PS[C]] = ValueOf(volume(c)._5.asInstanceOf[PS[C]])

  protected implicit def valueOfN[L <: Int](using l: Int): ValueOf[N[L]] = ValueOf(shape(l-K).asInstanceOf[N[L]])
  protected implicit def valueOfN1[L <: Int](using l: Int): ValueOf[N[L]+1] = ValueOf((shape(l-K-1)+1).asInstanceOf[N[L]+1])

  val K = valueOf[K]
  val M = valueOf[M]

  val maps = 1 to K
  def rows(using l: Int) = 0 until layers(l-1).neurons.size
  val cols = K+1 to M

  require(M == layers.size)
  require(!layers(0).isInstanceOf[Pooling[?, ?, ?]])
  require(!layers(M-1).isInstanceOf[Dropout[?, ?]])

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
    if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
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
  }

  extension [L <: Int, B <: Int, D <: Int](self: FeatureMap[Float, L, B, D])
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

  def train(data: Data[FL[0], FB[0], D[0], N[M]], batch: Int, epochs: Int = Int.MaxValue)(blink: (Int, Int) => Unit): Unit =
    // INITIALIZATION

    var NABLA: List[Seq[Tensor[Float, ?, ?, ?]]] = Nil // C
    var NBETA: List[Vector[Float, ?]] = Nil // C|P
    var NBIAS: List[Vector[Float, ?]] = Nil // C|P
    var nabla: List[Matrix[Float, ?, ?]] = Nil // FC

    var VELOCITY: List[Seq[Tensor[Float, ?, ?, ?]]] = Nil // C
    var VBETA: List[Vector[Float, ?]] = Nil // C|P
    var VBIAS: List[Vector[Float, ?]] = Nil // C|P
    var velocity: List[Matrix[Float, ?, ?]] = Nil // FC

    for // FULLY CONNECTED
      given Int <- cols.reverse
      l = given_Int-1
    do
      nabla ::= Matrix[Float].zero[N[given_Int.type], N[l.type]+1]

    for // CONVOLUTION|POOLING
      given Int <- maps.reverse
      l = given_Int-1
    do
      if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
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

    // VELOCITY

    if decayFactor > 0
    then

      for // FULLY CONNECTED
        given Int <- cols.reverse
        l = given_Int-1
      do
        velocity ::= Matrix[Float].zero[N[given_Int.type], N[l.type]+1]

      for // CONVOLUTION|POOLING
        given Int <- maps.reverse
        l = given_Int-1
      do
        if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
        then // CONVOLUTION
          VBETA ::= null
          VBIAS ::= Vector[Float].zero[D[given_Int.type]]
          velocity ::= null
          VELOCITY ::= (1 to valueOf[D[given_Int.type]]).map { _ =>
                      given ValueOf[D[l.type]] = ValueOf(volume(l)._7.asInstanceOf[D[l.type]])
                      Tensor[Float].zero[KL[given_Int.type], KB[given_Int.type], D[l.type]]
                    }
        else // POOLING
          layers(l).pooling match
            case subsampling(_, _) =>
              VBETA ::= Vector[Float].zero[D[given_Int.type]]
              VBIAS ::= Vector[Float].zero[D[given_Int.type]]
            case _ =>
              VBETA ::= null
              VBIAS ::= null
          velocity ::= null
          VELOCITY ::= null

    // TRAINING

    for
      count <- 0 until epochs
      _ = if count > 0 then blink(count-1, data.io.size) else ()
      done <- 0 until data.io.size by batch
      _ = blink(count, done)
      weights = this()
    do
      for
        (input, output) <- data.io.drop(done).take(batch)
      do
        // FORWARD PASS

        var PAD: List[FeatureMap[Float, ?, ?, ?]] = Nil
        var NET: List[FeatureMap[Float, ?, ?, ?]] = Nil
        var OUT: List[FeatureMap[Float, ?, ?, ?]] = List(input.image)
        var LRN: List[(Tensor[Float, ?, ?, ?], Image[Float, ?, ?, ?])] = List(null)

        for // CONVOLUTION|POOLING
          given Int <- maps
          l = given_Int-1
          padding = volume(given_Int)._6
        do
          val featureMapIn = {
            if l > 0 && layers(l-1).isInstanceOf[ConvolutionalLRN[?, ?, ?, ?]]
            then
              LRN.head._2.shaped[FL, FB, D](using l)
            else
              OUT.head.shaped[FL, FB, D](using l)
          }.pad(padding)
          PAD ::= featureMapIn
          var featureMapOut = {
            if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
            then // CONVOLUTION
              val kernels = layers(l).kernels[KL, KB, D[l.type]]
              featureMapIn[KS[given_Int.type]].⋆(kernels*)[D[given_Int.type]]
            else // POOLING
              val pooling = layers(l).pooling[PL, PB, PS]
              pooling(featureMapIn)
          }.shaped[FL, FB, D]
          NET ::= featureMapOut
          featureMapOut = featureMapOut --> layers(l).activation
          OUT ::= featureMapOut
          layers(l) match
            case lrn: ConvolutionalLRN[?, ?, ?, ?] =>
              LRN ::= featureMapOut.lrn(lrn.k, lrn.n, lrn.α, lrn.β)()
            case _ =>
              LRN ::= null

        PAD = PAD.reverse
        NET = NET.reverse
        OUT = OUT.reverse
        LRN = LRN.reverse

        val flattened = {
          given Int = K
          Vector[Float][N[K]](OUT(K).toSeq*)
        }

        var net: List[Vector[Float, ?]] = List.fill(maps.size)(null)
        var out: List[Vector[Float, ?]] = List.fill(maps.size)(null)
        var mask: List[Vector[Float, ?]] = List.fill(maps.size)(null)

        out ::= flattened.++(1)

        for // FULLY CONNECTED
          given Int <- cols
          l = given_Int-1
        do
          val w = weights(l-K).shaped[N[given_Int.type], N[l.type]+1]
          val x = out.head.shaped[N[l.type]+1]
          val a = w ⋅ x
          net ::= a
          layers(l) match
            case d: Layer.Dropout[?, ?] =>
              val m = dropout[N[given_Int.type]](d.keep)
              mask ::= m
              out ::= (apply(a) ⊙ m).++(1)
            case _ =>
              mask ::= null
              out ::= apply(a).++(1)

        net = net.reverse
        out = out.reverse
        mask = mask.reverse

        val y = output.answer
        val ŷ = out(M).shaped[N[M]+1].--

        // BACKPROPAGATION

        var DELTA: Tensor[Float, ?, ?, ?] = null
        var delta: Vector[Float, ?] =
          if softmax_cross_entropy
          then
            ŷ - y
          else {
            given Int = M
            Vector[Float][N[given_Int.type]](rows.map(loss.partial(y, ŷ)(_))*)
          ⊙ prime(given_Int)(net(M-1).shaped[N])
          }

        for // FULLY CONNECTED
          given Int <- cols.reverse
          l = given_Int-1
        do
          val ∇ = nabla(l).shaped[N[given_Int.type], N[l.type]+1]
          val δ = delta.shaped[N]
          val h = out(l).shaped[N[l.type]+1]

          nabla(l) := ∇ + δ ⋅ h

          val w = weights(l-K).shaped[N[given_Int.type], N[l.type]+1]
          val δʹ: Vector[Float, N[l.type]] =
            layers(l-1) match
              case _: Layer.Dropout[?, ?] =>
                (~w ⋅ δ).-- ⊙ mask(l-1).shaped[N](using l)
              case _ =>
                (~w ⋅ δ).--
          if l > K
          then
            delta = δʹ ⊙ prime(l)(net(l-1).shaped[N](using l))
          else
            given Int = K
            DELTA = δʹ.reshape[FL[given_Int.type]].reshape[D[given_Int.type]]

        for // CONVOLUTION|POOLING
          given Int <- maps.reverse
          l = given_Int-1
          padding = volume(given_Int)._6
        do
          layers(l) match
            case lrn: ConvolutionalLRN[?, ?, ?, ?] =>
              val a = OUT(given_Int).shaped[FL, FB, D]
              val b = LRN(given_Int)._2.shaped[FL, FB, D]
              val S = LRN(given_Int)._1.shaped[FL, FB, D]
              val δ = DELTA.shaped[FL, FB, D]
              val N = a.depth
              val n = lrn.n
              val α = lrn.α
              val β = lrn.β
              val n_2 = n/2
              DELTA = Tensor[Float][FL[given_Int.type], FB[given_Int.type], D[given_Int.type]]({
                for
                  i <- 0 until a.length
                  j <- 0 until a.breadth
                yield
                  for
                    c <- 0 until N
                    s = 0 max (c-n_2)
                    e = (N-1) min (c+n_2)
                  yield δ(i, j, c) * pow(S(i, j, c), β).toFloat - {
                    for
                      h <- s to e
                    yield
                      δ(i, j, h) * b(i, j, h) / S(i, j, h)
                  }.sum * a(i, j, c) * 2*α*β / n
              }.flatten*)
            case _ =>
          val a = NET(l).shaped[FL, FB, D]
          val δ = DELTA.shaped[FL, FB, D]
                ⊙ (a -->> layers(l).activation)
          if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
          then // CONVOLUTION
            val x = PAD(l).shaped[padding.type][FL, FB, D]
            val kernels = layers(l).kernels[KL, KB, D[l.type]]
            { for
                d <- 0 until kernels(0).depth
              yield
                x(d)
            } match
              case x =>
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
                          x(d).⋆[P, Q](δ(k))
                      { val ∇ = NABLA(l)(k).shaped[KL, KB][D[l.type]]
                        val ∂ = { given Int = l; Tensor[Float].stack(ms*)[D[given_Int.type]] }.shaped[KL, KB][D[l.type]]
                        NABLA(l)(k) := ∇ + ∂
                      }
                      { val ∇ = NBIAS(l).shaped[D]
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
                  DELTA = { given Int = l; Tensor[Float].stack(ms*)[D[given_Int.type]] }
                  if padding > 0
                  then
                    type P = padding.type
                    given ValueOf[P] = ValueOf(padding)
                    DELTA = DELTA.crop[P, P]
          else // POOLING
            type P = (FL[l.type] - PL[given_Int.type]) / PS[given_Int.type] + 1
            type Q = (FB[l.type] - PB[given_Int.type]) / PS[given_Int.type] + 1
            val pa = a.shaped[P, Q, D[l.type]]
            val pδ = δ.shaped[P, Q, D[l.type]]
            val pooling = layers(l).pooling[PL, PB, PS]
            pooling match
              case subsampling(_, _) =>
                { for
                    k <- 0 until valueOf[D[given_Int.type]]
                  yield
                    pδ(k)
                } match
                  case δ =>
                    { val ∇ = NBETA(l).shaped[D]
                      pa.volume match
                        case SubsamplingVolume(avg, _) =>
                          for
                            k <- 0 until valueOf[D[given_Int.type]]
                          do
                            NBETA(l)(k) = ∇(k) + (δ(k) ⊙ avg(k)).sum
                    }
                    { val ∇ = NBIAS(l).shaped[D]
                      for
                        k <- 0 until valueOf[D[given_Int.type]]
                      do
                        NBIAS(l)(k) = ∇(k) + δ(k).sum
                    }
              case _ =>
            if l > 0
            then
              DELTA = pooling(OUT(l).shaped[FL, FB, D](using l).shape, pa, pδ)
              if padding > 0
              then
                type P = padding.type
                given ValueOf[P] = ValueOf(padding)
                DELTA = DELTA.crop[P, P]

      if decayFactor > 0
      then // MOMENTUM GRADIENT DESCENT

        for // CONVOLUTION|POOLING
          given Int <- maps
          l = given_Int-1
        do
          if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
          then // CONVOLUTION
            for
              k <- 0 until layers(l).kernels.size
            do
              val ∇ = NABLA(l)(k).shaped[KL, KB][D[l.type]]
              val v = VELOCITY(l)(k).shaped[KL, KB][D[l.type]]
              val update = ∇.op(-learningRate * _ / batch)
              val weights = layers(l).kernels[KL, KB, D[l.type]](k).volume.data
              v.velocityUpdate(momentumCoefficient, decayFactor, learningRate)(weights, update)
              layers(l).kernels(k).volume.data := weights + v
            val ∇ = NBIAS(l).shaped[D]
            val v = VBIAS(l).shaped[D]
            val update = ∇.op(-learningRate * _ / batch)
            val biases = Vector[Float][D[given_Int.type]](layers(l).kernels.map(_.bias)*)
            v.velocityUpdate(momentumCoefficient, decayFactor, learningRate)(biases, update)
            for
              k <- 0 until v.size
            do
              layers(l).kernels(k).bias += v(k)
            NBIAS(l).reset
            NABLA(l).foreach(_.reset)
          else // POOLING
            layers(l).pooling match
              case ss @ subsampling(β: Vector[Float, D[given_Int.type]], b: Vector[Float, D[given_Int.type]]) =>
                {
                  val ∇ = NBETA(l).shaped[D]
                  val v = VBETA(l).shaped[D]
                  val update = ∇.op(-learningRate * _ / batch)
                  v.velocityUpdate(momentumCoefficient, decayFactor, learningRate)(β, update)
                  ss.beta := β + v
                }
                {
                  val ∇ = NBIAS(l).shaped[D]
                  val v = VBIAS(l).shaped[D]
                  val update = ∇.op(-learningRate * _ / batch)
                  v.velocityUpdate(momentumCoefficient, decayFactor, learningRate)(b, update)
                  ss.bias := b + v
                }
                NBETA(l).reset
                NBIAS(l).reset
              case _ =>

        for // FULLY CONNECTED
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

        for // CONVOLUTION|POOLING
          given Int <- maps
          l = given_Int-1
        do
          if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
          then // CONVOLUTION
            for
              k <- 0 until layers(l).kernels.size
            do
              val ∇ = NABLA(l)(k).shaped[KL, KB][D[l.type]]
              val update = ∇.op(-learningRate * _ / batch)
              val weights = layers(l).kernels[KL, KB, D[l.type]](k).volume.data
              layers(l).kernels(k).volume.data := weights + update
            val ∇ = NBIAS(l).shaped[D]
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
                  val ∇ = NBETA(l).shaped[D]
                  val update = ∇.op(-learningRate * _ / batch)
                  ss.beta := β + update
                }
                {
                  val ∇ = NBIAS(l).shaped[D]
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
          val ∇ = nabla(l).shaped[N[given_Int.type], N[l.type]+1]
          val update = ∇.op(-learningRate * _ / batch)
          for
            n <- rows
            update0 = update(n, 0)
            update1 = update(n).--
          do
            val weights = layers(l).neurons(n).weights.shaped[N[l.type]]
            layers(l).neurons(n).bias += update0
            layers(l).neurons(n).weights := weights + update1

          nabla(l).reset

    blink(epochs, data.io.size)

  def test(input: Input[FL[0], FB[0], D[0]], weights: List[Matrix[Float, ?, ?]] = this()): Output[N[M]] =
    var OUT: FeatureMap[Float, ?, ?, ?] = input.image

    for // CONVOLUTION|POOLING
      given Int <- maps
      l = given_Int-1
      padding = volume(given_Int)._6
    do
      val featureMapIn = OUT.shaped[FL, FB, D](using l)
      var featureMapOut = {
        if !layers(l).isInstanceOf[Pooling[?, ?, ?]]
        then // CONVOLUTION
          val kernels = layers(l).kernels[KL, KB, D[l.type]]
          featureMapIn[KS[given_Int.type]].⋆(kernels*)[D[given_Int.type]]
        else // POOLING
          val pooling = layers(l).pooling[PL, PB, PS]
          pooling(featureMapIn)
      }.pad(padding).shaped[FL, FB, D]
      featureMapOut = featureMapOut --> layers(l).activation
      layers(l) match
        case lrn: ConvolutionalLRN[?, ?, ?, ?] =>
          OUT = featureMapOut.lrn(lrn.k, lrn.n, lrn.α, lrn.β)(false)._2
        case _ =>
          OUT = featureMapOut

    val flattened = {
      given Int = K
      Vector[Float][N[K]](OUT.toSeq*)
    }

    var out: Vector[Float, ?] = flattened.++(1)

    for // FULLY CONNECTED
      given Int <- cols
      l = given_Int-1
    do
      val w = weights(l-K).shaped[N[given_Int.type], N[l.type]+1]
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

  extension [M <: Int, N <: Int, O <: Int](self: Tensor[Float, M, N, O])
    def velocityUpdate(momentumCoefficient: Float, decayFactor: Float, learningRate: Float)(biases: Tensor[Float, M, N, O], update: Tensor[Float, M, N, O]): Unit =
      for
        i <- 0 until self.rows
        j <- 0 until self.cols
        k <- 0 until self.depth
      do
        self((i, j, k)) = momentumCoefficient*self(i, j, k) - decayFactor*learningRate*biases(i, j, k) - update(i, j , k)

  extension [N <: Int](self: Vector[Float, N])
    def velocityUpdate(momentumCoefficient: Float, decayFactor: Float, learningRate: Float)(biases: Vector[Float, N], update: Vector[Float, N]): Unit =
      for
        n <- 0 until self.rows
      do
        self(n) = momentumCoefficient*self(n) - decayFactor*learningRate*biases(n) - update(n)

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

      def kaiming[I <: Int: ValueOf,
                  O <: Int: ValueOf](activation: nnn.float.Activation = nnn.float.Activation.ReLU): Seq[Neuron[I]] =
        apply[I, O](Kaiming(valueOf[I]), activation)

    def apply[I <: Int: ValueOf,
              O <: Int: ValueOf](initialization: Initialization,
                                 activation: nnn.float.Activation): Seq[Neuron[I]] =
      (1 to valueOf[O]).map(_ => Neuron(Vector[Float][I](initialization), initialization(), activation))

    def glorot[I <: Int: ValueOf,
               O <: Int: ValueOf](activation: nnn.float.Activation): Seq[Neuron[I]] =
      apply[I, O](Glorot(valueOf[I], valueOf[O]), activation)

    def xavier[I <: Int: ValueOf,
               O <: Int: ValueOf](activation: nnn.float.Activation): Seq[Neuron[I]] =
      apply[I, O](Xavier(valueOf[I], valueOf[O]), activation)

    def kaiming[I <: Int: ValueOf,
                O <: Int: ValueOf](activation: nnn.float.Activation = nnn.float.Activation.ReLU): Seq[Neuron[I]] =
      apply[I, O](Kaiming(valueOf[I]), activation)

  sealed trait Layer

  object Layer:

    case class Convolutional[
      L <: Int,
      B <: Int,
      D <: Int,
      O <: Int: ValueOf
    ](activation: nnn.float.Activation,
      kernels: Kernel[Float, L, B, D]*) extends Layer:
      require(valueOf[O] == kernels.size)

    class ConvolutionalLRN[
      L <: Int,
      B <: Int,
      D <: Int,
      O <: Int: ValueOf
    ](val k: Float, val n: Int, val α: Float, val β: Float,
      activation: nnn.float.Activation,
      kernels: Kernel[Float, L, B, D]*)
        extends Convolutional[L, B, D, O](activation, kernels*)

    case class Pooling[
      P <: Int,
      Q <: Int,
      S <: Int: ValueOf
    ](pooling: nnn.cnn.Pooling[Float, P, Q, S],
      activation: nnn.float.Activation = nnn.float.Activation.Linear()) extends Layer

    case class Dense[N <: Int, O <: Int: ValueOf](neurons: Neuron[N]*) extends Layer:
      require(valueOf[O] == neurons.size)

    class Dropout[N <: Int, O <: Int: ValueOf](val keep: Float, neurons: Neuron[N]*)
        extends Dense[N, O](neurons*)

    class Softmax[N <: Int: ValueOf](initialization: Initialization,
                                     bias: Float = 0,
                                     activation: nnn.float.Activation = nnn.float.Activation.Softmax)
        extends Dense[N, N](Neuron[N, N](initialization, activation)*):
      this.neurons.foreach(_.bias = bias)

    extension (self: Layer)
      def pooling[P[_ <: Int] <: Int,
                  Q[_ <: Int] <: Int,
                  S[_ <: Int] <: Int](using l: Int): nnn.cnn.Pooling[Float, P[l.type], Q[l.type], S[l.type]] =
        self.asInstanceOf[Pooling[P[l.type], Q[l.type], S[l.type]]].pooling
      def activation: nnn.float.Activation =
        if self.isInstanceOf[Pooling[?, ?, ?]]
        then
          self.asInstanceOf[Pooling[?, ?, ?]].activation
        else
          self.asInstanceOf[Convolutional[?, ?, ?, ?]].activation
      def kernels[L[_ <: Int] <: Int, B[_ <: Int] <: Int, D <: Int](using l: Int): Seq[Kernel[Float, L[l.type], B[l.type], D]] =
        self.asInstanceOf[Convolutional[L[l.type], B[l.type], D, ?]].kernels
      def neurons: Seq[Neuron[?]] = self.asInstanceOf[Dense[?, ?]].neurons
