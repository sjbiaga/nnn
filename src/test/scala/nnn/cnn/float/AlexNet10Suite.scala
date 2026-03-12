package nnn
package cnn
package float

import scala.compiletime.ops.int.*
import scala.concurrent.duration.*

import munit.FunSuite

import spire.implicits.*

import nnn.float.Activation.{ Gaussian => _, * }
import Initialization.*
import nnn.float.Loss.*
import nnn.float.Util.given
import Image.*
import Pooling.*
import Network.*
import AlexNet10Suite.*


class AlexNet10Suite extends FunSuite:

  //override val munitTimeout = 14.days

  test("CNN:CIFAR-10 https://en.wikipedia.org/wiki/AlexNet#/media/File:AlexNet_block_diagram.svg") {

    // TYPE CHECKING

    def imageOf(using fl: ValueOf[FL[0]], fb: ValueOf[FL[0]], d: ValueOf[D[0]]): (Int, Int, Int, Int, Int, Int, Int) =
      (fl.value, fb.value, 0, 0, 0, 0, d.value)

    def kernelsOf[C <: Int](using fl: ValueOf[FL[C]], fb: ValueOf[FL[C]],
                                  kl: ValueOf[KL[C]], kb: ValueOf[KB[C]], ks: ValueOf[KS[C]],
                                   p: ValueOf[P[C]], d: ValueOf[D[C]]): (Int, Int, Int, Int, Int, Int, Int) =
      (fl.value, fb.value, kl.value, kb.value, ks.value, p.value, d.value)

    def poolingOf[C <: Int](using fl: ValueOf[FL[C]], fb: ValueOf[FL[C]],
                                  pl: ValueOf[PL[C]], pb: ValueOf[PB[C]], ps: ValueOf[PS[C]],
                                  d: ValueOf[D[C]]): (Int, Int, Int, Int, Int, Int, Int) =
      (fl.value, fb.value, pl.value, pb.value, ps.value, 0, d.value)

    def shapeOf[L <: Int](using n: ValueOf[N[L]]): Int = n.value

    def C[C <: Int,
          O <: Int: ValueOf](layer: Layer.Convolutional[?, ?, ?, O])
                            (using flp: ValueOf[FL[C-1]], fbp: ValueOf[FB[C-1]],
                                   fl: ValueOf[FL[C]], fb: ValueOf[FL[C]],
                                   kl: ValueOf[KL[C]], kb: ValueOf[KB[C]], ks: ValueOf[KS[C]],
                                   p: ValueOf[P[C]], d: ValueOf[D[C-1]]): layer.type =
      require((flp.value+2*p.value-kl.value)%ks.value == 0 && (fbp.value+2*p.value-kb.value)%ks.value == 0)
      require((flp.value+2*p.value-kl.value)/ks.value+1 == fl.value && (fbp.value+2*p.value-kb.value)/ks.value+1 == fb.value)
      require(valueOf[O] == layer.kernels.size)
      require(layer.kernels.forall { it => it.length == kl.value && it.breadth == kb.value && it.depth == d.value })
      layer

    def P[C <: Int](layer: Layer.Pooling[?, ?, ?])
                   (using pl: ValueOf[PL[C]], pb: ValueOf[PB[C]], ps: ValueOf[PS[C]]): layer.type =
      require(layer.pooling match { case it => it.rows == pl.value && it.cols == pb.value && it.stride == ps.value })
      layer

    def D[L <: Int](layer: Layer.Dense[?, ?])
                   (using p: ValueOf[N[L-1]], n: ValueOf[N[L]]): layer.type =
      require(n.value == layer.neurons.size)
      require(layer.neurons.forall { it => it.weights.size == p.value })
      layer

    // SHAPES

    type FL[C <: Int] = C match { case 0 => 32 case 1 => 32 case 2 | 3 => 16 case 4 | 5 | 6 | 7 => 8 case 8 => 4 } // Feature Length

    type FB[C <: Int] = C match { case 0 => 32 case 1 => 32 case 2 | 3 => 16 case 4 | 5 | 6 | 7 => 8 case 8 => 4 } // Feature Breadth

    type KL[C <: Int] = C match { case 1 | 3 | 5 | 6 | 7 => 3 } // Kernel Length

    type KB[C <: Int] = C match { case 1 | 3 | 5 | 6 | 7 => 3 } // Kernel Breadth

    type KS[C <: Int] = C match { case 1 | 3 | 5 | 6 | 7 => 1 } // Kernel Stride

    type P[C <: Int] = C match { case 1 | 3 | 5 | 6 | 7 => 1 } // Padding

    type D[C <: Int] = C match { case 0 => 3 case 1 | 2 => 96 case 3 | 4 | 7 | 8 => 256 case 5 | 6 => 384 } // Feature/Kernel/Pooling Depth

    type PL[C <: Int] = C match { case 2 | 4 | 8 => 2 } // Pooling Length

    type PB[C <: Int] = C match { case 2 | 4 | 8 => 2 } // Pooling Breadth

    type PS[C <: Int] = C match { case 2 | 4 | 8 => 2 } // Pooling Stride

    type N[L <: Int] = L match { case 8 => FL[8]*FB[8]*D[8] case 9 | 10 => 512 case 11 | 12 => 10 }

    // true     FL   FB   KL   KB   KS   P    D
    // false    FL   FB   PL   PB   PS   P    D
    given List[(Int, Int, Int, Int, Int, Int, Int)] = imageOf
                                                   :: kernelsOf[1] // convolutional C1
                                                   :: poolingOf[2] // max pooling   P1
                                                   :: kernelsOf[3] // convolutional C2
                                                   :: poolingOf[4] // max pooling   P2
                                                   :: kernelsOf[5] // convolutional C3
                                                   :: kernelsOf[6] // convolutional C4
                                                   :: kernelsOf[7] // convolutional C5
                                                   :: poolingOf[8] // max pooling   P3
                                                   :: Nil

    given List[Int] = shapeOf[8] :: shapeOf[9] :: shapeOf[10] :: shapeOf[11] :: shapeOf[12] :: Nil

    print("Initializing CIFAR-10 AlexNet CNN...")

    val an = Network[FL, FB, KL, KB, KS, D, PL, PB, PS, 8, N, 12](
      loss = CCE[10](),
      (0.001, 0.5, 0.001),
      C[1, 96](Layer.ConvolutionalLRN[3, 3, 3, 96](2, 5, 1E-4, 0.75, ReLU, Kernel.bias(0f)[3, 3, 3](gaussian)[96]*)),       // C1
      P[2](Layer.Pooling[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue))),                                                    // P1
      C[3, 256](Layer.ConvolutionalLRN[3, 3, 96, 256](2, 5, 1E-4, 0.75, ReLU, Kernel.bias(1f)[3, 3, 96](gaussian)[256]*)),  // C2
      P[4](Layer.Pooling[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue))),                                                    // P2
      C[5, 384](Layer.Convolutional[3, 3, 256, 384](ReLU, Kernel.bias(0f)[3, 3, 256](gaussian)[384]*)),                     // C3
      C[6, 384](Layer.Convolutional[3, 3, 384, 384](ReLU, Kernel.bias(1f)[3, 3, 384](gaussian)[384]*)),                     // C4
      C[7, 256](Layer.Convolutional[3, 3, 384, 256](ReLU, Kernel.bias(1f)[3, 3, 384](gaussian)[256]*)),                     // C5
      P[8](Layer.Pooling[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue))),                                                    // P3
      D[9](Layer.Dropout[4096, 512](0.5, Neuron.bias(1f)[4096, 512](gaussian, ReLU)*)),
      D[10](Layer.Dropout[512, 512](0.5, Neuron.bias(1f)[512, 512](gaussian, ReLU)*)),
      D[11](Layer.Dense[512, 10](Neuron.bias(1f)[512, 10](gaussian, Linear())*)),
      D[12](Layer.Softmax[10](gaussian, 1f))
    )

    val label = -1 // rnd.nextInt(10)

    {
      val batch = 1 // 128

      val data = Data[32, 32, 3, 10]({
        val read = 0 // 10000
        val train = 0 // 10000

        val drop = rnd.nextInt(read-train+1)

        val trainData = rnd.shuffle(trainCIFAR10("./data/cifar-10-batches-bin", label, drop+train, true).drop(drop))

        val size = (trainData.size / batch) * batch

        print(s""" Reading $size images${if label < 0 then "" else " labelled '${labels(label)}'"}...""")

        for
          case image @ Image(label: Int, _) <- trainData.drop(trainData.size - size)
        yield
          Input(image) -> OneHotOutput(label)
      }*)

      println(" done.")

      val epochs = 0 // 90

      print(s"Training ${data.io.size} images in $batch batches and $epochs epochs...")

      an.train(data, batch, epochs) {
        case (count, done) if count % 5 == 0 && done % 256 == 0 =>
          print(s" Passing through $count epochs and $done images...")
        case _ =>
      }
    }

    println(" done.")

    val weights = an()

    {
      val read = 0 // 10000
      val test = 0 // 10000

      val drop = rnd.nextInt(read-test+1)

      val testData = rnd.shuffle(testCIFAR10("./data/cifar-10-batches-bin", label, drop+test, true).drop(drop))

      val size = testData.size

      print(s"""Testing $size images${if label < 0 then "" else " labelled '${labels(label)}'"}...""")

      var correct = 0

      for
        case image @ Image(label: Int, _) <- testData
        Output(answer) = an.test(Input(image), weights)
        if label == answer.toSeq.zipWithIndex.maxBy(_._1)._2
      do
        correct += 1

      println(" done.")

      val accuracy = ((1f * correct / size) * 100).toInt

      //assert(accuracy >= 98, s"Accuracy: $accuracy% < 98%")
    }

  }


object AlexNet10Suite:

  val rnd = scala.util.Random

  val gaussian: Initialization = Gaussian(0, 0.01)

  import java.io.FileInputStream
  import java.nio.file.Paths

  val labels = List("airplane",
                    "automobile",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck")

  def trainCIFAR10(path: String, label: Int, size: Int, zero: Boolean = true): Seq[Image[Float, 32, 32, 3]] =
    var sum = 0
    ( for
        batch <- rnd.shuffle(1 to 5)
        if sum < size
      yield
        val seq = CIFAR10(path, s"data_batch_$batch", label, size - sum, zero)
        sum += seq.size
        seq
    ).flatten

  def testCIFAR10(path: String, label: Int, size: Int = Int.MaxValue, zero: Boolean = true): Seq[Image[Float, 32, 32, 3]] =
    CIFAR10(path, "test_batch", label, size, zero)

  def CIFAR10(path: String, name: String, label: Int, size: Int, zero: Boolean): Seq[Image[Float, 32, 32, 3]] =
    if label < 0
    then
      CIFAR10(path, name, size, zero)
    else
      CIFAR10(path, name, Int.MaxValue, zero)
        .filter(_.label == label)
        .take(size)

  def CIFAR10(path: String, name: String, size: Int, zero: Boolean): Seq[Image[Float, 32, 32, 3]] =

    val binFileName = s"$name.bin"

    val binInputStream =
      try

        FileInputStream(Paths.get(path).resolve(binFileName).toFile)

      catch t =>

        println("Cannot open CIFAR-10 data files: please consult README.md for instructions!")
        throw t

    try

      for
        _ <- 0 until (size min 10000)
      yield
        // https://gist.github.com/sugyan/fa8391fb2b2da68ee981a5962d58e834

        val b = Array.fill(3073)(0.toByte)
        assert(binInputStream.read(b) == 3073)

        val label = b(0) & 0xFF

        val data = Array.fill(32, 32, 3)(if zero then 0f else -1f)

        for
          row <- 0 until 32
          col <- 0 until 32
          red = b(1 + 1024 * 0 + row * 32 + col) & 0xFF
          green = b(1 + 1024 * 1 + row * 32 + col) & 0xFF
          blue = b(1 + 1024 * 2 + row * 32 + col) & 0xFF
        do
          data(row)(col)(0) = if zero then red / 255.0f else 2 * red / 255.0f - 1
          data(row)(col)(1) = if zero then green / 255.0f else 2 * green / 255.0f - 1
          data(row)(col)(2) = if zero then blue / 255.0f else 2 * blue / 255.0f - 1

        Image[Float, 32, 32, 3](label, Tensor[Float, 32, 32, 3](data))

    finally
      binInputStream.close
