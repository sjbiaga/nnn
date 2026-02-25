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
import AlexNetSuite.*


class AlexNetSuite extends FunSuite:

  //override val munitTimeout = 24.hours

  test("CNN:CIFAR-10 https://en.wikipedia.org/wiki/AlexNet#/media/File:AlexNet_block_diagram.svg") {

    // TYPE CHECKING

    def C[C <: Int: ValueOf, O <: Int: ValueOf](layer: Layer.Convolutional[?, ?, ?, O])
                                               (using flp: ValueOf[FL[C-1]], fbp: ValueOf[FB[C-1]],
                                                      fl: ValueOf[FL[C]], fb: ValueOf[FL[C]],
                                                      kl: ValueOf[KL[C]], kb: ValueOf[KB[C]], ks: ValueOf[KS[C]],
                                                      d: ValueOf[D[C-1]])
                                               (using volume: List[(Int, Int, Int, Int, Int, Int, Int)]): layer.type =
      val padding = volume(valueOf[C])._6
      require((flp.value+2*padding-kl.value)%ks.value == 0 && (fbp.value+2*padding-kb.value)%ks.value == 0)
      require((flp.value+2*padding-kl.value)/ks.value+1 == fl.value && (fbp.value+2*padding-kb.value)/ks.value+1 == fb.value)
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

    type FL[C <: Int] = C match { case 0 => 227 case 1 => 55 case 2 | 3 => 27 case 4 | 5 | 6 | 7 => 13 case 8 => 6 } // Feature Length

    type FB[C <: Int] = C match { case 0 => 227 case 1 => 55 case 2 | 3 => 27 case 4 | 5 | 6 | 7 => 13 case 8 => 6 } // Feature Breadth

    type KL[C <: Int] = C match { case 1 => 11 case 3 => 5 case 5 | 6 | 7 => 3 } // Kernel Length

    type KB[C <: Int] = C match { case 1 => 11 case 3 => 5 case 5 | 6 | 7 => 3 } // Kernel Breadth

    type KS[C <: Int] = C match { case 1 => 4 case 3 => 1 case 5 | 6 | 7 => 1 } // Kernel Stride

    type D[C <: Int] = C match { case 0 => 3 case 1 | 2 => 96 case 3 | 4 | 7 | 8 => 256 case 5 | 6 => 384 } // Feature/Kernel/Pooling Depth

    type PL[C <: Int] = C match { case 2 | 4 | 8 => 3 } // Pooling Length

    type PB[C <: Int] = C match { case 2 | 4 | 8 => 3 } // Pooling Breadth

    type PS[C <: Int] = C match { case 2 | 4 | 8 => 2 } // Pooling Stride

    type N[L <: Int] = L match { case 8 => FL[8]*FB[8]*D[8] case 9 | 10 => 512 case 11 | 12 => 10 }

    given List[Boolean] = false :: true :: false :: true :: false :: true :: true :: true :: false :: Nil

    // true     FL   FB   KL   KB   KS   pad  D
    // false    FL   FB   PL   PB   PS   pad  D
    given List[(Int, Int, Int, Int, Int, Int, Int)] = (227, 227,  0,  0, 0, 0,   3) // image
                                                   :: ( 55,  55, 11, 11, 4, 0,  96) // convolutional
                                                   :: ( 27,  27,  3,  3, 2, 0,  96) // max pooling
                                                   :: ( 27,  27,  5,  5, 1, 2, 256) // convolutional
                                                   :: ( 13,  13,  3,  3, 2, 0, 256) // max pooling
                                                   :: ( 13,  13,  3,  3, 1, 1, 384) // convolutional
                                                   :: ( 13,  13,  3,  3, 1, 1, 384) // convolutional
                                                   :: ( 13,  13,  3,  3, 1, 1, 256) // convolutional
                                                   :: (  6,   6,  3,  3, 2, 0, 256) // max pooling
                                                   :: Nil

    given List[Int] = 9216/*=6*6*256*/ :: 512 :: 512 :: 10 :: 10 :: Nil

    print("Initializing AlexNet CNN...")

    val an = Network[FL, FB, KL, KB, KS, D, PL, PB, PS, 8, N, 12](
      loss = CCE[10](),
      (0.01, 0.9, 0.0005),
      C[1, 96](Layer.ConvolutionalLRN[11, 11, 3, 96](2, 5, 1E-4, 0.75, ReLU, Kernel.bias(0f)[11, 11, 3](gaussian)[96]*)),   // C1
      P[2](Layer.Pooling[3, 3, 2](max[Float, 3, 3, 2](Float.MinValue))),                                                    // P1
      C[3, 256](Layer.ConvolutionalLRN[5, 5, 96, 256](2, 5, 1E-4, 0.75, ReLU, Kernel.bias(1f)[5, 5, 96](gaussian)[256]*)),  // C2
      P[4](Layer.Pooling[3, 3, 2](max[Float, 3, 3, 2](Float.MinValue))),                                                    // P2
      C[5, 384](Layer.Convolutional[3, 3, 256, 384](ReLU, Kernel.bias(0f)[3, 3, 256](gaussian)[384]*)),                     // C3
      C[6, 384](Layer.Convolutional[3, 3, 384, 384](ReLU, Kernel.bias(1f)[3, 3, 384](gaussian)[384]*)),                     // C4
      C[7, 256](Layer.Convolutional[3, 3, 384, 256](ReLU, Kernel.bias(1f)[3, 3, 384](gaussian)[256]*)),                     // C5
      P[8](Layer.Pooling[3, 3, 2](max[Float, 3, 3, 2](Float.MinValue))),                                                    // P3
      D[9](Layer.Dropout[9216, 512](0.5, Neuron.bias(1f)[9216, 512](gaussian, ReLU)*)),
      D[10](Layer.Dropout[512, 512](0.5, Neuron.bias(1f)[512, 512](gaussian, ReLU)*)),
      D[11](Layer.Dense[512, 10](Neuron.bias(1f)[512, 10](gaussian, ReLU)*)),
      D[12](Layer.Softmax[10](gaussian))
    )

    println(" done.")

    val data = Data[227, 227, 3, 10]({
      val read = 1 // 10000
      val train = 0 // 10000

      val drop = rnd.nextInt(read-train+1)
      val batch = 1+rnd.nextInt(5)

      for
        case image @ Image(label: Int, _) <- rnd.shuffle(trainCIFAR10("./data/cifar-10-batches-bin", batch, drop+train, true).drop(drop))
      yield
        Input(image) -> OneHotOutput(label)
    }*)

    val batch = 1 // 100
    val epochs = 1 // 20

    print(s"Training ${data.io.size} images in $batch batches and $epochs epochs...")

    an(data, batch, epochs) {
      // case (count, done) if count % 5 == 0 && done % 5000 == 0 =>
      //   print(s" Passing through $count epochs and $done images...")
      case _ =>
    }

    println(" done.")

    val read = 1 // 10000
    val test = 0 // 10000

    val drop = rnd.nextInt(read-test+1)

    print(s"Testing $test images...")

    var correct = 0

    for
      case image @ Image(label: Int, _) <- rnd.shuffle(testCIFAR10("./data/cifar-10-batches-bin", drop+test, false).drop(drop))
      Output(answer) = an(Input(image))
      if label == answer.toSeq.zipWithIndex.maxBy(_._1)._2
    do
      correct += 1

    println(" done.")

    val accuracy = ((1f * correct / test) * 100).toInt

    //assert(accuracy >= 98, s"Accuracy: $accuracy% < 98%")
  }


object AlexNetSuite:

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

  def trainCIFAR10(path: String, batch: Int, size: Int = 10000, zero: Boolean = true): Seq[Image[Float, 227, 227, 3]] =
    CIFAR10(path, s"data_batch_$batch", size, zero)

  def testCIFAR10(path: String, size: Int = Int.MaxValue, zero: Boolean = true): Seq[Image[Float, 227, 227, 3]] =
    CIFAR10(path, "test_batch", size, zero)

  def CIFAR10(path: String, name: String, size: Int, zero: Boolean): Seq[Image[Float, 227, 227, 3]] =

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

        import java.awt.Color
        import java.awt.Image.SCALE_DEFAULT
        import java.awt.image.BufferedImage

        val b = Array.fill(3073)(0.toByte)
        binInputStream.read(b)

        val image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB)

        val label = b(0) & 0xFF

        for
          row <- 0 until 32
          col <- 0 until 32
        do
          val color = new Color(b(1 + 1024 * 0 + row * 32 + col) & 0xFF,
                                b(1 + 1024 * 1 + row * 32 + col) & 0xFF,
                                b(1 + 1024 * 2 + row * 32 + col) & 0xFF)
          image.setRGB(col, row, color.getRGB)

        // https://stackoverflow.com/questions/20016123/scala-image-resize-and-crop

        val resized = new BufferedImage(227, 227, BufferedImage.TYPE_INT_RGB)
        resized.getGraphics.drawImage(image.getScaledInstance(224, 224, SCALE_DEFAULT), 0, 0, null)

        // https://github.com/eliasyilma/CNN/blob/master/src/cnn/CNN.java#L63

        val pixels = resized.getRGB(0, 0, 224, 224, null, 0, 224)

        val data = Array.fill(227, 227, 3)(if zero then 0f else -1f)

        var row = 0
        var col = 0

        // https://stackoverflow.com/questions/22391353/get-color-of-each-pixel-of-an-image-using-bufferedimages

        for
          pixel <- pixels
          red = (pixel >> 16) & 0xFF
          green = (pixel >> 8) & 0xFF
          blue = pixel & 0xFF
        do
          data(row+1)(col+1)(0) = if zero then red / 255.0f else 2 * red / 255.0f - 1
          data(row+1)(col+1)(1) = if zero then green / 255.0f else 2 * green / 255.0f - 1
          data(row+1)(col+1)(2) = if zero then blue / 255.0f else 2 * blue / 255.0f - 1
          col += 1
          if col == 224
          then
            col = 0
            row += 1

        Image[Float, 227, 227, 3](label, Tensor[Float, 227, 227, 3](data))

    finally
      binInputStream.close
