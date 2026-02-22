package nnn
package cnn
package float

import scala.compiletime.ops.int.*
import scala.concurrent.duration.*

import munit.FunSuite

import spire.implicits.*

import nnn.float.Activation.*
import Initialization.*
import nnn.float.Loss.*
import nnn.float.Util.given
import Image.*
import Pooling.*
import Network.*
import NetworkSuite.*


class NetworkSuite extends FunSuite:

  override val munitTimeout = 1.day

  test("CNN https://en.wikipedia.org/wiki/LeNet#/media/File:LeNet-5_architecture_block_diagram.svg") {

    type FL[C <: Int] = C match { case 0 => 32 case 1 => 28 case 2 => 14 case 3 => 10 case 4 => 5 } // Feature Length

    type FB[C <: Int] = C match { case 0 => 32 case 1 => 28 case 2 => 14 case 3 => 10 case 4 => 5 } // Feature Breadth

    type KL[C <: Int] = C match { case 1 => 5 case 3 => 5 } // Kernel Length

    type KB[C <: Int] = C match { case 1 => 5 case 3 => 5 } // Kernel Breadth

    type KS[C <: Int] = C match { case 1 => 1 case 3 => 1 } // Kernel Stride

    type D[C <: Int] = C match { case 0 => 1 case 1 | 2 => 6 case 3 | 4 => 16 } // Feature/Kernel/Pooling Depth

    type PL[C <: Int] = C match { case 2 => 2 case 4 => 2 } // Pooling Length

    type PB[C <: Int] = C match { case 2 => 2 case 4 => 2 } // Pooling Breadth

    type PS[C <: Int] = C match { case 2 => 2 case 4 => 2 } // Pooling Stride

    type N[L <: Int] = L match { case 4 => FL[4]*FB[4]*D[4] case 5 => 120 case 6 => 84 case 7 => 10 case 8 => 10 }

    given List[Boolean] = false :: true :: false :: true :: false :: Nil

    // true     FL   FB   KL   KB   KS   D
    // false    FL   FB   PL   PB   PS   D
    given List[(Int, Int, Int, Int, Int, Int)] = (32, 32, 0, 0, 0,  1) // image
                                              :: (28, 28, 5, 5, 1,  6) // convolutional
                                              :: (14, 14, 2, 2, 2,  6) // subsampling
                                              :: (10, 10, 5, 5, 1, 16) // convolutional
                                              :: ( 5,  5, 2, 2, 2, 16) // subsampling
                                              :: Nil

    given List[Int] = 400/*=5*5*16*/ :: 120 :: 84 :: 10 :: 10 :: Nil

    // val ln = Network[FL, FB, KL, KB, KS, D, PL, PB, PS, 4, N, 8](
    //   loss = CCE[10](),
    //   learningRate = 0.01,
    //   Layer.Convolution[5, 5, 1, 6](LeakyReLU(), Kernel[Float, 5, 5, 1](glorot[5, 5, 1, 6])[6]*),
    //   Layer.Pool[2, 2, 2](subsampling[Float, 2, 2, 2, 6](null), LeCunnTanh),
    //   Layer.Convolution[5, 5, 6, 16](LeakyReLU(), Kernel[Float, 5, 5, 6](glorot[5, 5, 6, 16])[16]*),
    //   Layer.Pool[2, 2, 2](subsampling[Float, 2, 2, 2, 16](null), LeCunnTanh),
    //   Layer.Dense[400, 120](Neuron.xavier[400, 120](LeakyReLU())*),
    //   Layer.Dense[120, 84](Neuron.xavier[120, 84](LeakyReLU())*),
    //   Layer.Dense[84, 10](Neuron.xavier[84, 10](LeakyReLU())*),
    //   Layer.Softmax[10](Xavier(10, 10))
    // )

    val ln = Network[FL, FB, KL, KB, KS, D, PL, PB, PS, 4, N, 8](
      loss = CCE[10](),
      learningRate = 0.01,
      Layer.Convolution[5, 5, 1, 6](LeakyReLU(), Kernel[Float, 5, 5, 1](kaiming[5, 5, 1]())[6]*),
      Layer.Pool[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue), Linear()),
      Layer.Convolution[5, 5, 6, 16](LeakyReLU(), Kernel[Float, 5, 5, 6](kaiming[5, 5, 6]())[16]*),
      Layer.Pool[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue), Linear()),
      Layer.Dense[400, 120](Neuron.kaiming[400, 120](LeakyReLU())*),
      Layer.Dense[120, 84](Neuron.kaiming[120, 84](LeakyReLU())*),
      Layer.Dense[84, 10](Neuron.kaiming[84, 10](LeakyReLU())*),
      Layer.Softmax[10](Kaiming(10))
    )

    val data = Data[32, 32, 1, 10]({
      val read = 60000 // 1100 // 20000
      val train = 60000 // 100 // 20000

      val drop = rnd.nextInt(read-train+1)

      for
        case image @ Image(label: Int, _) <- rnd.shuffle(trainMNIST("./data/MNIST", drop+train, false).drop(drop))
      yield
        Input(image) -> OneHotOutput(label)
    }*)

    val batch = 100 // 1
    val epochs = 20 // 2

    print(s"Training ${data.io.size} images in $batch batches and $epochs epochs...")

    ln(data, batch, epochs) {
      case (count, done) if done % 5000 == 0 =>
        print(s" Passing through $count epochs and $done images...")
      case _ =>
    }

    println(" Done.")

    val read = 10000 // 1010
    val test = 10000

    val drop = rnd.nextInt(read-test+1)

    var correct = 0

    for
      case image @ Image(label: Int, _) <- rnd.shuffle(testMNIST("./data/MNIST", drop+test, false).drop(drop))
      Output(answer) = ln(Input(image))
      if label == answer.toSeq.zipWithIndex.maxBy(_._1)._2
    do
      correct += 1

    val accuracy = ((1f * correct / test) * 100).toInt

    println(s"Accuracy: $accuracy%")
  }


object NetworkSuite:

  val rnd = scala.util.Random

  def kaiming[L <: Int: ValueOf, // kernel Length
              B <: Int: ValueOf, // kernel Breadth
              I <: Int: ValueOf, // Input channels
              ](α: Double = 0.01): Initialization = Kaiming(valueOf[L] * valueOf[B] * valueOf[I], α)

  def glorot[L <: Int: ValueOf, // kernel Length
             B <: Int: ValueOf, // kernel Breadth
             I <: Int: ValueOf, // Input channels
             O <: Int: ValueOf, // Output filters
             ]: Initialization = Glorot(valueOf[L] * valueOf[B] * valueOf[I], valueOf[L] * valueOf[B] * valueOf[O])

  def xavier[L <: Int: ValueOf, // kernel Length
             B <: Int: ValueOf, // kernel Breadth
             I <: Int: ValueOf, // Input channels
             O <: Int: ValueOf, // Output filters
             ]: Initialization = Xavier(valueOf[L] * valueOf[B] * valueOf[I], valueOf[L] * valueOf[B] * valueOf[O])

  // https://github.com/javagl/MnistReader

  import java.io.{ DataInputStream, FileInputStream, InputStream }
  import java.nio.file.{ Path, Paths }
  import java.util.zip.GZIPInputStream

  def trainMNIST(path: String, size: Int = Int.MaxValue, zero: Boolean = true): Seq[Image[Float, 32, 32, 1]] =
    MNIST(path, "train", size, zero)

  def testMNIST(path: String, size: Int = Int.MaxValue, zero: Boolean = true): Seq[Image[Float, 32, 32, 1]] =
    MNIST(path, "t10k", size, zero)

  def MNIST(path: String, name: String, size: Int, zero: Boolean): Seq[Image[Float, 32, 32, 1]] =

    val imagesFileName = s"$name-images-idx3-ubyte.gz"
    val labelsFileName = s"$name-labels-idx1-ubyte.gz"

    val compressedImagesInputStream = FileInputStream(Paths.get(path).resolve(imagesFileName).toFile)
    val compressedLabelsInputStream = FileInputStream(Paths.get(path).resolve(labelsFileName).toFile)

    val decompressedImagesInputStream = GZIPInputStream(compressedImagesInputStream)
    val decompressedLabelsInputStream = GZIPInputStream(compressedLabelsInputStream)

    val imagesDataInputStream = DataInputStream(decompressedImagesInputStream)
    val labelsDataInputStream = DataInputStream(decompressedLabelsInputStream)

    try
      assert(imagesDataInputStream.readInt == 0x803)
      assert(labelsDataInputStream.readInt == 0x801)

      val numberOfImages = imagesDataInputStream.readInt
      val numberOfLabels = labelsDataInputStream.readInt

      assert(numberOfImages == numberOfLabels)

      val numRows = imagesDataInputStream.readInt
      val numCols = imagesDataInputStream.readInt

      assert(numRows == 28 && numCols == 28)

      for
        _ <- 0 until (size min numberOfImages)
      yield
        val label = labelsDataInputStream.readByte() & 0xFF
        val data = Array.fill(numRows * numCols)(0.toByte)

        var offset = 0
        while offset < data.length
        do
          val read = imagesDataInputStream.read(data, offset, data.length - offset)
          assert(read > 0)
          offset += read

        // https://dilithjay.com/blog/read-mnist-images-in-java

        var image = Array.fill(32, 32, 1)(if zero then 0f else -1f)

        for
          i <- 0 until 784
          x = (data(i) & 0xFF) / 255f
        do
          image((32-28)/2 + i/28)((32-28)/2 + i%28)(0) = if zero then x else 2*x-1

        Image[Float, 32, 32, 1](label, Tensor[Float, 32, 32, 1](image))

    finally
      labelsDataInputStream.close
      imagesDataInputStream.close
