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
import LeNetSuite.*


class LeNetSuite extends FunSuite:

  override val munitTimeout = 16.hours

  test("CNN:MNIST https://en.wikipedia.org/wiki/LeNet#/media/File:LeNet-5_architecture_block_diagram.svg") {

    // TYPE CHECKING

    def C[C <: Int, O <: Int: ValueOf](layer: Layer.Convolutional[?, ?, ?, O])
                                      (using kl: ValueOf[KL[C]], kb: ValueOf[KB[C]], d: ValueOf[D[C-1]]): layer.type =
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

    type FL[C <: Int] = C match { case 0 => 32 case 1 => 28 case 2 => 14 case 3 => 10 case 4 => 5 } // Feature Length

    type FB[C <: Int] = C match { case 0 => 32 case 1 => 28 case 2 => 14 case 3 => 10 case 4 => 5 } // Feature Breadth

    type KL[C <: Int] = C match { case 1 | 3 => 5 } // Kernel Length

    type KB[C <: Int] = C match { case 1 | 3 => 5 } // Kernel Breadth

    type KS[C <: Int] = C match { case 1 | 3 => 1 } // Kernel Stride

    type D[C <: Int] = C match { case 0 => 1 case 1 | 2 => 6 case 3 | 4 => 16 } // Feature/Kernel/Pooling Depth

    type PL[C <: Int] = C match { case 2 | 4 => 2 } // Pooling Length

    type PB[C <: Int] = C match { case 2 | 4 => 2 } // Pooling Breadth

    type PS[C <: Int] = C match { case 2 | 4 => 2 } // Pooling Stride

    type N[L <: Int] = L match { case 4 => FL[4]*FB[4]*D[4] case 5 => 120 case 6 => 84 case 7 | 8 => 10 }

    given List[Boolean] = false :: true :: false :: true :: false :: Nil

    // true     FL   FB   KL   KB   KS   pad  D
    // false    FL   FB   PL   PB   PS   pad  D
    given List[(Int, Int, Int, Int, Int, Int, Int)] = (32, 32, 0, 0, 0, 0,  1) // image
                                                   :: (28, 28, 5, 5, 1, 0,  6) // convolutional
                                                   :: (14, 14, 2, 2, 2, 0,  6) // subsampling
                                                   :: (10, 10, 5, 5, 1, 0, 16) // convolutional
                                                   :: ( 5,  5, 2, 2, 2, 0, 16) // subsampling
                                                   :: Nil

    given List[Int] = 400/*=5*5*16*/ :: 120 :: 84 :: 10 :: 10 :: Nil

    print("Initializing LeNet CNN...")

    // val ln = Network[FL, FB, KL, KB, KS, D, PL, PB, PS, 4, N, 8](
    //   loss = CCE[10](),
    //   learningRate = 0.01,
    //   C[1, 6](Layer.Convolutional[5, 5, 1, 6](LeakyReLU(), Kernel[Float, 5, 5, 1](glorot[5, 5, 1, 6])[6]*)),
    //   P[2](Layer.Pooling[2, 2, 2](subsampling[Float, 2, 2, 2, 6](null), LeCunnTanh)),
    //   C[3, 16](Layer.Convolutional[5, 5, 6, 16](LeakyReLU(), Kernel[Float, 5, 5, 6](glorot[5, 5, 6, 16])[16]*)),
    //   P[4](Layer.Pooling[2, 2, 2](subsampling[Float, 2, 2, 2, 16](null), LeCunnTanh)),
    //   D[5](Layer.Dense[400, 120](Neuron.xavier[400, 120](LeakyReLU())*)),
    //   D[6](Layer.Dense[120, 84](Neuron.xavier[120, 84](LeakyReLU())*)),
    //   D[7](Layer.Dense[84, 10](Neuron.xavier[84, 10](LeakyReLU())*)),
    //   D[8](Layer.Softmax[10](Xavier(10, 10)))
    // )

    val ln = Network[FL, FB, KL, KB, KS, D, PL, PB, PS, 4, N, 8](
      loss = CCE[10](),
      learningRate = 0.01,
      C[1, 6](Layer.Convolutional[5, 5, 1, 6](LeakyReLU(), Kernel[Float, 5, 5, 1](kaiming[5, 5, 1]())[6]*)),
      P[2](Layer.Pooling[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue))),
      C[3, 16](Layer.Convolutional[5, 5, 6, 16](LeakyReLU(), Kernel[Float, 5, 5, 6](kaiming[5, 5, 6]())[16]*)),
      P[4](Layer.Pooling[2, 2, 2](max[Float, 2, 2, 2](Float.MinValue))),
      D[5](Layer.Dense[400, 120](Neuron.kaiming[400, 120](LeakyReLU())*)),
      D[6](Layer.Dense[120, 84](Neuron.kaiming[120, 84](LeakyReLU())*)),
      D[7](Layer.Dense[84, 10](Neuron.kaiming[84, 10](LeakyReLU())*)),
      D[8](Layer.Softmax[10](Kaiming(10)))
    )

    println(" done.")

    val data = Data[32, 32, 1, 10]({
      val read = 1010 // 60000 // 20000
      val train = 10 // 60000 // 20000

      val drop = rnd.nextInt(read-train+1)

      for
        case image @ Image(label: Int, _) <- rnd.shuffle(trainMNIST("./data/MNIST", drop+train, false).drop(drop))
      yield
        Input(image) -> OneHotOutput(label)
    }*)

    val batch = 1 // 100
    val epochs = 2 // 20

    print(s"Training ${data.io.size} images in $batch batches and $epochs epochs...")

    ln(data, batch, epochs) {
      case (count, done) if count % 5 == 0 && done % 5000 == 0 =>
        print(s" Passing through $count epochs and $done images...")
      case _ =>
    }

    println(" done.")

    val read = 1010 // 10000
    val test = 10 // 10000

    val drop = rnd.nextInt(read-test+1)

    print(s"Testing $test images...")

    var correct = 0

    for
      case image @ Image(label: Int, _) <- rnd.shuffle(testMNIST("./data/MNIST", drop+test, false).drop(drop))
      Output(answer) = ln(Input(image))
      if label == answer.toSeq.zipWithIndex.maxBy(_._1)._2
    do
      correct += 1

    println(" done.")

    val accuracy = ((1f * correct / test) * 100).toInt

    assert(accuracy >= 98, s"Accuracy: $accuracy% < 98%")
  }


object LeNetSuite:

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

    val (imagesDataInputStream, labelsDataInputStream) =
      try

        val imagesFileName = s"$name-images-idx3-ubyte.gz"
        val labelsFileName = s"$name-labels-idx1-ubyte.gz"

        val compressedImagesInputStream = FileInputStream(Paths.get(path).resolve(imagesFileName).toFile)
        val compressedLabelsInputStream = FileInputStream(Paths.get(path).resolve(labelsFileName).toFile)

        val decompressedImagesInputStream = GZIPInputStream(compressedImagesInputStream)
        val decompressedLabelsInputStream = GZIPInputStream(compressedLabelsInputStream)

        DataInputStream(decompressedImagesInputStream) ->
        DataInputStream(decompressedLabelsInputStream)

      catch t =>

        println("Cannot open MNIST data files: please consult README.md for instructions!")
        throw t

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
