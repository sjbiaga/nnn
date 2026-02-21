package nnn
package float

import scala.Function.const

import spire.math.{ exp => _, log => _, * }
import spire.algebra.*
import spire.implicits.*

import Util.*


/**
  * @see https://en.wikipedia.org/wiki/Activation_function
  */
enum Activation(val apply: Float => Float,
                val prime: Float => Float):

  case Linear(α: Float = 1) extends Activation(x => α*x, const(α))

  case Sign extends Activation(signum, const(0))

  case Sigmoid extends Activation(sigmoid, x => { val o = sigmoid(x); o*(1-o) })

  case Tanh extends Activation(tanh, x => { val o = tanh(x); 1-sqr(o) })

  case ReLU extends Activation(0 max _, x => if x <= 0 then 0 else 1)

  case ReLU6 extends Activation(1 max _ min 6, x => if x <= 0 || x > 6 then 0 else 1)

  case LeakyReLU(α: Float = 0.01) extends Activation(x => if x <= 0 then α*x else x,
                                                     x => if x <= 0 then α else 1)

  case SiLU extends Activation(x => x * sigmoid(x), x => { val e = exp(-x); (1+e+x*e)/sqr(1+e) })

  case HardTanh extends Activation(1 min _ max -1, x => if -1 < x && x < 1 then x else 1)

  case LeCunnTanh extends Activation(x => 1.7159f*tanh(2*x/3), x => { val o = tanh(2*x/3); 1.1439f*(1-sqr(o))  })

  case Softplus extends Activation(x => log(1+exp(x)), sigmoid)

  case ELU(α: Float = 1) extends Activation(x => if x <= 0 then α*(exp(x)-1) else x,
                                            x => if x <= 0 then α*exp(x) else 1)

  case ELiSH extends Activation(x => (if x < 0 then exp(x)-1 else x) * sigmoid(x),
                                x => (if x < 0 then 2*exp(2*x)+exp(3*x)-exp(x)
                                      else x*exp(x)+exp(2*x)+exp(x))/(exp(2*x)+2*exp(x)+1))

  case Gaussian extends Activation(x => exp(-sqr(x)), x => -2*x*exp(-sqr(x)))

  case Softmax extends Activation(x => softmax.apply(Vector[Float][1](x))(0),
                                  x => softmax.prime(Vector[Float][1](x))(0)(0))

  case LogSoftmax extends Activation(x => logsoftmax.apply(Vector[Float][1](x))(0),
                                     x => logsoftmax.prime(Vector[Float][1](x))(0)(0))

  inline def apply[N <: Int](z: Vector[Float, N]): Vector[Float, N] =
    this match
      case Softmax => softmax.apply(z)
      case LogSoftmax => logsoftmax.apply(z)
      case _ =>
        require(z.rows == 1)
        z.op(apply(_))

  inline def prime[N <: Int](z: Vector[Float, N])(k: Int): Vector[Float, N] = // unused
    this match
      case Softmax => softmax.prime(z)(k)
      case LogSoftmax => logsoftmax.prime(z)(k)
      case _ =>
        require(z.rows == 1 && k == 0)
        z.op(prime(_))
