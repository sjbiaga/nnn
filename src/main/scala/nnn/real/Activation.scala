package nnn
package real

import scala.Function.const

import spire.math.*
import spire.algebra.*
import spire.implicits.*

import Util.*


/**
  * @see https://en.wikipedia.org/wiki/Activation_function
  */
enum Activation(val apply: Real => Real,
                val prime: Real => Real):

  case Linear(α: Double = 1) extends Activation(x => α*x, const(α))

  case Sign extends Activation(signum, const(0))

  case Sigmoid extends Activation(sigmoid, x => { val o = sigmoid(x); o*(1-o) })

  case Tanh extends Activation(tanh, x => { val o = tanh(x); 1-sqr(o) })

  case ReLU extends Activation(0 max _, x => if x <= 0 then 0 else 1)

  case ReLU6 extends Activation(1 max _ min 6, x => if x < 0 || x > 6 then 0 else 1)

  case LeakyReLU(α: Double = 0.01) extends Activation(x => if x <= 0 then α*x else x,
                                                      x => if x <= 0 then α else 1)

  case SiLU extends Activation(x => x * sigmoid(x), x => { val e = exp(-x); (1+e+x*e)/sqr(1+e) })

  case HardTanh extends Activation(1 min _ max -1, x => if -1 < x && x < 1 then x else 1)

  case LeCunnTanh extends Activation(x => Real(1.7519)*tanh(Real(2)*x/Real(3)), x => Real(1.7519)*(Real(2)/Real(3))*(Real(1) - sqr(tanh(Real(2)*x/Real(3)))))

  case Softplus extends Activation(x => log(1+exp(x)), sigmoid)

  case ELU(α: Double = 1) extends Activation(x => if x <= 0 then α*(exp(x)-1) else x,
                                             x => if x <= 0 then α*exp(x) else 1)

  case ELiSH extends Activation(x => (if x < 0 then exp(x)-1 else x) * sigmoid(x),
                                x => (if x < 0 then 2*exp(2*x)+exp(3*x)-exp(x)
                                      else x*exp(x)+exp(2*x)+exp(x))/(exp(2*x)+2*exp(x)+1))

  case Gaussian extends Activation(x => exp(-sqr(x)), x => -2*x*exp(-sqr(x)))

  case Softmax extends Activation(x => softmax.apply(Vector[Real][1](x))(0),
                                  x => softmax.prime(Vector[Real][1](x))(0)(0))

  case LogSoftmax extends Activation(x => logsoftmax.apply(Vector[Real][1](x))(0),
                                     x => logsoftmax.prime(Vector[Real][1](x))(0)(0))

  inline def apply[N <: Int](z: Vector[Real, N]): Vector[Real, N] =
    this match
      case Softmax => softmax.apply(z)
      case LogSoftmax => logsoftmax.apply(z)
      case _ =>
        require(z.rows == 1)
        z.op(apply(_))

  inline def prime[N <: Int](z: Vector[Real, N])(k: Int): Vector[Real, N] = // unused
    this match
      case Softmax => softmax.prime(z)(k)
      case LogSoftmax => logsoftmax.prime(z)(k)
      case _ =>
        require(z.rows == 1 && k == 0)
        z.op(prime(_))
