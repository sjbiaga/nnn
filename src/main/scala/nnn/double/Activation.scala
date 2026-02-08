package nnn
package double

import scala.Function.const

import spire.math.*
import spire.algebra.*
import spire.implicits.*

import Util.*


/**
  * @see https://en.wikipedia.org/wiki/Activation_function
  */
enum Activation(val apply: Double => Double,
                val prime: Double => Double):

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

  case LeCunnTanh extends Activation(x => 1.7519d*tanh(2d*x/3d), x => 1.7519d*(2d/3d)*(1d - sqr(tanh(2d*x/3d))))

  case Softplus extends Activation(x => log(1+exp(x)), sigmoid)

  case ELU(α: Double = 1) extends Activation(x => if x <= 0 then α*(exp(x)-1) else x,
                                             x => if x <= 0 then α*exp(x) else 1)

  case ELiSH extends Activation(x => (if x < 0 then exp(x)-1 else x) * sigmoid(x),
                                x => (if x < 0 then 2*exp(2*x)+exp(3*x)-exp(x)
                                      else x*exp(x)+exp(2*x)+exp(x))/(exp(2*x)+2*exp(x)+1))

  case Gaussian extends Activation(x => exp(-sqr(x)), x => -2*x*exp(-sqr(x)))

  case Softmax extends Activation(x => softmax.apply(Vector[Double][1](x))(0),
                                  x => softmax.prime(Vector[Double][1](x))(0)(0))

  inline def apply[N <: Int](z: Vector[Double, N]): Vector[Double, N] =
    this match
      case Softmax => softmax.apply(z)
      case _ =>
        require(z.rows == 1)
        z.op(apply(_))

  inline def prime[N <: Int](z: Vector[Double, N])(k: Int): Vector[Double, N] = // unused
    this match
      case Softmax => softmax.prime(z)(k)
      case _ =>
        require(z.rows == 1 && k == 0)
        z.op(prime(_))
