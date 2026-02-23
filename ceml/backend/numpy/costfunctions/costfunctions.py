# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import approx_fprime

from ....costfunctions import CostFunctionDifferentiable
from .losses import custom_dist, l1, l2, lmad, min_of_list, negloglikelihood


class CostFunctionDifferentiableNumpy(CostFunctionDifferentiable):
    """
    Base class of differentiable cost functions implemented in NumPy.
    Uses numerical gradient approximation via scipy.optimize.approx_fprime.
    """

    def __init__(self, input_to_output=None, **kwds):
        super().__init__(input_to_output=input_to_output, **kwds)

    def grad(self, mask=None):
        """
        Computes the gradient with respect to the input using numerical approximation.

        Parameters
        ----------
        mask : `numpy.array`, optional
            A mask that is multiplied elementwise to the gradient.
            If `mask` is None, the gradient is not masked.
            The default is None.

        Returns
        -------
        `callable`
            The gradient function.
        """
        epsilon = np.sqrt(np.finfo(np.float64).eps)

        if mask is not None:

            def masked_grad(x):
                x = np.asarray(x, dtype=np.float64)
                g = approx_fprime(x, self.score, epsilon)
                return np.multiply(g, mask)

            return masked_grad

        def unmasked_grad(x):
            x = np.asarray(x, dtype=np.float64)
            return approx_fprime(x, self.score, epsilon)

        return unmasked_grad


class TopKMinOfListDistCost(CostFunctionDifferentiableNumpy):
    """
    Computes the sum of the distances to the k closest samples.
    """

    def __init__(self, dist, samples, k, input_to_output=None, **kwds):
        self.dist = dist
        self.samples = samples
        self.k = k

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, x):
        d = np.array([self.dist(x, x1) for x1 in self.samples])
        return np.sum(d[np.argsort(d)[: self.k]])


class DummyCost(CostFunctionDifferentiableNumpy):
    """
    Dummy cost function - always returns zero.
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def score_impl(self, x):
        return 0.0


class L1Cost(CostFunctionDifferentiableNumpy):
    """
    L1 cost function.
    """

    def __init__(self, x_orig, input_to_output=None, **kwds):
        self.x_orig = x_orig

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, x):
        return l1(x, self.x_orig)


class L2Cost(CostFunctionDifferentiableNumpy):
    """
    L2 cost function.
    """

    def __init__(self, x_orig, input_to_output=None, **kwds):
        self.x_orig = x_orig

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, x):
        return l2(x, self.x_orig)


class LMadCost(CostFunctionDifferentiableNumpy):
    """
    Manhattan distance weighted feature-wise with the inverse median absolute deviation (MAD).
    """

    def __init__(self, x_orig, mad, input_to_output=None, **kwds):
        self.x_orig = x_orig
        self.mad = mad

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, x):
        return lmad(x, self.x_orig, self.mad)


class MinOfListDistCost(CostFunctionDifferentiableNumpy):
    """
    Minimum distance to a list of data points.
    """

    def __init__(self, dist, samples, input_to_output=None, **kwds):
        self.dist = dist
        self.samples = samples

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, x):
        return min_of_list([self.dist(x, x1) for x1 in self.samples])


class MinOfListDistExCost(CostFunctionDifferentiableNumpy):
    """
    Minimum distance to a list of data points.

    In contrast to :class:`MinOfListDistCost`, :class:`MinOfListDistExCost` uses a user defined metric matrix (distortion of the Euclidean distance).
    """

    def __init__(self, omegas, samples, input_to_output=None, **kwds):
        self.omegas = omegas
        self.samples = samples

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, x):
        return min_of_list([custom_dist(x, x1, omega) for x1, omega in zip(self.samples, self.omegas)])


class NegLogLikelihoodCost(CostFunctionDifferentiableNumpy):
    """
    Negative-log-likelihood cost function.
    """

    def __init__(self, input_to_output, y_target, **kwds):
        self.y_target = y_target

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, y):
        return negloglikelihood(y, self.y_target)


class SquaredError(CostFunctionDifferentiableNumpy):
    """
    Squared error cost function.
    """

    def __init__(self, input_to_output, y_target, **kwds):
        self.y_target = y_target

        super().__init__(input_to_output=input_to_output, **kwds)

    def score_impl(self, y):
        return l2(y, self.y_target)


class RegularizedCost(CostFunctionDifferentiableNumpy):
    """
    Regularized cost function.
    """

    def __init__(self, penalize_input, penalize_output, C=1.0, **kwds):
        if not isinstance(penalize_input, CostFunctionDifferentiable):
            raise TypeError(
                f"penalize_input has to be an instance of 'CostFunctionDifferentiable' but not of '{type(penalize_input)}'"
            )
        if not isinstance(penalize_output, CostFunctionDifferentiable):
            raise TypeError(
                f"penalize_output has to be an instance of 'CostFunctionDifferentiable' but not of {type(penalize_output)}"
            )

        self.penalize_input = penalize_input
        self.penalize_output = penalize_output
        self.C = C

        super().__init__(**kwds)

    def score_impl(self, x):
        return self.C * self.penalize_input(x) + self.penalize_output(x)
