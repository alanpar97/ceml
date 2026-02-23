# -*- coding: utf-8 -*-
import numpy as npx


def create_tensor(x):
    return npx.array(x, dtype=npx.float64)


def affine(x, w, b):
    return npx.dot(w, x) + b


def softmax(x):
    e_x = npx.exp(x - npx.max(x))
    return e_x / npx.sum(e_x, axis=0)


def softmax_binary(x):
    a = 1.0 / (1 + npx.exp(-1.0 * x))
    return create_tensor([a, 1.0 - a])


def normal_distribution(x, mean, variance):
    return npx.exp(-0.5 * npx.square(x - mean) / variance) / npx.sqrt(2.0 * npx.pi * variance)


def log_normal_distribution(x, mean, variance):
    return -0.5 * npx.square(x - mean) / variance - 0.5 * (2.0 + npx.pi + variance)


def log_multivariate_normal(x, mean, sigma_inv, k):
    return (
        0.5 * npx.log(npx.linalg.det(sigma_inv))
        - 0.5 * k * npx.log(2.0 * npx.pi)
        - 0.5 * npx.dot(x - mean, npx.dot(sigma_inv, (x - mean)))
    )
