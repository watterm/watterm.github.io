---
layout: post
title: "The Variational Auto-Encoder"
date: 2014-12-10 16:12:00
comments: true
---

In this post, I attempt to give a summary of the [Variational Auto-Encoder introduced
by Kingma and Welling](http://arxiv.org/abs/1312.6114).

### EM Recap

As the formulation is an extension of the probabilistic problem setting in the
Expectation Maximization algorithm, let's start with a recap with adjusted notation:

- _Goal_: 

    Maximum Likelihood or MAP estimation of $$p(X)$$.


- _Problem_: 

    This can be hard, if the data was created by a process with 
    latent variables $$Z$$. So it's actually a marginal likelihood 
    $$p(X) = \int_Z p(X|Z) p(Z) dZ$$.

- _Assumptions_:
    - If we knew $$Z$$ (which we don't), it would be easy to optimize the 
      joint probability $$p(X, Z)$$ under the posterior over the latent variables 
      $$p(Z|X)$$:

      $$
      \max \mathbb{E}_{p(Z|X)} \left[p(X,Z) \right] \label{joint}
      $$
    - _Intractability_: The posterior is rarely tractable. For complicated
      likelihood functions $$p(x|z)$$ the marginal likelihood is intractable
      as well, if we don't want to make simplifying assumptions.
    - _Dataset size_: If we have lots of data, sampling would be prohibitively
      slow, as we would need to do this for every datapoint.

### The Variational Auto-Encoder (Short version)

Let's first make the connection to the usual lingo for auto-encoders. The latent
variables $$Z$$ can be seen as a (often low-dimensional) representation
from which we can construct our data $$X$$. Estimating the latent value $$z^{(i)}$$
from a datapoint $$x^{(i)}$$, i.e. evaluating $$p(Z|X)$$, therefore corresponds
to _encode_ this sample. Constructing a datapoint from a latent value by evaluating 
$$p(X|Z)$$ means in turn to _decode_ from the latent representation.

The variational auto-encoder promises the means to carry out efficient approximate
inference of the posterior $$p(Z|X)$$ as well as the marginal $$p(X)$$, i.e.
determine the parameters for a probabilistic formulation of an encoder and a
decoder!

Since we want to capture even complicated relationships between $$X$$ and $$Z$$,
we need powerful models: Neural Networks. For training we therefore need the
gradients of the loss function $$\mathcal{L}$$. Unfortunately taking the gradient
wrt to latent variables can yield computationally expensive equations, e.g.
calculating the Hessian for a Gaussian distribution. The central trick now is
to apply a _reparameterization_ to $$Z$$. For this, let $$q_\phi (Z|X)$$ be
our approximation of the true posterior parameterized through $$\Phi$$. We can 
do this:

> Express the random variable $$Z \sim q_\phi (Z|X)$$ as a deterministic
  variable $$Z_D = g_\phi (X, \epsilon)$$ with $$g_\phi$$ being a differentiable
  transformation function and $$\epsilon \sim p(\epsilon)$$.

By decoupling the random from the deterministic part, we can now easily sample
from the parameter-independent $$p(\epsilon)$$, while the gradients through
$$g_\phi$$ can be computed quickly using backpropagation. 

Let's look at a simple example for such a reparameterization, where $$\odot$$
is the element-wise multiplication:

$$
\begin{align*}
Z \sim q_\phi(Z|X) & = \mathcal{N}(\mu, \sigma^2 I) \\
\epsilon & \sim \mathcal{N}(0, I) \\
Z_D = g_\phi(X, \epsilon) & = \mu_\phi (X) + \sigma_\phi (X) \odot \epsilon \\
\end{align*}
$$

So by formulating a diagonal Gaussian in terms of an isotropic Gaussian, we
can see that $$g$$ is easily differentiable. Also note, that $$\mu$$ and $$\sigma$$
are parameterized under $$\phi$$ given $$X$$ as input. This is the task of
the encoding network: it needs to predict the mean and standard deviation for
every datapoint. With the independently drawn $$\epsilon$$ we then have a
probabilistic recognition model that can be trained with backpropagation.

By doing so, we also gain another advantage. The lower bound in $$\eqref{joint}$$
can be formulated as a sum of the Kullback-Leibler divergence and expected 
reconstruction cost:

$$ 
\mathcal{L}(\theta, \phi; x) = - KL(q_\phi(Z|X) || p_\theta(Z)) + \mathbb{E}_{q_\phi(Z|X)}
\left[\ln p_\theta(X|Z) \right]
$$

We actually still need to sample $$\epsilon$$ to calculate $$\mathcal{L}$$. But
with compatible choices of $$q_\phi$$ and $$p_\theta$$ the KL-divergence can
be computed and differentiated without estimation. The sampling thus only effects
the reconstruction term and it seems that for practical purposes one sample is
enough in a mini-batch setting. This would result in $$L = 1$$ for the following
final cost definition for the variational auto-encoder with diagonal Gaussian
prior and posterior:

$$
\mathcal{L}(\theta, \phi; X) \simeq \frac{1}{2} \sum_{j=1}^J \left(
1 + \ln (\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right) +
\frac{1}{L} \sum_{l=1}^{L} \ln p_\theta(X|Z_D)
$$

Although this 'short' version is not so short anymore, let's summarize what we gained:
We have now a probabilistic version of an auto-encoder trainable by backpropagation,
that employs (possibly deep) neural networks as encoder and decoder and is as
such able to capture complicated relationships between latent and observed
variables, while still scaling to large datasets.
