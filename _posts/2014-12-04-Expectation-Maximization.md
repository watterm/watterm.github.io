---
layout: post
title: "Expectation Maximization"
date: 2014-12-04 14:46:01
comments: true
---
As a first post, I'd like to summarize the Expectaction Maximization (EM) algorithm,
as described in Bishop 2006 (Bibtex plugin?).

### The Essence

We find ourselves in the usual `Maximum Likelihood Estimation` setting, i.e. we
are trying to maximize the log likelihood $$\ln p(X \vert \Theta)$$, where
$$X$$ is the design matrix of the observed variables and $$\Theta$$ are the
parameters of the model we are optimizing. It turns out that it can be hard to
fit a model if the generative process depends on other hidden, _latent_
variables $$Z$$. The EM algorithm therefore tries to optimize the joint
distribution of observable and latent variables. In short:

- _Assumptions_:
  1. We get the Maximum Likelihood by marginalizing over latent variables: 

        $$
        \begin{align}
        \ln p(X \vert \Theta) = \ln \int_Z p(X, Z \vert \Theta) dZ\\
        \end{align}
        $$

  1. Maximizing the incomplete-data log-likelihood $$\ln p(X \vert \Theta)$$ is
    hard.
  1. Maximizing the complete-data log-likelihood $$\ln p(X, Z \vert \Theta)$$ is easy.
- _Problems_:
  1. We can't observe $$Z$$ and could only estimate it through its posterior
    $$p(Z \vert X, \Theta)$$.
  1. The $$\ln$$ outside of the marginalization in $$\ln \int_Z p(X, Z \vert \Theta) dZ$$
    creates complicated expressions.
- _Solution_:

  > Maximize the expectation of the complete-data log likelihood $$p(X, Z \vert
  > \Theta)$$ under the
  > posterior distribution of the latent variables $$p(Z \vert X, \Theta)$$.

This is the core idea of EM and where it derives its name from. If we calculate
the expectation, we will see that it becomes feasible to maximize the joint
distribution of observed and latent variables. This gives rise to an iterative 
2-step scheme, that consists of the E-Step and the M-Step, i.e. calculating the 
_Expectation_ and then _Maximizing_ it wrt. to the model parameters $$\Theta$$. 
Formally:

1. `E-Step`: 

    $$
    \begin{align*}
    \mathcal{Q}(\Theta, \Theta^{\text{old}}) &= 
        \mathbb{E}_{p(Z\vert X, \Theta^{\text{old}})} \left[ \ln p(X, Z \vert \Theta) \right] \\ 
        &= \int_Z p(Z \vert X, \Theta^{\text{old}}) \ln p(X, Z \vert \Theta)dZ
    \end{align*}
    $$

2. `M-Step`:

    $$
    \Theta^{\text{new}} = arg\,max_{\Theta} Q(\Theta, \Theta^{\text{old}}) \notag
    $$

Taking the expectation moved the $$\ln$$ inside the integral in the
$$\mathcal{Q}$$-function and - according to
our assumptions - maximizing the complete-data log-likelihood is now tractable.
With known parameters $$\Theta^{\text{old}}$$ the distribution over $$Z$$ is
calculated and can be used for the marginalization in the _E-Step_. By solving
for the optimal values of $$\Theta$$ of the $$\mathcal{Q}$$-function and using these
as $$\Theta^{\text{old}}$$ in the next iteration, we are guaranteed to find
a local maximum of the likelihood.

### A Change in Perspective

Difficulties in the first part arise concerning $$p(Z \vert X, \Theta)$$. We 
have never assumed that this posterior is easy to model (or known at all). Say
that the distribution is either too complex or we want to restrict ourselves to
a family of distributions that can be calculated quickly, but is only an approximation
to the true posterior. Let's call such a distribution $$q(Z)$$. In this part we'll
see a different, information-theoretic formulation of the original problem and
the EM-algorithm from the previous section.

We keep our assumptions, but model the log likelihood like this:

$$
\ln p(X \vert \Theta) = \mathcal{L}(q, \Theta) + KL(q \vert \vert p)
$$

with

$$
\begin{align}
\mathcal{L}(q, \Theta) &= \int_Z q(Z) \ln \left( \frac{p(X, Z \vert \Theta)}{q(Z)} \right) dZ \label{bound} \\
KL(q \vert \vert p) &= - \int_Z q(Z) \ln \left( \frac{p(Z \vert X, \Theta)}{q(Z)} \right) dZ \label{kl}
\end{align}
$$
Let's first have a look at the shape of $$\eqref{bound}$$ and $$\eqref{kl}$$.
They differ in sign and the term in the numerator. $$\mathcal{L}(q, \Theta)$$
is the lower bound of the log likelihood (which we'll see in a minute) and depends
on the previously stated function $$q$$. $$KL(q \vert \vert p)$$ is the [Kullback-Leibler 
divergence](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), 
which represents an asymmetric measure of the difference of the
two probability distributions. 

Warning: The next part can be rather hard to understand. If you're only interested
in the algorithmic formulation, you may safely skip to the [next section](#the-em-algorithm-again).

Let's make sure the sum of those two terms is really the log likelihood:

$$
\begin{align}
\ln p(X \vert \Theta) &= \mathcal{L}(q, \Theta) + KL(q \vert \vert p) \notag \\
    &= \int_Z q(Z) \ln \left(\frac{p(X, Z  \vert \Theta)}{q(Z)} \right) dZ - \notag \\
    &  \hspace{4em} \int_Z q(Z) \ln \left(\frac{p(Z \vert X, \Theta)}{q(Z)} \right) dZ \notag \\
    &= \int_Z q(Z) \left( \ln \left(\frac{p(Z \vert X, \Theta)}{q(Z)} \right) 
            + \ln p(X | \Theta) \right) dZ - \notag \\
    & \hspace{4em} \int_Z q(Z) \ln \left(\frac{p(Z \vert X, \Theta)}{q(Z)} \right) dZ \notag \\
    &= \int_Z q(Z) \ln p(X|\Theta) dZ \label{q} \\
    &= \ln p(X|\Theta) \notag
\end{align}
$$

Despite the messy formulaes, there is not much magic going on. After inserting
the terms for $$\mathcal{L}$$ and $$KL$$ we use 
$$\ln p(X, Z|\Theta) = \ln p(Z|X, \Theta) + \ln p(X|\Theta)$$ and $$\int_Z q(Z) = 1$$
to get rid of unwanted terms. If you look closely, you can see that one of those
terms is $$-KL(q||p)$$ contained in $$\mathcal{L}(q, \Theta)$$!

We should now have a look at equation $$\eqref{q}$$ and recall two properties of
the Kullback-Leibler divergence:

1. It always satisfies $$KL(q\vert \vert p) \geq 0$$. This gives rise to $$\mathcal{L}$$ being the lower bound:

   $$
   \mathcal{L}(q, \Theta) \leq \ln p(X|\Theta) \notag
   $$

2. It also satisfies $$KL(q||p) = 0 \Leftrightarrow q(Z) = p(Z|X, \Theta)$$. That is,
   the log likelihood is recovered if $$q$$ is the true posterior and the lower
   bound is maximal.

Recall that our goal is to maximize the log likelihood and that we can adjust
$$q$$ and $$\Theta$$ in this setting. 
So why this formulation as $$\mathcal{L} + KL$$? We have seen that this optimizes
the log likelihood, but the KL-divergence requires the true posterior over the latent variables
$$p(Z|X, \Theta)$$,  which we don't know and therefore can't optimize! But we do
know per our assumption that the joint probability of observed and latent variables
$$p(X, Z| \Theta)$$ in $$\mathcal{L}$$ is computable, which makes it optimizable.


### The EM Algorithm (again)

So we maximize $$\mathcal{L}(q, \Theta)$$.

If we maximize wrt. to $$q$$, we need to consider the interaction with the joint
distribution $$p(X, Z|\Theta)$$. We apply now the same idea as in the first part,
i.e. we fix it as $$\Theta^{\text{old}}$$, so we can find a maximizing q. The largest
value for $$\mathcal{L}$$ will occur when $$q$$ is equal to $$p$$, as stated above.

Maximizing wrt. to $$\Theta$$ will now necessarily increase the log likelihood.
Unfortunately $$q$$ was determined with an old set of parameters, it won't
correspond to the posterior with our shiny new $$\Theta$$. Therefore the
KL-divergence will increase again, though not as much as we gain from this step
(recall that it is already implicitly modelled!).

What we just described is actually just the same iterative approach as before.


1. `E-Step`:

    $$
    q' = arg\,max_q \mathcal{L}(q, \Theta') \notag
    $$

2. `M-Step`:
    
    $$
    \Theta' = arg\,max_\Theta \mathcal{L}(q', \Theta) \notag
    $$

Iterating these to steps will continually improve the lower bound until a local
maximum is reached. 

### Tying up

We have seen two different ways to model our assumptions. But why should they
be the same? Let's try plugging in the posterior estimated by the E-Step with 
$$\Theta^{\text{old}}$$ into the lower bound:

$$
\begin{align*}
\mathcal{L}(p, \Theta) &= \int_Z p(Z|X, \Theta^{\text{old}}) \ln \left(
    \frac{p(X, Z|\Theta)}{p(Z|X, \Theta^{\text{old}})} \right) dZ \\
    &= \int_Z p(Z|X, \Theta^{\text{old}}) \ln p(X, Z | \Theta)dZ -
        \int_Z p(Z|X, \Theta^{\text{old}}) \ln p(Z|X, \Theta^{\text{old}}) \\
    &= \mathbb{E}_{p(Z, X|\Theta^{\text{old}})}\left[\ln p(X, Z|\Theta) \right] + h(p) \\
    &= Q(\Theta, \Theta^{\text{old}}) + \text{const}
\end{align*}
$$

So we're able to recover the formulation as the expectation of the joint probability
under the posterior distribution with an additional entropy term.


With this I'll end the first post, which is now a lot longer and a lot less
transparent than I intended, but I'm still practicing :p
