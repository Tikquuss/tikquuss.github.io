---
title: "Grokking Beyond the Euclidean Norm of Model Parameters"
date: "2025-07-06"
category: "Research Notes"
image: "/images/publications/grokking-norm-comparison.png"
tags:
  - Grokking
  - Regularization
  - Sparsity
  - Low-Rank Recovery
  - Implicit Bias
excerpt: "Why grokking is governed by the property favored late in training—and why that property need not be the Euclidean norm."
---

<p class="article-lede">The Euclidean norm explains some grokking experiments, especially those driven by weight decay, but it is not a universal measure of complexity. The relevant quantity is the property favored late in training.</p>

This post develops the main idea of our ICML 2025 paper, **Grokking Beyond the Euclidean Norm of Model Parameters**. We will use the memorization time $t_1$ and the generalization time $t_2$ defined in [What Is Grokking?](/blog/what-is-grokking/).

> **Paper.** Pascal Jr. Tikeng Notsawo, Guillaume Dumas, and Guillaume Rabusseau, *Grokking Beyond the Euclidean Norm of Model Parameters*, ICML 2025. [arXiv](https://arxiv.org/abs/2506.05718) · [OpenReview](https://openreview.net/forum?id=FRjRuSWF3e)

## Why the Euclidean norm cannot be universal

The **LU mechanism** gives one explanation of grokking. As the Euclidean norm of the parameters decreases, the training loss follows an L-shaped curve: it falls and remains low. The test loss follows a U-shaped curve: it is low only in a “Goldilocks” range of parameter norms. With large initialization and weak weight decay, the model can therefore fit the training data quickly, then travel slowly toward this range and generalize much later.

This picture is useful, but a raw parameter norm depends on how we parameterize the same function. Consider

$$
\mathbf y(\mathbf x)=\mathbf B\,\phi(\mathbf A\mathbf x),
$$

where $\phi$ is positive-$L$-homogeneous: $\phi(\lambda z)=\lambda^L\phi(z)$ for every $\lambda>0$. ReLU, for instance, is positive-$1$-homogeneous. The reparameterization

$$
\mathbf A\longmapsto\lambda\mathbf A,
\qquad
\mathbf B\longmapsto\lambda^{-L}\mathbf B
$$

does not change the predictor, since

$$
\frac{\mathbf B}{\lambda^L}\phi(\lambda\mathbf A\mathbf x)
=
\mathbf B\phi(\mathbf A\mathbf x).
$$

However, its squared parameter norm becomes

$$
\lambda^2\|\mathbf A\|_F^2
+
\lambda^{-2L}\|\mathbf B\|_F^2,
$$

which can be made arbitrarily large without changing the function. A particular Euclidean shell cannot characterize all generalizing predictors.

There is also a direct experimental objection. We trained the same modular-addition MLP with a layerwise $\ell_1$ norm, Frobenius norm, or nuclear norm. If $\mathcal W$ is the set of its weight matrices, these regularizers are

$$
h_1=\sum_{\mathbf W\in\mathcal W}\sum_{i,j}|W_{ij}|,
\qquad
h_2=\sum_{\mathbf W\in\mathcal W}\|\mathbf W\|_F,
\qquad
h_*=\sum_{\mathbf W\in\mathcal W}\|\mathbf W\|_*.
$$

The first sum is the entrywise $\ell_1$ norm.

All three induce delayed generalization. Under $\ell_1$ regularization, the Euclidean norm of the full parameter vector can even increase through generalization.

<figure class="article-figure article-figure-wide">
  <img src="/images/publications/grokking-norm-comparison.png" width="3123" height="1533" alt="Grokking under L1, L2, and nuclear-norm regularization, with the three norms tracked under L1 regularization" loading="lazy" />
  <figcaption>Addition modulo 97 under ℓ₁, layerwise Frobenius, and nuclear-norm regularization. Top: training and test accuracy. Bottom: under ℓ₁ regularization, the ℓ₁, Euclidean, and nuclear norms of the parameters. The model generalizes even when its Euclidean norm increases.</figcaption>
</figure>

The conclusion is not that the LU picture is useless. It is that the horizontal axis must represent the inductive bias that is active in the late phase. Sometimes this is the Euclidean norm; sometimes it is sparsity, low rank, smoothness, or a property induced implicitly by the parameterization.

## From the kernel regime to the rich regime

Another explanation describes grokking as a transition from lazy, kernel-like dynamics to feature learning. With a sufficiently large initialization, a neural network first behaves approximately like its linearization around initialization. Continued training can eventually leave this kernel regime and enter a rich regime in which the representation changes substantially.

This describes an important change in the dynamics, but the change alone does not imply that the model has learned the intended rule. A late transition generalizes only when the bias of the rich regime is aligned with the target. We will return to this point in “grokking without understanding.”

## A property-based mechanism

Let $\mathbf x\in\mathbb R^p$ denote the parameters being optimized. We consider

$$
f(\mathbf x)=g(\mathbf x)+\beta h(\mathbf x),
$$

where $g:\mathbb R^p\to[0,\infty)$ is the training loss, $h:\mathbb R^p\to[0,\infty)$ measures the property favored by regularization, and $\beta>0$ is its strength. We write $\partial h(\mathbf x)$ for the convex subdifferential of $h$ at $\mathbf x$. Subgradient descent with step size $\alpha>0$ gives

$$
\mathbf x^{(t+1)}
=
\mathbf x^{(t)}
-\alpha\bigl(G(\mathbf x^{(t)})+\beta H(\mathbf x^{(t)})\bigr),
\qquad
G=\nabla g,
\quad
H(\mathbf x^{(t)})\in\partial h(\mathbf x^{(t)}).
$$

For small $\beta$, the dynamics have two time scales.

1. **Memorization.** Initially, $G$ dominates $\beta H$. The iterates remain close to $\mathbf x^{(0)}$ and rapidly reduce $g$.
2. **Generalization.** Once $g$ and $G$ are small, the slower term $\beta H$ becomes visible. It moves the solution toward smaller values of $h$ while the training loss remains small.

The paper formalizes the first phase with a local condition. For $r>0$, define

$$
B(\mathbf x,r):=\{\mathbf y:\|\mathbf y-\mathbf x\|_2\leq r\},
$$

and the Chatterjee–Łojasiewicz constant

$$
\chi(g,\mathbf x,r)
:=
\inf_{\substack{\mathbf y\in B(\mathbf x,r)\\g(\mathbf y)\neq0}}
\frac{\|\nabla g(\mathbf y)\|_2^2}{g(\mathbf y)}.
$$

We say that $g$ is $r$-CL at $\mathbf x$ when

$$
4g(\mathbf x)<r^2\chi(g,\mathbf x,r).
$$

This condition only concerns a neighbourhood of the initialization. Under the regularity assumptions in Theorem 2.1 of the paper, if $g$ is $r$-CL at $\mathbf x^{(0)}$, then sufficiently small $\alpha$ and $\beta$ produce the two phases above. For some constant $C>0$, the first reaches $g(\mathbf x^{(t_1)})\leq\epsilon_g$ for any attainable precision $\epsilon_g=\Omega(\beta^C)$, while staying in $B(\mathbf x^{(0)},r)$. For fixed tolerance and distance to the solution set, the sufficient late-phase horizon contains the factor $1/(\alpha\beta)$.

Here is the precise late-phase statement we will prove. Let

$$
\Theta_f:=\operatorname*{argmin}_{\mathbf x}f(\mathbf x),
\qquad
f^*:=\inf_{\mathbf x}f(\mathbf x),
$$

and assume that $\Theta_f$ is nonempty.

Assume $g^*:=\inf_{\mathbf x}g(\mathbf x)=0$. When zero loss is attained, define

$$
\Theta_g:=\{\mathbf x:g(\mathbf x)=0\},
\qquad
h_g^*:=\inf_{\mathbf x\in\Theta_g}h(\mathbf x).
$$

We use

$$
\operatorname{dist}(\mathbf x,\Theta_f)
:=
\inf_{\mathbf u\in\Theta_f}\|\mathbf x-\mathbf u\|_2.
$$

Fix integers $t_2>t_1$ and, for $t_1\leq t<t_2$, write

$$
F(\mathbf x^{(t)}):=G(\mathbf x^{(t)})+\beta H(\mathbf x^{(t)}).
$$

Assume throughout this interval that

$$
f(\mathbf u)
\geq
f(\mathbf x^{(t)})
+\left\langle F(\mathbf x^{(t)}),\mathbf u-\mathbf x^{(t)}\right\rangle
\qquad
\text{for every }\mathbf u\in\mathbb R^p.
$$

Convexity of $g$ and $h$ is sufficient for this inequality. Also assume, for a constant $C'>0$ independent of $t$, $\alpha$, and $\beta$, that

$$
\|G(\mathbf x^{(t)})+\beta H(\mathbf x^{(t)})\|_2^2
\leq C'\beta^2.
$$

Then, for any $\eta>0$,

$$
t_2-t_1
\geq
\frac{\operatorname{dist}^2(\mathbf x^{(t_1)},\Theta_f)}
{\alpha\beta\eta}
$$

is sufficient to guarantee

$$
\min_{t_1\leq t<t_2}
\bigl(f(\mathbf x^{(t)})-f^*\bigr)
\leq
\frac{\beta}{2}\bigl(\eta+C'\alpha\beta\bigr).
$$

If $\Theta_f\cap\Theta_g\neq\varnothing$, the same interval contains an iterate satisfying

$$
h(\mathbf x^{(t)})-h_g^*
\leq
\frac12\bigl(\eta+C'\alpha\beta\bigr).
$$

The bound is on the best iterate in the interval. It gives a sufficient observation horizon, not an exact first-crossing time.

## Proof of the two phases

We will first prove the fast-phase contraction under stronger global assumptions to keep the calculation readable. The paper replaces them with the local $r$-CL condition above and proves that the iterates remain in the relevant ball. The smoothness inequality used below is proved in [Smoothness, Descent, and Cocoercivity](/blog/smoothness-descent-cocoercivity/).

Assume that $L,\mu>0$, that $g$ is $L$-smooth, and that it satisfies the $\mu$-Polyak–Łojasiewicz inequality

$$
\|G(\mathbf x)\|_2^2
\geq
2\mu\bigl(g(\mathbf x)-g^*\bigr).
$$

Suppose that, at step $t$,

$$
\beta\|H(\mathbf x^{(t)})\|_2
\leq
\gamma\|G(\mathbf x^{(t)})\|_2
$$

for some fixed $0\leq\gamma<1$. To shorten the calculation, write $G=G(\mathbf x^{(t)})$ and $H=H(\mathbf x^{(t)})$. Smoothness and the update give

$$
\begin{aligned}
g(\mathbf x^{(t+1)})
&\leq
g(\mathbf x^{(t)})
-\alpha\langle G,G+\beta H\rangle
+\frac{L\alpha^2}{2}\|G+\beta H\|_2^2\\
&\leq
g(\mathbf x^{(t)})
-\alpha\left[
1-\gamma-\frac{L\alpha}{2}(1+\gamma)^2
\right]\|G\|_2^2.
\end{aligned}
$$

Choose $\alpha$ so that

$$
c:=1-\gamma-\frac{L\alpha}{2}(1+\gamma)^2>0,
\qquad
\delta:=2\mu\alpha c\in(0,1].
$$

The PL inequality then yields

$$
g(\mathbf x^{(t+1)})-g^*
\leq
(1-\delta)\bigl(g(\mathbf x^{(t)})-g^*\bigr).
$$

If the same $\gamma$-dominance condition holds at every step $t=0,\ldots,k-1$, iterating the inequality proves

$$
g(\mathbf x^{(k)})-g^*
\leq
(1-\delta)^k\bigl(g(\mathbf x^{(0)})-g^*\bigr).
$$

Thus the training loss decreases geometrically during the first phase. Moreover, nonnegative $L$-smooth losses satisfy $\|G(\mathbf x)\|_2^2\leq2L g(\mathbf x)$, so the training-loss gradient becomes small as well. The local CL proof in the paper shows that, for sufficiently small $\beta$, the dominance condition lasts until the loss reaches the $\Omega(\beta^C)$ scale. This is the technical step that turns the calculation above into the first part of Theorem 2.1.

The late phase follows from a distance argument. For $\mathbf x^*\in\Theta_f$, the inequality assumed above and the update imply

$$
\begin{aligned}
\|\mathbf x^{(t+1)}-\mathbf x^*\|_2^2
&=
\|\mathbf x^{(t)}-\mathbf x^*\|_2^2
-2\alpha\langle F(\mathbf x^{(t)}),\mathbf x^{(t)}-\mathbf x^*\rangle
+\alpha^2\|F(\mathbf x^{(t)})\|_2^2\\
&\leq
\|\mathbf x^{(t)}-\mathbf x^*\|_2^2
-2\alpha\bigl(f(\mathbf x^{(t)})-f^*\bigr)
+\alpha^2\|F(\mathbf x^{(t)})\|_2^2.
\end{aligned}
$$

Summing from $t_1$ to $t_2-1$, dropping the final nonnegative distance, and minimizing over $\mathbf x^*\in\Theta_f$ gives

$$
\min_{t_1\leq t<t_2}
\bigl(f(\mathbf x^{(t)})-f^*\bigr)
\leq
\frac{\operatorname{dist}^2(\mathbf x^{(t_1)},\Theta_f)}
{2\alpha(t_2-t_1)}
+
\frac{\alpha}{2}
\max_{t_1\leq t<t_2}\|F(\mathbf x^{(t)})\|_2^2.
$$

Using $\|F(\mathbf x^{(t)})\|_2^2\leq C'\beta^2$ and the stated lower bound on $t_2-t_1$ proves

$$
\min_{t_1\leq t<t_2}
\bigl(f(\mathbf x^{(t)})-f^*\bigr)
\leq
\frac{\beta}{2}\bigl(\eta+C'\alpha\beta\bigr).
$$

Finally, if $\Theta_f\cap\Theta_g\neq\varnothing$, then $f^*=\beta h_g^*$ and, since $g\geq0$,

$$
\beta\bigl(h(\mathbf x^{(t)})-h_g^*\bigr)
\leq
f(\mathbf x^{(t)})-f^*.
$$

Dividing the preceding objective bound by $\beta$ gives the property bound. This proves the late-phase result and exhibits the $1/(\alpha\beta)$ factor in its sufficient horizon.

## Sparse recovery: the ℓ₁ bias

Let $\mathbf a^*\in\mathbb R^n$ be a sparse vector and suppose that we observe

$$
\mathbf y^*=\mathbf X\mathbf a^*+\boldsymbol\xi,
\qquad
\mathbf X\in\mathbb R^{N\times n},
$$

where $\mathbf y^*,\boldsymbol\xi\in\mathbb R^N$ and $\boldsymbol\xi$ is measurement noise. We optimize the coefficients $\mathbf a\in\mathbb R^n$ using

$$
f(\mathbf a)
=
\frac12\|\mathbf X\mathbf a-\mathbf y^*\|_2^2
+\beta\|\mathbf a\|_1.
$$

With the near-zero initialization used in our experiment, the early, data-fit-dominated phase moves toward the minimum-Euclidean-norm least-squares solution

$$
\widehat{\mathbf a}
:=
(\mathbf X^\top\mathbf X)^\dagger\mathbf X^\top\mathbf y^*.
$$

This solution minimizes the measurement residual and, when $\mathbf y^*$ lies in the range of $\mathbf X$, fits the measurements. In an underdetermined problem it need not equal the sparse target. After memorization, the $\ell_1$ subgradient dominates and pushes the iterates toward a sparse fitting solution.

When $\mathbf X$ contains enough information for $\mathbf a^*$ to be the stable minimum-$\ell_1$ fit, the recovery theorems give a best-iterate $\ell_1$ recovery error of order $\eta+\alpha\beta+\|\boldsymbol\xi\|_2$ once the late-phase horizon is of order

$$
\frac{\|\mathbf a^{(t_1)}-\mathbf a^*\|_2^2}
{\alpha\beta\eta}.
$$

In the noiseless scaling experiment, let $t_{\mathrm{plat}}$ denote the selected plateau checkpoint. We observe

$$
\|\mathbf a^{(t_{\mathrm{plat}})}-\mathbf a^*\|_1
\propto\alpha\beta,
\qquad
t_{\mathrm{plat}}
\propto
\frac{\|\widehat{\mathbf a}-\mathbf a^*\|_\infty}{\alpha\beta}.
$$

This is grokking in a recovery problem. A small residual $\|\mathbf X\mathbf a^{(t)}-\mathbf y^*\|_2$ marks memorization; a small recovery error $\|\mathbf a^{(t)}-\mathbf a^*\|_2$ marks generalization.

<figure class="article-figure">
  <img src="/images/blog/grokking/sparse-recovery-dynamics.png" width="1501" height="2329" alt="Sparse-recovery errors, gradient ratio, L1 norm, and coefficient trajectories across memorization and generalization" loading="lazy" />
  <figcaption>The loss gradient dominates before memorization. Afterwards, the ℓ₁ subgradient controls the slow motion toward the sparse target.</figcaption>
</figure>

<figure class="article-figure">
  <img src="/images/blog/grokking/sparse-recovery-scaling.png" width="1545" height="1114" alt="Sparse-recovery time and error as functions of alpha beta" loading="lazy" />
  <figcaption>The selected plateau time scales as the distance from the least-squares fit to the target, divided by αβ, while the error at that checkpoint scales as αβ. The plateau is the first pair of recorded errors lying within 5% of the mean of the final three; runs without such a pair are shown at their final checkpoint.</figcaption>
</figure>

Small $\alpha\beta$ therefore lowers the observed recovery error, but makes the generalization phase longer. This is the same tradeoff predicted by the generic theorem.

## Low-rank recovery: the nuclear norm

The matrix analogue replaces sparsity by low rank. Let $\mathbf A^*\in\mathbb R^{n_1\times n_2}$ have rank much smaller than $\min(n_1,n_2)$, and observe

$$
\mathbf y^*=\mathbf X\operatorname{vec}(\mathbf A^*)+\boldsymbol\xi,
\qquad
\mathbf y^*,\boldsymbol\xi\in\mathbb R^N.
$$

Here $\operatorname{vec}(\mathbf A)$ stacks the columns of $\mathbf A$ into a vector and $\mathbf X\in\mathbb R^{N\times n_1n_2}$ is the measurement matrix. We optimize the matrix $\mathbf A$ through

$$
f(\mathbf A)
=
\frac12\|\mathbf X\operatorname{vec}(\mathbf A)-\mathbf y^*\|_2^2
+\beta\|\mathbf A\|_*,
$$

where $\|\mathbf A\|_*=\sum_i\sigma_i(\mathbf A)$ is the nuclear norm. If

$$
\operatorname{vec}(\widehat{\mathbf A})
:=
(\mathbf X^\top\mathbf X)^\dagger\mathbf X^\top\mathbf y^*,
$$

then, from the near-zero initialization used in our experiment, the early phase approaches the least-squares fit $\widehat{\mathbf A}$, while the late nuclear-norm dynamics favor a low-rank solution. Under the recovery conditions in the paper, a best iterate has nuclear-norm recovery error of order $\eta+\alpha\beta+\|\boldsymbol\xi\|_2$ once the late-phase horizon is of order

$$
\frac{\|\mathbf A^{(t_1)}-\mathbf A^*\|_F^2}
{\alpha\beta\eta}.
$$

In the noiseless scaling experiment, again let $t_{\mathrm{plat}}$ denote the selected plateau checkpoint. We observe

$$
\|\mathbf A^{(t_{\mathrm{plat}})}-\mathbf A^*\|_*
\propto\alpha\beta,
\qquad
t_{\mathrm{plat}}
\propto
\frac{\|\widehat{\mathbf A}-\mathbf A^*\|_{2\to2}}{\alpha\beta},
$$

where $\|\cdot\|_{2\to2}$ is the spectral norm. If $\mathbf A=\mathbf U\boldsymbol\Sigma\mathbf V^\top$ is a thin singular-value decomposition, the experiment selects the canonical nuclear-norm subgradient $H(\mathbf A)=\mathbf U\mathbf V^\top$. A pure regularization step decreases each positive singular value by $\alpha\beta$ until discretization causes an $\mathcal O(\alpha\beta)$ oscillation; the small loss gradient perturbs this picture. The formal argument is given in the paper’s appendix.

<figure class="article-figure">
  <div class="figure-image-overlay">
    <img src="/images/blog/grokking/low-rank-recovery-scaling.png" width="1528" height="1115" alt="Low-rank matrix-recovery time and nuclear-norm error as functions of alpha beta" loading="lazy" />
    <svg viewBox="0 0 1528 1115" preserveAspectRatio="none" aria-hidden="true">
      <rect x="946" y="788" width="46" height="70" fill="#fff" />
      <text x="969" y="809" text-anchor="middle" class="legend-fraction-text">1</text>
      <line x1="948" y1="817" x2="990" y2="817" class="legend-fraction-line" />
      <text x="969" y="847" text-anchor="middle" class="legend-fraction-text">√n</text>
    </svg>
  </div>
  <figcaption>Low-rank matrix completion shows the same scaling: the late-phase time is proportional to the spectral distance from the least-squares fit to the target, divided by αβ, and the nuclear-norm recovery error is proportional to αβ. The orange recovery error is normalized by √n, where n = n₁n₂, correcting the archived figure legend. Plateau checkpoints use the same empirical rule as in the sparse experiment.</figcaption>
</figure>

Sparse and low-rank recovery make the point especially clean: generalization is controlled by the property that identifies the hidden object, not by a universal parameter norm.

## Grokking without understanding

A late transition can still occur when the active bias is misaligned with the target. Return to sparse recovery, but replace $\ell_1$ regularization with weight decay:

$$
f(\mathbf a)
=
\frac12\|\mathbf X\mathbf a-\mathbf y^*\|_2^2
+\frac\beta2\|\mathbf a\|_2^2.
$$

For

$$
0<\alpha<
\frac{2}{\sigma_{\max}(\mathbf X^\top\mathbf X+\beta\mathbf I_n)},
$$

gradient descent converges to the ridge solution

$$
\widehat{\mathbf a}_\beta
=
(\mathbf X^\top\mathbf X+\beta\mathbf I_n)^{-1}
\mathbf X^\top\mathbf y^*.
$$

As $\beta\to0$, this approaches the minimum-Euclidean-norm least-squares solution. If the measurements are insufficient to identify $\mathbf a^*$, that solution generally differs from the sparse target. With a large initialization, weight decay can nevertheless produce an abrupt late drop in a proxy error as the parameters move toward $\widehat{\mathbf a}_\beta$.

We call this **grokking without understanding**. A sharp transition in training loss, parameter norm, or another proxy does not establish recovery of the intended rule. The generalization observable must measure what we actually want the model to learn.

## The bias need not be an explicit norm

The same mechanism extends beyond an explicit penalty in the objective.

**Depth.** In sparse recovery, let $D\geq2$ and parameterize the effective coefficient vector with $\mathbf u_1,\ldots,\mathbf u_D\in\mathbb R^n$ as

$$
\mathbf a=\mathbf u_1\odot\cdots\odot\mathbf u_D,
$$

where $\odot$ denotes the coordinatewise product. The predictions remain $\mathbf X\mathbf a$, but gradient descent acts on the factors $\mathbf u_1,\ldots,\mathbf u_D$. With small initialization, this parameterization can create an implicit bias toward sparsity and recover the target without an explicit $\ell_1$ term.

**Data selection.** In matrix completion, sampling entries with high leverage scores can lower the sample requirement and shorten the recovery time. In compressed sensing, choosing measurements incoherent with the sparsifying basis has the same role. The data determine whether the desired low-complexity solution is identifiable and how quickly it can be reached.

**Domain-specific regularization.** For a student $\mathbf F_\theta$ learning from a teacher $\mathbf F_*$ on inputs $\mathbf x_1,\ldots,\mathbf x_N$, a Sobolev penalty such as

$$
h(\theta)
=
\frac1N\sum_{i=1}^N
\|\nabla_{\mathbf x}\mathbf F_\theta(\mathbf x_i)
-\nabla_{\mathbf x}\mathbf F_*(\mathbf x_i)\|_F^2
$$

favors agreement of input derivatives; $\nabla_{\mathbf x}\mathbf F$ denotes the input Jacobian. Here the relevant property is Jacobian agreement, rather than the size of the parameter vector.

The common structure is simple: the fast dynamics fit the observations; the slow bias selects one fitting solution. Grokking occurs when that selection process eventually favors a solution that generalizes.

## References

- Pascal Jr. Tikeng Notsawo, Guillaume Dumas, and Guillaume Rabusseau, [“Grokking Beyond the Euclidean Norm of Model Parameters”](https://arxiv.org/abs/2506.05718), ICML 2025.
- Ziming Liu, Eric J. Michaud, and Max Tegmark, [“Omnigrok: Grokking Beyond Algorithmic Data”](https://openreview.net/forum?id=zDiHoIWa0q1), ICLR 2023.
- Kaifeng Lyu et al., [“Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking”](https://arxiv.org/abs/2311.18817), 2023.
- Tanishq Kumar et al., [“Grokking as the Transition from Lazy to Rich Training Dynamics”](https://arxiv.org/abs/2310.06110), 2023.
