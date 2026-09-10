---
title: "What Is Grokking? A Formal Definition"
date: "2025-05-25"
category: "Research Notes"
image: "/images/blog/grokking/modular-addition-training.png"
tags:
  - Grokking
  - Generalization
  - Overfitting
  - Optimization
excerpt: "A formal definition of grokking based on the delay between memorization and generalization."
---

<p class="article-lede">Grokking refers to delayed generalization following overfitting: a model first memorizes its training data and only much later learns a rule that generalizes.</p>

The term was introduced by [Power et al.](https://arxiv.org/abs/2201.02177) for neural networks trained on small algorithmic datasets. Let us begin with the standard example of addition modulo a prime, then make the definition precise.

## Modular Addition

Let $p=97$ and $\mathcal S:=\mathbb Z/p\mathbb Z$. For $x\in\mathcal S$, the notation $\langle x\rangle\in\{0,\ldots,p-1\}$ denotes its integer representative and therefore its token index. For $\mathbf x=(x_1,x_2)\in\mathcal S^2$, define the label and dataset by

$$
\begin{equation}
\begin{aligned}
y(\mathbf x)
&:=
\left\langle (x_1+x_2)\bmod p\right\rangle,\\
\mathcal D
&:=
\left\{
\bigl(\mathbf x,y(\mathbf x)\bigr)
:
\mathbf x\in\mathcal S^2
\right\}.
\end{aligned}
\tag{1}
\end{equation}
$$

We randomly split $\mathcal D$ into a $40\%$ training set $\mathcal D_{\mathrm{train}}$ and a $60\%$ test set $\mathcal D_{\mathrm{test}}$.

The model is

$$
\begin{equation}
\begin{aligned}
\mathbf e(x)
&:=
\mathbf E_{\langle x\rangle,:}^{\top},\\
\mathbf z_\theta(x_1,x_2)
&:=
\mathbf W^{(2)}
\operatorname{ReLU}\!\left(
\mathbf W^{(1)}[\mathbf e(x_1)+\mathbf e(x_2)]+\mathbf b^{(1)}
\right)
+\mathbf b^{(2)},\\
\theta
&:=
(\mathbf E,\mathbf W^{(1)},\mathbf b^{(1)},\mathbf W^{(2)},\mathbf b^{(2)}).
\end{aligned}
\tag{2}
\end{equation}
$$

In Equation (2), $\mathbf E\in\mathbb R^{p\times d_1}$ is the embedding table and $\mathbf e(x)\in\mathbb R^{d_1}$ is the embedding selected by token index $\langle x\rangle$. The matrices $\mathbf W^{(1)}\in\mathbb R^{d_2\times d_1}$ and $\mathbf W^{(2)}\in\mathbb R^{p\times d_2}$ are the hidden and output weights, while $\mathbf b^{(1)}\in\mathbb R^{d_2}$ and $\mathbf b^{(2)}\in\mathbb R^p$ are their biases. Thus $d_1$ is the embedding dimension, $d_2$ is the hidden width, and $\mathbf z_\theta(x_1,x_2)\in\mathbb R^p$ contains the class logits. Every array collected in $\theta$ is optimized.

The two embeddings are added before being sent through the one-hidden-layer ReLU network. We optimize $\theta$ with Adam on the average training cross-entropy, using learning rate $\alpha=10^{-3}$ and the squared Euclidean penalty $\beta\|\theta\|_2^2$ with $\beta=10^{-6}$.

<figure class="article-figure">
  <img src="/images/blog/grokking/modular-addition-training.png" width="1049" height="695" alt="Training and test loss and accuracy during grokking on addition modulo 97" loading="lazy" />
  <figcaption>One run on addition modulo 97. The model reaches perfect training accuracy around t₁ ≈ 470, but test accuracy reaches the same level only around t₂ ≈ 9000.</figcaption>
</figure>

The first transition is ordinary memorization. The striking part is the long interval during which training accuracy is perfect while test accuracy remains poor, followed by generalization without any new data.

## A formal definition

For shorthand, set $\mathbf z_\theta(\mathbf x):=\mathbf z_\theta(x_1,x_2)$ and let $\theta^{(t)}$ denote the parameters after the integer optimization step $t\in\mathbb N_0$.

For $\mathcal B\in\{\mathcal D_{\mathrm{train}},\mathcal D_{\mathrm{test}}\}$, define

$$
\begin{equation}
\mathcal A_{\mathcal B}(t)
:=
\frac{1}{|\mathcal B|}
\sum_{(\mathbf x,y)\in\mathcal B}
\mathbf 1\!\left\{
\operatorname*{argmax}_{j\in\{0,\ldots,p-1\}}
[\mathbf z_{\theta^{(t)}}(\mathbf x)]_j=y
\right\}.
\tag{3}
\end{equation}
$$

This is the accuracy after $t$ optimization steps. We write the two instances of Equation (3) as $\mathcal A_{\mathrm{train}}$ and $\mathcal A_{\mathrm{test}}$.

Fix a near-perfect accuracy threshold $q\in(1/p,1]$. The memorization and generalization times are

$$
\begin{equation}
t_1:=\inf\{t\in\mathbb N_0:\mathcal A_{\mathrm{train}}(t)\geq q\},
\qquad
t_2:=\inf\{t\in\mathbb N_0:\mathcal A_{\mathrm{test}}(t)\geq q\},
\tag{4}
\end{equation}
$$

where $\inf\varnothing=\infty$. When both times are finite, the generalization delay is

$$
\begin{equation}
\Delta t_{\mathrm{gen}}:=t_2-t_1.
\tag{5}
\end{equation}
$$

> **Definition.** Choose $q$ and a minimum relative delay $\kappa>0$ before looking at the trajectory. A run exhibits grokking when
>
> $$
> \begin{equation}
> \begin{gathered}
> 0<t_1<t_2<\infty,
> \qquad
> \frac{\Delta t_{\mathrm{gen}}}{t_1}\geq\kappa,\\
> \mathcal A_{\mathrm{train}}(t)\geq q
> \quad\text{for every }t_1\leq t\leq t_2.
> \end{gathered}
> \tag{6}
> \end{equation}
> $$

The last condition distinguishes delayed generalization after memorization from two unrelated threshold crossings. The constant $\kappa$ gives a precise meaning to a “long” delay; suddenness is common in grokking curves, but it is not part of the definition.

## From a run to a random regime

The training split, initialization, minibatch order, and any stochastic optimizer choices make the trajectory random. Let $\omega$ denote their joint outcome under a fixed experimental protocol, and write $t_1(\omega)$, $t_2(\omega)$, and $\mathcal A_{\mathrm{train}}^\omega$ for the corresponding quantities. The grokking event is

$$
\begin{equation}
\mathsf G_{q,\kappa}
:=
\left\{
\omega:
\begin{array}{l}
0<t_1(\omega)<t_2(\omega)<\infty,\\[2pt]
\big(t_2(\omega)-t_1(\omega)\big)/t_1(\omega)\geq\kappa,\\[2pt]
\mathcal A_{\mathrm{train}}^\omega(t)\geq q
\text{ for every }t_1(\omega)\leq t\leq t_2(\omega)
\end{array}
\right\}.
\tag{7}
\end{equation}
$$

The randomized training regime exhibits grokking almost surely when

$$
\begin{equation}
\mathbb P_{\mathrm{gen}}\!\left(\mathsf G_{q,\kappa}\right)=1,
\tag{8}
\end{equation}
$$

where $\mathbb P_{\mathrm{gen}}$ is the probability law of the random experiment. This is a statement about the full grokking event, not merely eventual test success. A finite collection of seeds can estimate its probability but cannot prove that it is one; a run stopped before $t_2$ is observed is censored rather than evidence that $t_2=\infty$.

## The three phases

This definition separates a grokking trajectory into three phases:

| Phase | Optimization steps | Behaviour |
| --- | --- | --- |
| Confusion | $t<t_1$ | The model has not yet fit the training data. |
| Memorization | $t_1\leq t<t_2$ | Training accuracy is high, but test accuracy is still low. |
| Comprehension | at $t_2$ | Training and test accuracy both meet the threshold $q$; in a typical grokking trajectory, they remain high afterwards. |

Taking $q=1$ for the curve above gives $\Delta t_{\mathrm{gen}}\approx8530$: the memorization phase lasts much longer than the initial fitting phase.

## Accuracy is not always the right observable

Accuracy is natural for modular arithmetic, but it is only one possible observable. A smooth change in the logits can look abrupt after taking an $\operatorname{argmax}$, so loss curves should be shown as well. In regression, signal recovery, or matrix completion, the same definition can use predeclared error tolerances instead of accuracy thresholds.

The observable must match the scientific question. In sparse recovery, a small measurement residual says that the observations have been fit; distance to the latent signal says whether the signal has actually been recovered. Confusing the two can produce what we call *grokking without understanding* in the [next post](/blog/grokking-beyond-l2/).

## What to report

A grokking experiment should report:

- the task, split, model, optimizer, and regularization;
- the observable and the chosen values of $q$ and $\kappa$;
- the training and test curves, together with $t_1$, $t_2$, and $\Delta t_{\mathrm{gen}}$ across seeds.

With these choices fixed, grokking is no longer just the shape of one attractive curve. It is a measurable event: rapid memorization, a non-trivial delay, and eventual generalization.

## References

- Alethea Power et al., [“Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets”](https://arxiv.org/abs/2201.02177), 2022.
- Ziming Liu, Eric J. Michaud, and Max Tegmark, [“Omnigrok: Grokking Beyond Algorithmic Data”](https://openreview.net/forum?id=zDiHoIWa0q1), ICLR 2023.
- Tanishq Kumar et al., [“Grokking as the Transition from Lazy to Rich Training Dynamics”](https://arxiv.org/abs/2310.06110), 2023.
