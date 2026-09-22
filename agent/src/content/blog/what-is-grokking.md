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

<p class="article-lede">Grokking refers to delayed generalization following overfitting: a model first <span class="grokking-mark grokking-mark--memorization">memorizes</span> its training data and only much later learns a rule that <span class="grokking-mark grokking-mark--comprehension">generalizes</span>.</p>

The term was introduced by [Power et al. (2022)](#ref-power2022grokking) for neural networks trained on small algorithmic datasets.
Our goal in this blog post is to provide a formal definition of grokking in line with the current state of the art on the subject.
Let us begin with the standard example of addition modulo a prime, then make the definition precise.

## Modular Addition

Let $p$ be a prime integer and <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="integers-mod-p" aria-label="Explain the integers modulo p">$\mathcal S:=\mathbb Z/p\mathbb Z$</button><span id="integers-mod-p" class="explanation-popover" popover="auto" role="note" aria-label="Integers modulo p" data-label="Notation">$\mathbb Z/p\mathbb Z$ is the set of residue classes modulo $p$. We represent them by $\{0,\ldots,p-1\}$, with addition wrapping around after $p-1$.</span></span>. For $x\in\mathcal S$, the notation $\langle x\rangle\in\{0,\ldots,p-1\}$ denotes its integer representative and therefore its token index. For $\mathbf x=(x_1,x_2)\in\mathcal S^2$, define the label by $y(\mathbf x)
:= \left\langle (x_1+x_2)\bmod p\right\rangle$ and the dataset by
$$
\begin{equation}
\begin{aligned}
%y(\mathbf x) &:= \left\langle (x_1+x_2)\bmod p\right\rangle,\\
\mathcal D &:= \left\{ \bigl(\mathbf x,y(\mathbf x)\bigr) : \mathbf x\in\mathcal S^2 \right\}.
\end{aligned}
\end{equation}
$$

We randomly split $\mathcal D$ into a training set $\mathcal D_{\mathrm{train}}$ and a test set $\mathcal D_{\mathrm{test}}$ according to a ratio $r := |\mathcal{D}_{\text{train}}| / |\mathcal{D}| \in (0, 1)$.

Let $\mathbf{E} \in \mathbb{R}^{p \times d_1}$ be the trainable embedding table for all the symbols in $\mathcal{S}$, and let $\mathbf e(x) := \mathbf E_{\langle x\rangle,:} \in \mathbb{R}^{d_1}$ be the embedding of token $x \in \mathcal{S}$.
For $\mathbf x=(x_1,x_2)\in\mathcal S^2$, the embeddings of $x_1$ and $x_2$ are added before being sent through the one-hidden-layer ReLU network.
More specifically, the model is
$$
\begin{equation}
\begin{aligned}
\mathbf z_\theta(\mathbf x)
&:=
\mathbf W^{(2)}
\operatorname{ReLU}\!\left(
\mathbf W^{(1)}[\mathbf e(x_1)+\mathbf e(x_2)]+\mathbf b^{(1)}
\right)
+\mathbf b^{(2)}
\end{aligned}
\end{equation}
$$
where the matrices $\mathbf W^{(1)}\in\mathbb R^{d_2\times d_1}$ and $\mathbf W^{(2)}\in\mathbb R^{p\times d_2}$ are the hidden and output weights, while $\mathbf b^{(1)}\in\mathbb R^{d_2}$ and $\mathbf b^{(2)}\in\mathbb R^p$ are their biases. Thus $d_1$ is the embedding dimension, $d_2$ is the hidden width, and $\mathbf z_\theta(\mathbf x)\in\mathbb R^p$ contains the class <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="class-logits" aria-label="Define class logits">logits</button><span id="class-logits" class="explanation-popover" popover="auto" role="note" aria-label="Class logits" data-label="Definition">Logits are the model's unnormalized class scores. Applying softmax turns them into probabilities; taking $\operatorname{argmax}$ selects the largest score directly.</span></span>.
Every array collected in $\theta := \{ \mathbf E,\mathbf W^{(1)},\mathbf b^{(1)},\mathbf W^{(2)},\mathbf b^{(2)} \}$ is optimized.

We train this model to minimize $f(\theta):=g(\theta)+\beta\|\theta\|_2^2$, where $g(\theta)$ is the average <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="cross-entropy" aria-label="Explain cross-entropy loss">cross-entropy loss</button><span id="cross-entropy" class="explanation-popover" popover="auto" role="note" aria-label="Cross-entropy loss" data-label="Definition">For a true class $i$, cross-entropy is the negative log of the softmax probability assigned to $i$. It strongly penalizes confident predictions of the wrong class.</span></span> on $\mathcal{D}_{\text{train}}$:

$$
\begin{equation}
g(\theta) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text{train}}} \ell \left( \mathbf{z}_{\theta}(\mathbf{x}), y \right)
\end{equation}
$$
$$
\begin{equation}
\ell \left( \mathbf{z}, i \right) := - \log \frac{ \exp\left( \mathbf{z}_i \right) }{ \sum_{j} \exp\left( \mathbf{z}_j \right)}, \qquad \forall \mathbf{z} \in \mathbb{R}^p, \ i \in [p]
\end{equation}
$$

For the following figure, we use $(p,r)=(97,0.4)$ and optimize $f(\theta)$ with <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="adam-optimizer" aria-label="Explain the Adam optimizer">Adam</button><span id="adam-optimizer" class="explanation-popover" popover="auto" role="note" aria-label="Adam optimizer" data-label="Optimizer">Adam maintains moving averages of each parameter's gradient and squared gradient, then uses them to adapt the update scale coordinate by coordinate.</span></span>, using a learning rate $\alpha=10^{-3}$ and penalty strength $\beta=10^{-6}$. Source code: [Tikquuss/grokking_algorithmic](https://github.com/Tikquuss/grokking_algorithmic).


<figure class="article-figure">
  <img src="/images/blog/grokking/modular-addition-training.png" width="1049" height="695" alt="Training and test loss and accuracy during grokking on addition modulo 97" loading="lazy" />
  <figcaption>One run on addition modulo 97. The model reaches perfect training accuracy around <span class="grokking-mark grokking-mark--memorization">t₁ ≈ 470</span>, but test accuracy reaches the same level only around <span class="grokking-mark grokking-mark--comprehension">t₂ ≈ 9000</span>.</figcaption>
</figure>

The first transition is ordinary memorization. The striking part is the long interval during which training accuracy is perfect while test accuracy remains poor, followed by generalization without any new data.

## A formal definition

Let $\theta^{(t)}$ denote the parameters after $t\in\mathbb N$ optimization steps.
The model optimization process can be divided into three consecutive phases when studying delayed generalization (see the figure above).
The initial learning phase, $t \in [0,t_{1/2}]$, is called the <span class="grokking-mark grokking-mark--confusion">confusion phase</span>, during which both training and validation performance are poor. During the <span class="grokking-mark grokking-mark--memorization">memorization phase</span>, $t \in [t_1,t_{3/2}]$, training performance is nearly perfect while validation performance remains low. During the <span class="grokking-mark grokking-mark--comprehension">comprehension phase</span>, $t \in [t_{3/2},\infty)$, validation performance improves and can eventually match training performance at step $t_2$.
For a classification task such as the example above, $t_{1/2}$ can be taken as the first step at which the training accuracy $\mathcal{A}_{\text{train}}$ becomes strictly greater than $0\%$; $t_1$ as the first step at which $\mathcal{A}_{\text{train}}$ reaches $\approx1$; $t_{3/2}$ as the first step at which the test accuracy $\mathcal{A}_{\text{test}}$ becomes strictly greater than $0\%$; and $t_2$ as the first step at which $\mathcal{A}_{\text{test}}$ reaches $\approx1$.

The training of a deep-learning model by gradient descent generally, but not necessarily, goes through these phases. In the literature, training that ends in the confusion, memorization, or comprehension phase is referred to as *underfitting*, *overfitting*, or *generalization*, respectively.


More formally, define:
$$
\begin{equation}
\mathcal A_{\mathcal B}(t)
:=
\frac{1}{|\mathcal B|}
\sum_{(\mathbf x,y)\in\mathcal B}
\mathbf 1\!\left\{
\operatorname*{argmax}_{j\in\{0,\ldots,p-1\}}
[\mathbf z_{\theta^{(t)}}(\mathbf x)]_j=y
\right\} \in [0, 1]
\end{equation}
$$
for $\mathcal B\in\{\mathcal D_{\mathrm{train}},\mathcal D_{\mathrm{test}}\}$. This is the accuracy after $t$ optimization steps. We write the two instances of Equation (3) as $\mathcal A_{\mathrm{train}}$ and $\mathcal A_{\mathrm{test}}$.

Fix a near-perfect accuracy threshold $q\in(1/p,1]$. The memorization and generalization times are
$$
\begin{equation}
t_1:=\inf\{t\in\mathbb N_0:\mathcal A_{\mathrm{train}}(t)\geq q\},
\qquad
t_2:=\inf\{t\in\mathbb N_0:\mathcal A_{\mathrm{test}}(t)\geq q\},
\end{equation}
$$

where $\inf\varnothing=\infty$. When both times are finite, the generalization delay is

$$
\begin{equation}
\Delta t_{\mathrm{gen}}:=t_2-t_1.
\end{equation}
$$

> <span class="grokking-kicker grokking-kicker--definition">Definition</span> Choose $q$ and a minimum relative delay $\kappa>0$ before looking at the trajectory. A run exhibits grokking when
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
> \end{equation}
> $$

The last condition distinguishes delayed generalization after memorization from two unrelated threshold crossings. The constant $\kappa$ gives a precise meaning to a “long” delay; suddenness is common in grokking curves, but it is not part of the definition.

> <span class="grokking-kicker grokking-kicker--remark">Remark</span> For the sake of simplicity, many works studying grokking characterize all training experiments that eventually generalize as grokking, thereby avoiding a debate about what could be considered a trivial number of steps.

> <span class="grokking-kicker grokking-kicker--remark">Remark</span> The definition above is not fully general because it assumes an ideal context in which training progresses “smoothly.” It ignores <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="progressive-sharpening" aria-label="Explain progressive sharpening">progressive sharpening</button><span id="progressive-sharpening" class="explanation-popover" popover="auto" role="note" aria-label="Progressive sharpening" data-label="Dynamics">The largest Hessian eigenvalue $\lambda_{\max}(\mathcal H^{(t)})$ increases during training, so the local loss landscape becomes progressively sharper. Cohen et al. (2021) observed it approaching the gradient-descent stability scale $2/\alpha_t$.</span></span>, studied by [Cohen et al. (2021)](#ref-cohen2021gradient), and the resulting <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="edge-of-stability" aria-label="Explain the edge-of-stability regime">edge-of-stability regime</button><span id="edge-of-stability" class="explanation-popover" popover="auto" role="note" aria-label="Edge of stability" data-label="Dynamics">When curvature is near or beyond the nominal stability threshold $2/\alpha_t$, individual gradient-descent steps need not decrease the loss, even though the longer-run trajectory can continue to make progress.</span></span>. It also ignores the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="slingshot-mechanism" aria-label="Explain the slingshot mechanism">slingshot mechanism</button><span id="slingshot-mechanism" class="explanation-popover" popover="auto" role="note" aria-label="Slingshot mechanism" data-label="Dynamics">The slingshot mechanism consists of cyclic transitions between relatively stable and unstable training regimes, often accompanied by spikes in loss or parameter norm.</span></span> studied by [Thilak et al. (2022)](#ref-thilak2022slingshot). The phases must therefore be adapted to the context. [Lyu et al. (2023)](#ref-lyu2023dichotomy) even observed <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="misgrokking" aria-label="Explain misgrokking">misgrokking</button><span id="misgrokking" class="explanation-popover" popover="auto" role="note" aria-label="Misgrokking" data-label="Phenomenon">In misgrokking, the model first generalizes and then, after a long training period, moves toward memorization and worse generalization—the temporal order is reversed.</span></span>.


## From a run to a random regime

For a fixed choice $H$ of hyperparameters (e.g., learning rate, weight decay, and mini-batch size), the training split, initialization, minibatch order, and any stochastic optimizer choices make the trajectory random. Let $\omega$ denote their joint outcome under a fixed experimental protocol, and write $t_1(\omega)$, $t_2(\omega)$, $\Delta t_{\mathrm{gen}}(\omega):=t_2(\omega)-t_1(\omega)$, and $\mathcal A_{\mathrm{train}}^\omega$ for the corresponding quantities. The grokking event is

$$
\begin{equation}
\mathsf G_{q,\kappa}(H)
:=
\left\{
\omega:
\begin{array}{l}
0<t_1(\omega)<t_2(\omega)<\infty,\\[2pt]
\Delta t_{\mathrm{gen}}(\omega)/t_1(\omega)\geq\kappa,\\[2pt]
\mathcal A_{\mathrm{train}}^\omega(t)\geq q
\text{ for every }t_1(\omega)\leq t\leq t_2(\omega)
\end{array}
\right\}.
\end{equation}
$$

Grokking corresponds to $\mathbb P_{\mathrm{gen}}\!\left[\mathsf G_{q,\kappa}(H)\right]=1$, where $\mathbb P_{\mathrm{gen}}$ is the probability law of the random experiment. This is a statement about the full grokking event, not merely eventual test success. A finite collection of seeds can estimate its probability but cannot prove that it is one; a run stopped before $t_2$ is observed is <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="right-censoring" aria-label="Explain censoring">right-censored</button><span id="right-censoring" class="explanation-popover" popover="auto" role="note" aria-label="Right-censored training run" data-label="Statistics">We know only that the unobserved generalization time exceeds the stopping time. Treating such a run as if $t_2=\infty$ would confuse missing future observation with evidence of non-generalization.</span></span>, rather than evidence that $t_2=\infty$.

Although it is easy to identify grokking, it is very difficult to give a formal definition of its opposite since, in practice, we cannot optimize a model for an infinite number of steps. In many cases, we do not even have access to the true data distribution $\mathbb P_{\mathrm{data}}$ needed to properly define $\mathbb P_{\mathrm{gen}}$. Even if this distribution is available, exactly or approximately, computing $\mathbb P_{\mathrm{gen}}\!\left[\mathsf G_{q,\kappa}(H)\right]$ for a fixed choice of hyperparameters remains intractable given the complexity of the stochastic process defined by the optimization procedure, a complexity inherited in part from that of the model whose parameters are being optimized.

Faced with this challenge, we proceeded empirically in [Notsawo et al. (2023)](#ref-notsawo2023predicting). For models of the same family—that is, with the same architecture—and a given quantity of training data $r$, we train several models with different hyperparameters and initializations. When all possible input-output pairs form a finite, tractable dataset, as in addition modulo a small prime integer, $r\in[0,1]$ can instead denote the training-data fraction. We fit a function that predicts $t_2$, the generalization step, for each $r$. Then, if for a given choice of hyperparameters, initialization, and $r$, we train a model for more than the predicted $t_2(r)$ steps without generalization, we can stop training and report the outcome as confusion or memorization—that is, non-grokking—according to the observed training and validation performance. This corresponds to an empirical definition in which we use the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="empirical-vs-population-law" aria-label="Compare the empirical measure and population law">empirical measure $\widehat{\mathbb P}_{\mathrm{gen}}$</button><span id="empirical-vs-population-law" class="explanation-popover" popover="auto" role="note" aria-label="Empirical measure and population law" data-label="Statistics">$\mathbb P_{\mathrm{gen}}$ is the ideal distribution over all runs allowed by the protocol. $\widehat{\mathbb P}_{\mathrm{gen}}$ places equal mass on the finite runs actually observed and therefore only estimates that population law.</span></span> instead of $\mathbb P_{\mathrm{gen}}$.

In general, more data leads to faster grokking: $t_2(r)$ is a decreasing function of $r$, as reported by [Power et al. (2022)](#ref-power2022grokking), [Liu et al. (2023)](#ref-liu2023grokking), [Žunkovič and Ilievski (2022)](#ref-zunkovic2022grokking), and [Gromov (2023)](#ref-gromov2023grokking). Some authors report a <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="grokking-power-law" aria-label="Explain the power-law notation">power law</button><span id="grokking-power-law" class="explanation-popover" popover="auto" role="note" aria-label="Power law for grokking time" data-label="Scaling law">$t_2(r)=\Theta(r^{-\gamma})$ means that, up to constant factors, the generalization time scales like $r^{-\gamma}$. On logarithmic axes, this relationship appears approximately linear with slope $-\gamma$.</span></span> ([Žunkovič and Ilievski, 2022](#ref-zunkovic2022grokking); [Notsawo et al., 2023](#ref-notsawo2023predicting)):

$$
t_2(r)=\Theta\!\left(r^{-\gamma}\right),\qquad \gamma>0.
$$

This law generally breaks at a <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="critical-training-fraction" aria-label="Explain the critical training fraction">critical value $r_c>0$</button><span id="critical-training-fraction" class="explanation-popover" popover="auto" role="note" aria-label="Critical training fraction" data-label="Threshold">Below $r_c$, the available data do not identify the target well enough for the scaling law to continue. In inverse problems, this can coincide with an information-theoretic recovery threshold.</span></span>: it is valid only for $r\geq r_c$. In the toy model of [Liu et al. (2023)](#ref-liu2023grokking), $r_c$ can be estimated using the quality of the representations learned by the model. For sparse recovery and matrix factorization problems, for which we recently proved the existence of grokking in [Notsawo et al. (2025)](#ref-notsawo2025), this limit is the minimum number of measurements below which no recovery is possible by any method whatsoever; see, for example, [Rauhut (2010)](#ref-rauhut2010) for sparse recovery and [Candès and Recht (2012)](#ref-candes2012) for matrix completion.

## The three phases

This definition separates a grokking trajectory into three phases:

| Phase | Optimization steps | Behaviour |
| --- | --- | --- |
| <span class="grokking-phase grokking-phase--confusion">Confusion</span> | $t<t_1$ | The model has not yet fit the training data. |
| <span class="grokking-phase grokking-phase--memorization">Memorization</span> | $t_1\leq t<t_2$ | Training accuracy is high, but test accuracy is still low. |
| <span class="grokking-phase grokking-phase--comprehension">Comprehension</span> | at $t_2$ | Training and test accuracy both meet the threshold $q$; in a typical grokking trajectory, they remain high afterwards. |

Taking $q=1$ for the curve above gives $\Delta t_{\mathrm{gen}}\approx8530$: the memorization phase lasts much longer than the initial fitting phase.

[Liu et al. (2023)](#ref-liu2023grokking) used the terms *confusion*, *memorization*, and *comprehension* in a phase diagram based on different hyperparameters. In this post, we also use them to refer to phases along a single training trajectory, as defined above.

It is common in the deep-learning literature to divide neural-network optimization into <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="conventional-two-phases" aria-label="Explain the conventional two-phase picture">two phases</button><span id="conventional-two-phases" class="explanation-popover" popover="auto" role="note" aria-label="Conventional two-phase picture" data-label="Context">The usual picture has an initial fitting phase with a small generalization gap, followed by overfitting as test error rises. Grokking requires a different temporal pattern: a prolonged memorization interval followed by improved generalization.</span></span> ([Shwartz-Ziv and Tishby, 2017](#ref-shwartz-ziv2017); [Nakkiran et al., 2020](#ref-nakkiran2020); [Feng and Tu, 2021](#ref-feng-tu2021)). However, [Nakkiran et al. (2020)](#ref-nakkiran2020) show that in some regimes the test error decreases again and can reach a lower value at the end of training than at the first minimum, suggesting potential training phases to exploit. [Feng and Tu (2021)](#ref-feng-tu2021) distinguish an initial fast-learning phase, in which the loss decreases quickly and sometimes abruptly, followed by an exploration phase, in which the training error has reached its minimum and the overall loss continues to decrease, but much more slowly and gradually. The defining ingredient of grokking lies in the memorization phase and in the transition from memorization to generalization.

> <span class="grokking-kicker grokking-kicker--remark">Remark</span> The definition of grokking evolved between 2024 and 2025. Initially, grokking, as observed by [Power et al. (2022)](#ref-power2022grokking), corresponded to a sudden transition from a long phase of perfect memorization to generalization. Over time, however, the term has evolved to the point where, when a model generalizes late—whether abruptly or gradually—some authors refer to it as *grokking*. This is the case in [Wang et al. (2024)](#ref-wang2024) and [Abramov et al. (2025)](#ref-abramov2025), who show that grokking enables Transformers to develop reasoning abilities that emerge only after extended training, whether on **synthetic comparison or composition tasks** or on real-world multi-hop reasoning **augmented with inferred facts**.

In general, this type of “reasoning” task exhibits a typical progression during training: training and test performance improve similarly from the beginning of training until the model reaches a more or less acceptable level of generalization. Then, when the model is trained for longer, a gradual increase in generalization performance is observed. We will return to this type of grokking in a future post. This seems to be the best kind of grokking we can have on “real,” non-synthetic tasks.

## Accuracy is not always the right observable

Accuracy is natural for modular arithmetic, but it is only one possible observable. A smooth change in the logits can look abrupt after taking an $\operatorname{argmax}$, so loss curves should be shown as well. In regression, signal recovery, or matrix completion, the same definition can use predeclared error tolerances instead of accuracy thresholds.

In contexts where accuracy is not directly available, such as regression, one can use the value $\mathcal L$ of the loss function directly, so that “$\mathcal A\approx1.0$” becomes “$\mathcal L\approx0$.” Alternatively, following [Liu et al. (2023)](#ref-liu2023omnigrok), one can define accuracy as the empirical fraction of points whose prediction error is smaller than a chosen tolerance $\epsilon\geq0$:

$$
\mathcal A(\mathrm{data})
:=
\widehat{\mathbb E}_{z\sim\mathrm{data}}
\mathbf 1\!\left\{\ell(z)\leq\epsilon\right\}
=
\widehat{\mathbb P}_{z\sim\mathrm{data}}
[\ell(z)\leq\epsilon],
$$

where $\ell(z)$ is the value of the loss function on the sample $z$. Here <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="empirical-indicator-notation" aria-label="Explain the empirical expectation and indicator notation">the hat and indicator</button><span id="empirical-indicator-notation" class="explanation-popover" popover="auto" role="note" aria-label="Empirical expectation and indicator" data-label="Notation">$\widehat{\mathbb E}$ and $\widehat{\mathbb P}$ average over the finite dataset, while $\mathbf 1\{\ell(z)\le\epsilon\}$ equals $1$ when the tolerance is met and $0$ otherwise. Their average is therefore the fraction of successful samples.</span></span> turn a continuous error into an empirical accuracy.

It should be noted that the transition in $\mathcal L(t)$ is generally not as sharp as the transition in $\mathcal A(t)$, and it is possible to observe a transition in $\mathcal A(t)$ without observing one in $\mathcal L(t)$, as shown by [Kumar et al. (2023)](#ref-kumar2023grokking). It is preferable to study grokking using loss rather than accuracy: loss reflects the training dynamics, whereas accuracy is discontinuous and can exhibit apparent transitions without any internal change in the model. As [Kumar et al. (2023)](#ref-kumar2023grokking) point out in the context of grokking, hard-threshold measures of performance such as accuracy can be extremely misleading; continuously optimized measures such as loss should be studied instead ([Schaeffer et al., 2024](#ref-schaeffer2024)).

The observable must match the scientific question. In sparse recovery, <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="residual-vs-recovery" aria-label="Compare measurement residual and recovery error">measurement residual and recovery error answer different questions</button><span id="residual-vs-recovery" class="explanation-popover" popover="auto" role="note" aria-label="Measurement residual and recovery error" data-label="Distinction">A small residual means the estimate matches the observed measurements. A small recovery error means it is close to the unknown latent signal. With insufficient or ambiguous measurements, the former can be tiny while the latter remains large.</span></span>. Confusing the two can produce what we call *grokking without understanding* in the [next post](/blog/grokking-beyond-l2/).

## What to report

A grokking experiment should report:

- the task, split, model, optimizer, and regularization;
- the observable and the chosen values of $q$ and $\kappa$;
- the training and test curves, together with $t_1$, $t_2$, and $\Delta t_{\mathrm{gen}}$ across seeds.

With these choices fixed, grokking is no longer just the shape of one attractive curve. It is a measurable event: rapid memorization, a non-trivial delay, and eventual generalization.

## References

- <span id="ref-power2022grokking"></span>Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra, [“Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets”](https://arxiv.org/abs/2201.02177), 2022.
- <span id="ref-cohen2021gradient"></span>Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet S. Talwalkar, [“Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability”](https://arxiv.org/abs/2103.00065), ICLR 2021.
- <span id="ref-thilak2022slingshot"></span>Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, and Joshua Susskind, [“The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon”](https://arxiv.org/abs/2206.04817), 2022.
- <span id="ref-lyu2023dichotomy"></span>Kaifeng Lyu, Jikai Jin, Zhiyuan Li, Simon S. Du, Jason D. Lee, and Wei Hu, [“Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking”](https://arxiv.org/abs/2311.18817), 2023.
- <span id="ref-notsawo2023predicting"></span>Pascal Jr. Tikeng Notsawo, Hattie Zhou, Mohammad Pezeshki, Irina Rish, and Guillaume Dumas, [“Predicting Grokking Long Before It Happens: A Look into the Loss Landscape of Models Which Grok”](https://arxiv.org/abs/2306.13253), 2023.
- <span id="ref-liu2023grokking"></span>Ziming Liu, Ziqian Zhong, and Max Tegmark, [“Grokking as Compression: A Nonlinear Complexity Perspective”](https://arxiv.org/abs/2310.05918), 2023.
- <span id="ref-zunkovic2022grokking"></span>Bojan Žunkovič and Enej Ilievski, [“Grokking Phase Transitions in Learning Local Rules with Gradient Descent”](https://arxiv.org/abs/2210.15435), 2022.
- <span id="ref-gromov2023grokking"></span>Andrey Gromov, [“Grokking Modular Arithmetic”](https://arxiv.org/abs/2301.02679), 2023.
- <span id="ref-notsawo2025"></span>Pascal Jr. Tikeng Notsawo, Guillaume Dumas, and Guillaume Rabusseau, [“Grokking Beyond the Euclidean Norm of Model Parameters”](https://arxiv.org/abs/2506.05718), 2025.
- <span id="ref-rauhut2010"></span>Holger Rauhut, [“Compressive Sensing and Structured Random Matrices”](https://doi.org/10.1515/9783110226157.1), in *Theoretical Foundations and Numerical Methods for Sparse Recovery*, 2010.
- <span id="ref-candes2012"></span>Emmanuel Candès and Benjamin Recht, “Exact Matrix Completion via Convex Optimization,” *Communications of the ACM* 55(6):111–119, 2012.
- <span id="ref-shwartz-ziv2017"></span>Ravid Shwartz-Ziv and Naftali Tishby, [“Opening the Black Box of Deep Neural Networks via Information”](https://arxiv.org/abs/1703.00810), 2017.
- <span id="ref-nakkiran2020"></span>Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever, [“Deep Double Descent: Where Bigger Models and More Data Hurt”](https://openreview.net/forum?id=B1g5sA4twr), ICLR 2020.
- <span id="ref-feng-tu2021"></span>Yu Feng and Yuhai Tu, [“The Inverse Variance–Flatness Relation in Stochastic Gradient Descent Is Critical for Finding Flat Minima”](https://doi.org/10.1073/pnas.2015617118), *Proceedings of the National Academy of Sciences* 118(9):e2015617118, 2021.
- <span id="ref-wang2024"></span>Boshi Wang, Xiang Yue, Yu Su, and Huan Sun, [“Grokked Transformers Are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization”](https://arxiv.org/abs/2405.15071), 2024.
- <span id="ref-abramov2025"></span>Roman Abramov, Felix Steinbauer, and Gjergji Kasneci, [“Grokking in the Wild: Data Augmentation for Real-World Multi-Hop Reasoning with Transformers”](https://arxiv.org/abs/2504.20752), 2025.
- <span id="ref-liu2023omnigrok"></span>Ziming Liu, Eric J. Michaud, and Max Tegmark, [“Omnigrok: Grokking Beyond Algorithmic Data”](https://openreview.net/forum?id=zDiHoIWa0q1), ICLR 2023.
- <span id="ref-kumar2023grokking"></span>Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, and Cengiz Pehlevan, [“Grokking as the Transition from Lazy to Rich Training Dynamics”](https://arxiv.org/abs/2310.06110), 2023.
- <span id="ref-schaeffer2024"></span>Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo, “Are Emergent Abilities of Large Language Models a Mirage?” *Advances in Neural Information Processing Systems* 36, 2024.
