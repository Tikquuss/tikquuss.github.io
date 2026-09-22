---
title: "Epoch-wise Bias–Variance Decomposition"
date: "2023-05-01"
category: "Research Notes"
image: "/images/publications/bias-variance.png"
tags:
  - deep learning
  - statistical learning
  - bias-variance tradeoff
excerpt: "A bias–variance decomposition for tracking how bias, variance, and irreducible noise evolve throughout training."
---

<p class="article-lede">Bias and variance are usually studied at the end of training. Here, we instead track how they evolve <em>during</em> optimization, one epoch at a time.</p>

Suppose that we are training a model parameterized by $\theta$, and let $\theta_t$ denote the parameters at step $t$ produced by the optimization algorithm of our choice. In machine learning, it is often helpful to decompose the error $E(\theta)$ as $B^2(\theta)+V(\theta)+N(\theta)$, where $B$ represents the bias, $V$ the variance, and $N$ the noise (irreducible error). In most cases, the decomposition is performed at an optimal solution $\theta^*$—for instance, $\lim_{t \to \infty}\theta_t$, or an early-stopped version—to understand how bias and variance change with model complexity, model size, and related quantities. This has helped explain phenomena such as <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="model-wise-double-descent" aria-label="Explain model-wise double descent">model-wise double descent</button><span id="model-wise-double-descent" class="explanation-popover" popover="auto" role="note" aria-label="Model-wise double descent" data-label="Concept">As model capacity grows, test error may first decrease, then increase near the interpolation threshold, and finally decrease again in the overparameterized regime.</span></span>. It can also be useful to visualize how $B(\theta_t)$ and $V(\theta_t)$ evolve with $t$, which can help explain <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="epoch-wise-double-descent" aria-label="Explain epoch-wise double descent">epoch-wise double descent</button><span id="epoch-wise-double-descent" class="explanation-popover" popover="auto" role="note" aria-label="Epoch-wise double descent" data-label="Concept">For a fixed model, test error can decrease, rise around the time the training data are fitted, and then decrease again as training continues.</span></span>. That is what we will study in this post.

> <span class="article-kicker article-kicker--paper">Source</span> An earlier version of these notes is available on [HackMD](https://hackmd.io/@6LQ4mvRtS4Sc3LHkNEvDXQ/HJl86oh__2).

## Notations

* $\mathcal{X}$ : domain set (input space)
* $\mathcal{Y}$ : label set (output space)
* $\mathcal{H}$ : hypothesis class (class of possible models we can learn)

## Definitions and preliminaries

<span class="article-kicker article-kicker--definition">Definition 1</span> **Loss function.** The loss function $\ell(t,y)$ takes two labels, produces a value between $0$ and some constant $M\in[0,\infty]$, and measures the cost of predicting $y$ when the true value is $t$.

$$
\begin{align*}
\ell \colon \mathcal{Y} \times \mathcal{Y} &\to [0, M] \\
(t, y) &\mapsto \ell(t, y)
\end{align*}
$$

Examples include square loss $\ell(t,y)=(t-y)^2$, absolute loss $\ell(t,y)=|t-y|$, and zero-one loss $\ell(t,y)=\mathbb{1}_{\{t\ne y\}}$.

<span class="article-kicker article-kicker--definition">Definition 2</span> **Training set.** Let $S$ be a set of $|S|$ observations $z_i=(x_i,t_i)\in\mathcal X\times\mathcal Y$, where $x_i\in\mathcal X$ is a feature vector and $t_i\in\mathcal Y$ is the label of the $i$-th sample. The observations are assumed to be <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="iid-samples" aria-label="Explain i.i.d. samples">i.i.d.</button><span id="iid-samples" class="explanation-popover" popover="auto" role="note" aria-label="Independent and identically distributed samples" data-label="Notation"><strong>i.i.d.</strong> means independent and identically distributed: every $z_i$ follows the same distribution $\mathcal D$, and observing one sample does not change the distribution of the others.</span></span> draws from an unknown data distribution $\mathcal D$.

$$
S = \{z_1, \cdots , z_n\}
$$

Since training-set size is an important parameter of a learning problem, we assume below that all datasets have the same size $n$.

<span class="article-kicker article-kicker--definition">Definition 3</span> **Optimal prediction.**

Let $x$ be random and let $t=t(x)$ be either deterministic or random conditional on $x$. The <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="joint-conditional-factorization" aria-label="Explain the joint and conditional factorization">factorization $p(x,t)=p(x)p(t\mid x)$</button><span id="joint-conditional-factorization" class="explanation-popover" popover="auto" role="note" aria-label="Joint and conditional factorization" data-label="Probability">The joint distribution of an input and label equals the marginal distribution of the input multiplied by the conditional distribution of the label given that input.</span></span> gives the optimal prediction at $x$ as

$$
y^*(x)=\arg\min_y \mathbb E_{t\sim p(t\mid x)}[\ell(t,y)].
$$

In the deterministic case, there exists $t\in\mathcal Y$ such that $p(t\mid x)=1$, and therefore $y^*(x)=\arg\min_y\ell(t,y)=t$.

In the nondeterministic case:

* Using square loss, we have

$$
\begin{split}
y^{*}(x)
&= \arg\min_{y} \mathbb{E}_{t \sim p(t\mid x)}[(t - y)^2]
\\ &= \arg\min_{y} \mathbb{E}_{t \sim p(t\mid x)}[t^2 - 2yt  + y^2]
\\ &= \arg\min_{y} \ y^2 - 2 \mathbb{E}_{t \sim p(t\mid x)} [t] y + \mathbb{E}_{t \sim p(t\mid x)} [t^2]
\\ &= \mathbb{E}_{t \sim p(t\mid x)} [t]
\end{split}
$$

Thus the optimal prediction is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="conditional-mean-optimum" aria-label="Explain why square loss gives the conditional mean">conditional mean</button><span id="conditional-mean-optimum" class="explanation-popover" popover="auto" role="note" aria-label="Conditional mean under square loss" data-label="Derivation">As a function of $y$, the expected square loss is a convex quadratic. Its derivative is $2(y-\mathbb E[t\mid x])$, which vanishes at $y=\mathbb E[t\mid x]$.</span></span> of $p(t\mid x)$. For example, suppose that, for a fixed $x$, $(x,1)$ occurs with probability $p_x\in[0,1]$ and $(x,0)$ occurs with probability $1-p_x$:

$$
t(x)=
\begin{cases}
1, & \text{with probability }p_x,\\
0, & \text{with probability }1-p_x.
\end{cases}
$$

Then, using square loss, we have $y^*(x) = p_x$.

* Using absolute loss, we have

$$
\begin{split}
y^*(x)
&= \arg\min_{y} \mathbb{E}_{t \sim p(t\mid x)}[|y - t|]
\\ &= \arg\min_{y} \ 2 \int_{-\infty}^{y} F_{t\mid x}(t)dt - y + \mathbb{E}_{t \sim p(t\mid x)} [t]  + 2\lim_{t \rightarrow - \infty} t F_{t\mid x}(t)
\\ &= F_{t|x}^{-1}(1/2)
\end{split}
$$

where $F_{t\mid x}$ is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="cdf-generalized-inverse" aria-label="Define the cumulative distribution function and generalized inverse">conditional CDF</button><span id="cdf-generalized-inverse" class="explanation-popover" popover="auto" role="note" aria-label="Cumulative distribution function and generalized inverse" data-label="Definition">$F_{t\mid x}(u)=\mathbb P(t\le u\mid x)$. Its generalized inverse is $F_{t\mid x}^{-1}(q)=\inf\{u:F_{t\mid x}(u)\ge q\}$, which remains meaningful for discrete distributions.</span></span>. Thus $y^*(x)$ is a median of $p(t\mid x)$.

The last line follows because $y\mapsto 2F_{t\mid x}(y)-1$ is the derivative of the function

$$
y \mapsto 2 \int_{-\infty}^{y} F_{t\mid x}(t)dt - y + \mathbb{E}_{t \sim p(t\mid x)} [t] +2\lim_{t \rightarrow - \infty} t F_{t\mid x}(t),
$$

which therefore reaches its minimum at a median, where $y=F_{t\mid x}^{-1}(1/2)$.

<details class="article-disclosure">
<summary>Derivation of the absolute-loss identity</summary>
<div class="article-disclosure__body">

The second line follows from

$$
\begin{split}
\mathbb{E}_{t \sim p(t\mid x)}[|y - t|]
&= \mathbb{E}_{t \sim p(t\mid x)}\Big[(y - t) \mathbb{I}[y\ge t] - (y - t) \mathbb{I}[y<t] \Big]
\\ &= \mathbb{E}_{t \sim p(t\mid x)}\Big[(y - t) \mathbb{I}[y\ge t] - (y - t) (1-\mathbb{I}[y\ge t]) \Big]
\\ &= \mathbb{E}_{t \sim p(t\mid x)}\Big[(y - t) (2\mathbb{I}[y\ge t] - 1) \Big]
\\ &= \mathbb{E}_{t \sim p(t\mid x)}\Big[2y \mathbb{I}[y\ge t] - y - 2 t \mathbb{I}[y\ge t] + t \Big]
\\ &= 2y\mathbb{E}_{t \sim p(t\mid x)}\big[\mathbb{I}[y\ge t] \big] - y\mathbb{E}_{t \sim p(t\mid x)}[1] - 2\mathbb{E}_{t \sim p(t\mid x)}\big[t\mathbb{I}[y\ge t] \big] + \mathbb{E}_{t \sim p(t\mid x)}[t]
\\ &= 2yF_{t\mid x}(y) - y - 2 \int_{[-\infty, y]} t \ dF_{t\mid x}(t) + \mathbb{E}_{t \sim p(t\mid x)}[t]
\\ &= 2yF_{t\mid x}(y) - y - 2 \Big( [t F_{t\mid x}(t)]_{-\infty}^{y} - \int_{[-\infty, y]} F_{t\mid x} (t)dt \Big) + \mathbb{E}_{t \sim p(t\mid x)}[t]
\\ &= 2yF_{t\mid x}(y) - y - 2 \Big( y F_{t\mid x}(y) - \lim_{t \rightarrow - \infty} t F_{t\mid x}(t) - \int_{-\infty}^y F_{t\mid x}(t)dt \Big) + \mathbb{E}_{t \sim p(t\mid x)}[t]
\\ &= 2yF_{t\mid x}(y) - y - 2yF_{t\mid x}(y) + 2\lim_{t \rightarrow - \infty} t F_{t\mid x}(t) + 2\int_{-\infty}^y F_{t\mid x}(t)dt + \mathbb{E}_{t \sim p(t\mid x)}[t]
\\ &= 2\int_{-\infty}^y F_{t\mid x}(t)dt - y + \mathbb{E}_{t \sim p(t\mid x)}[t] + 2\lim_{t \rightarrow - \infty} t F_{t\mid x}(t)
\end{split}
$$

</div>
</details>

For the Bernoulli example above, $F_{t\mid x}(u)=(1-p_x)\mathbb{1}_{\{0\le u<1\}}+\mathbb{1}_{\{1\le u\}}$, so
$y^*(x)
= \mathbb{1}_{\{1-p_x<1/2\}}+\lambda\mathbb{1}_{\{1-p_x=1/2\}}
= \mathbb{1}_{\{p_x>1/2\}}+\lambda\mathbb{1}_{\{p_x=1/2\}},\qquad \lambda\in[0,1].$

* Using zero-one loss, we have

$$
\begin{split}
y^*(x)
&= \arg\min_{y} \mathbb{E}_{t \sim p(t\mid x)}[\mathbb{I}[t\ne y]]
\\ &= \arg\min_{y} \mathbb{E}_{t \sim p(t\mid x)}[1-\mathbb{I}[t=y]]
\\ &= \arg\min_{y} 1 - \mathbb{E}_{t \sim p(t\mid x)}[\mathbb{I}[t=y]]
\\ &= \arg\max_{y} \mathbb{E}_{t \sim p(t\mid x)}[\mathbb{I}[t=y]]
\\ &= \arg\max_{t} p(t\mid x) = \arg\max_{t} p(t,x)
\end{split}
$$

That is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="conditional-mode" aria-label="Explain the mode under zero-one loss">conditional mode</button><span id="conditional-mode" class="explanation-popover" popover="auto" role="note" aria-label="Conditional mode" data-label="Definition">A mode is any label with maximal conditional probability $p(t\mid x)$. Predicting it minimizes the probability of misclassification under zero-one loss.</span></span>. For the same Bernoulli example, we have

$$
\begin{split}
y^*(x) &= \arg\min_y \ p_x\mathbb{I}[y\ne 1] + (1-p_x)\mathbb{I}[y\ne 0]
\\ &= \mathbb{I}[p_x\ge 1-p_x]
\\ &= \mathbb{I}[p_x\ge 1/2]
\end{split}
$$

<span class="article-kicker article-kicker--definition">Definition 4</span> **Learning algorithm.** A learning algorithm is a map $\mathcal A:(\mathcal X\times\mathcal Y)^n\to\mathcal H$. It takes a dataset $S\in(\mathcal X\times\mathcal Y)^n$ containing $n$ samples and returns a model $h=\mathcal A(S)\in\mathcal H$.

The optimal model satisfies $f(x)=y^*(x)$ for every $x$. Under zero-one loss, this is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="bayes-classifier-rate" aria-label="Define the Bayes classifier and Bayes rate">Bayes classifier</button><span id="bayes-classifier-rate" class="explanation-popover" popover="auto" role="note" aria-label="Bayes classifier and Bayes rate" data-label="Definition">The Bayes classifier predicts a most probable class conditional on $x$. The smallest classification error it can attain is the Bayes rate; this is the irreducible classification error induced by class overlap or label noise.</span></span>. In the binary example above, it is $f(x)=\mathbb{1}_{\{\mathbb P(t=1\mid x)\ge 1/2\}}$.

<span class="article-kicker article-kicker--definition">Definition 5</span> **True risk.** Given $h\in\mathcal H$,

$$
R[h] = \mathbb{E}_{(x,t) \sim \mathcal{D}}[ \ell(t, h(x))]
$$

<span class="article-kicker article-kicker--definition">Definition 6</span> **Empirical risk.** For $h\in\mathcal H$ and $S=\{(x_1,t_1),\ldots,(x_n,t_n)\}$,

$$
\hat{R}_S[h] =\frac{1}{n} \sum_{i=1}^n \ell(t_i, h(x_i))
$$

The essential task of supervised learning is to obtain good performance on unseen data by adjusting $h$ using one sampled training set $S$. The <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="true-empirical-risk" aria-label="Compare true and empirical risk">two risks differ</button><span id="true-empirical-risk" class="explanation-popover" popover="auto" role="note" aria-label="True and empirical risk" data-label="Comparison">True risk averages over the unknown population distribution $\mathcal D$; empirical risk averages over the finite observed training set. Generalization concerns how well the latter controls the former.</span></span>. We recall them mainly to make the dependence $h=\mathcal A(S)$ explicit.

<span class="article-kicker article-kicker--definition">Definition 7</span> **Expected loss at an input.**

Since the same learner $\mathcal A$ generally produces different models $h$ for different training sets $S$, the loss $\ell(t,h(x))$ depends on $S$ through $h=\mathcal A(S)$. We expose this dependency by averaging over training sets.

Let $D_n$ be a collection of training sets of size $n$, let $\hat y_n(x)$ denote the prediction at $x$ obtained by applying the learner to a sampled training set, and let $Y_n(x)=\{\mathcal A(S)(x):S\in D_n\}$ be the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="prediction-multiset" aria-label="Explain why the predictions form a multiset">multiset of predictions</button><span id="prediction-multiset" class="explanation-popover" popover="auto" role="note" aria-label="Multiset of predictions" data-label="Notation">A multiset keeps multiplicity: the same numerical prediction appears several times when several training sets produce it. Equivalently, one may regard $Y_n(x)$ as the empirical distribution induced by sampling $S$ from $D_n$.</span></span>.

The quantity of interest is the expected loss

$$
E_n(x)
= \mathbb{E}_{D_n, \ t \sim p(t\mid x)}[\ell(t, \hat{y}_n(x))]
= \mathbb{E}_{y \sim Y_n(x), \ t \sim p(t\mid x)}[\ell(t, y)]
$$

Our objective is to decompose $E_n(x)$ into three terms: <span class="article-mark article-mark--rose">bias</span>, <span class="article-mark article-mark--teal">variance</span>, and <span class="article-mark article-mark--gold">noise (irreducible error)</span>. A standard decomposition exists for square loss, and several alternatives have been proposed for zero-one loss.

<span class="article-kicker article-kicker--definition">Definition 8</span> **Main prediction.** For a loss function $\ell$ and a collection of training sets $D_n$, the main prediction is

$$
y^{\ell, D_n} (x)
= \arg\min_{y'} \mathbb{E}_{D_n}[\ell(\hat{y}_n(x), y')]
= \arg\min_{y'} \mathbb{E}_{y \sim Y_n(x)}[\ell(y, y')]
$$

In words, the main prediction minimizes its average loss relative to all predictions in $Y_n(x)$. It is the prediction that “differs least” from the learner's possible predictions according to $\ell$, and therefore describes their central tendency.

> <span class="article-kicker article-kicker--remark">Remark</span> The main prediction need not belong to $Y_n(x)$. For example, the mean of finitely many predictions can lie strictly between all observed values.

> <span class="article-kicker article-kicker--theorem">Theorem 1</span> Under square loss, the main prediction is the mean of $Y_n(x)$; under absolute loss, it is a median; and under zero-one loss, it is a mode (a most frequent prediction).

<details class="article-disclosure">
<summary>Proof for the mean, median, and mode</summary>

* Under square loss, the main prediction is the mean because

$$
\begin{split}
y^{\ell, D_n}(x)
&= \arg\min_{y'} \mathbb{E}_{y \sim Y_n(x)}[(y - y')^2]
\\ &= \arg\min_{y'} \ {y'}^2 - 2 \mathbb{E}_{y \sim Y_n(x)} [y] y' + \mathbb{E}_{y \sim Y_n(x)} [y^2]
\\ &= \mathbb{E}_{y \sim Y_n(x)}[y]
\end{split}
$$

* Under absolute loss, it is a median because

$$
\begin{split}
y^{\ell, D_n}(x)
&= \arg\min_{y'} \mathbb{E}_{y \sim Y_n(x)}[|y' - y|]
\\ &= \arg\min_{y'} \ 2 \int_{-\infty}^{y'} F_{Y_n(x)}(y)dy - y' + \mathbb{E}_{y \sim Y_n(x)} [y]  + 2\lim_{y \rightarrow - \infty} y F_{Y_n(x)}(y)
\\ &= F_{Y_n(x)}^{-1}(1/2)
\end{split}
$$

where $F_{Y_n(x)}$ is the cumulative distribution function of $y\sim Y_n(x)$. This is the same absolute-loss derivation used in Definition 3.

* Under zero-one loss, it is a mode because

$$
\begin{split}
y^{\ell, D_n}(x)
&= \arg\min_{y'} \mathbb{E}_{y \sim Y_n(x)}[\mathbb{I}[y\ne y']]
\\ &= \arg\min_{y'} \mathbb{E}_{y \sim Y_n(x)}[1-\mathbb{I}[y=y']]
\\ &= \arg\min_{y'} 1 - \mathbb{E}_{y \sim Y_n(x)}[\mathbb{I}[y=y']]
\\ &= \arg\max_{y'} \mathbb{E}_{y \sim Y_n(x)}[\mathbb{I}[y=y']]
\\ &= \arg\max_{y'} f_{Y_n(x)} (y')
\end{split}
$$

where $f_{Y_n(x)}$ is the probability-mass function when $Y_n(x)$ is discrete, or a density when it is continuous.

</details>

<span class="article-kicker article-kicker--definition">Definition 9</span> **Bias, variance, and noise.** For an input $x$, define

$$
\begin{gathered}
B^2(x) = \ell( y^*(x), y^{\ell, D_n}(x))
\\ V(x) = \mathbb{E}_{D_n}[\ell( y^{\ell, D_n}(x), y_n(x))]
= \mathbb{E}_{y \sim Y_n(x)}[\ell( y^{\ell, D_n}(x), y )]
\\ N(x) = \mathbb{E}_{t \sim p(t\mid x)}[\ell(t, y^*(x))]
\end{gathered}
$$

In words, the square bias is the loss of the main prediction relative to the optimal prediction; the variance is the average loss of individual learned predictions relative to the main prediction; and the noise is the unavoidable component, independent of the learning algorithm. In the deterministic case, $N(x)=\ell(t(x),t(x))$ for every $x$.

Bias and variance may be averaged over all examples, in which case we will refer to them as average square bias

$$
\mathbb{E}_{x \sim p(x)}[B^2(x)]
$$

and average variance

$$
\mathbb{E}_{x \sim p(x)}[V(x)].
$$

The average noise is

$$
\mathbb{E}_{x \sim p(x)}[N(x)] = \mathbb{E}_{(x,t) \sim p(x,t)}[\ell(t, y^*(x))]
$$

> <span class="article-kicker article-kicker--theorem">Theorem 2</span> For square loss $\ell(t,y)=(t-y)^2$,

$$
V(x) = \mathbb{E}_{y \sim Y_n(x)}[y^2] - ( y^{\ell, D_n}(x))^2
\text{ and }
N(x) = \mathbb{E}_{t \sim p(t\mid x)}[t^2] - (y^*(x))^2
$$

<details class="article-disclosure">
<summary>Proof of the variance and noise identities</summary>

$$
\begin{split}
V(x)
&= \mathbb{E}_{y \sim Y_n(x)}[( y^{\ell, D_n}(x) - y)^2]
\\&= \mathbb{E}_{y \sim Y_n(x)}[( y^{\ell, D_n}(x))^2] - 2 y^{\ell, D_n}(x) \mathbb{E}_{y \sim Y_n(x)}[y] + \mathbb{E}_{y \sim Y_n(x)}[y^2]
\\&= ( y^{\ell, D_n}(x))^2 - 2 ( y^{\ell, D_n}(x))^2 + \mathbb{E}_{y \sim Y_n(x)}[y^2]
\\&= \mathbb{E}_{y \sim Y_n(x)}[y^2] - ( y^{\ell, D_n}(x))^2
\end{split}
$$

$$
\begin{split}
N(x)
&= \mathbb{E}_{t \sim p(t\mid x)}[(t - y^*(x))^2]
\\&= \mathbb{E}_{t \sim p(t\mid x)}[t^2] - 2 y^*(x) \mathbb{E}_{t \sim p(t\mid x)}[t] + \mathbb{E}_{t \sim p(t\mid x)}[(y^*(x))^2]
\\&= \mathbb{E}_{t \sim p(t\mid x)}[t^2] - 2 (y^*(x))^2 + (y^*(x))^2
\\&= \mathbb{E}_{t \sim p(t\mid x)}[t^2] - (y^*(x))^2
\end{split}
$$

</details>

## Bias-variance decomposition

For a given loss function $\ell$, we seek two constants $c_1(x,\ell)$ and $c_2(x,\ell)$ such that

$$
E_n(x) = \ B^2(x) + c_1(x, \ell) \ V(x) + c_2(x, \ell) \ N(x)
$$

> <span class="article-kicker article-kicker--theorem">Theorem 3</span> For square loss $\ell(t,y)=(t-y)^2$, $c_1(x,\ell)=c_2(x,\ell)=1$.

<details class="article-disclosure">
<summary>Proof of the square-loss decomposition</summary>

$$
\begin{split}
E_n(x)
&= \mathbb{E}_{y \sim Y_n(x),\,t \sim p(t\mid x)}[(t-y)^2]
\\ &= \mathbb{E}_{y \sim Y_n(x),\,t \sim p(t\mid x)}[(t-y^*(x)+y^*(x)-y)^2]
\\ &= \mathbb{E}_{t \sim p(t\mid x)}[(t-y^*(x))^2] + 2\bigl(\mathbb{E}_{t \sim p(t\mid x)}[t]-y^*(x)\bigr)\bigl(y^*(x)-\mathbb{E}_{y \sim Y_n(x)}[y]\bigr) + \mathbb{E}_{y \sim Y_n(x)}[(y^*(x)-y)^2]
\\ & = N(x) + 2\times0\times(y^*(x) - \mathbb{E}_{y \sim Y_n(x)}[y]) + \mathbb{E}_{y \sim Y_n(x)}[(y^*(x) - y^{\ell, D_n}(x) + y^{\ell, D_n}(x) - y)^2]
\\ & = N(x) + (y^*(x) - y^{\ell, D_n}(x))^2 + 2(y^*(x) - y^{\ell, D_n}(x))(y^{\ell, D_n}(x) - \mathbb{E}_{y \sim Y_n(x)}[y]) + \mathbb{E}_{y \sim Y_n(x)}[(y^{\ell, D_n}(x) - y)^2]
\\ & = N(x) + B^2(x) + 2(y^*(x) - y^{\ell, D_n}(x))\times 0 + V(x)
\\ & = B^2(x)+V(x)+N(x)
\end{split}
$$

</details>

Let $\mathbb{P}_{D_n}(x) = \mathbb{P}[y^*(x) \in Y_n(x)]$ be the probability over training sets in $D_n$ that the learner predicts the optimal class for $x$.

> <span class="article-kicker article-kicker--theorem">Theorem 4</span> For zero-one loss $\ell(t,y)=\mathbb{1}_{\{t\ne y\}}$ in binary classification, $c_1(x,\ell)=2\mathbb P_{D_n}(x)-1$ and $c_2(x,\ell)=2\mathbb{1}_{\{y^{\ell,D_n}(x)=y^*(x)\}}-1$.

The proof is given by <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="domingos-source" aria-label="Show the source for the zero-one-loss decomposition">Domingos (2000)</button><span id="domingos-source" class="explanation-popover" popover="auto" role="note" aria-label="Domingos source" data-label="Source">Pedro Domingos, <a href="https://www.aaai.org/Papers/AAAI/2000/AAAI00-084.pdf">“A Unified Bias-Variance Decomposition for Zero-One and Squared Loss,”</a> <em>AAAI 2000</em>, pp. 564–569.</span></span>.

The same paper treats multiclass zero-one loss and absolute loss $\ell(t,y)=|t-y|$.

## Application: teacher–student setup

> <span class="article-kicker article-kicker--remark">Work in progress</span> The teacher–student application remains in the earlier [HackMD version](https://hackmd.io/@6LQ4mvRtS4Sc3LHkNEvDXQ/HJl86oh__2) while I verify and format its derivation for this post.

## References

- Pedro Domingos, [“A Unified Bias-Variance Decomposition for Zero-One and Squared Loss”](https://www.aaai.org/Papers/AAAI/2000/AAAI00-084.pdf), *Proceedings of the 17th National Conference on Artificial Intelligence*, pp. 564–569, 2000.

- IFT 6085, Lecture 9: *Stability, Generalization and the Applications of Stability*.
