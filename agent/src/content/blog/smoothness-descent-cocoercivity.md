---
title: "Smoothness, Descent, and Cocoercivity"
date: "2025-02-05"
category: "Mathematics"
tags:
  - Optimization
  - Smoothness
  - Convex Analysis
  - Cocoercivity
excerpt: "Definitions, lemmas, and proofs connecting smoothness, descent guarantees, Hessian bounds, and the Baillon–Haddad theorem."
---

This document provides a rigorous and detailed exploration of **smoothness** for real-valued functions on $\mathbb{R}^p$, including formal definitions, key lemmas, and proofs. It connects smoothness to Lipschitz continuity of the gradient, descent guarantees in optimization, Hessian bounds, and cocoercivity. The relationships between convexity, smoothness, and cocoercivity are clarified through the Baillon–Haddad theorem.

## Notations

* For a vector $\mathbf{x} \in \mathbb{R}^n$, $\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |\mathbf{x}_i|^{p} \right)^{\frac{1}{p}} \ \forall p \in (0, \infty)$ and $\|\mathbf{x}\|_\infty = \max_{i \in [n]} |\mathbf{x}_i|$.
* $\sigma_{\max/\min}(\mathbf{A})$ is the maximum (resp. minimum) singular value of a matrix $\mathbf{A}$, with $\lambda_{\max/\min}(\mathbf{A})$ the corresponding eigenvalue
* For a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the induced $p \rightarrow q$ norm of $\mathbf{A}$ is $\|\mathbf{A}\|_{p \rightarrow q} = \sup_{\mathbf{x} \ne 0 }  \frac{\|\mathbf{A}\mathbf{x}\|_q}{\|\mathbf{x}\|_p} = \sup_{\|\mathbf{x}\|_p = 1}  \|\mathbf{A}\mathbf{x}\|_q$. We have $\|\mathbf{A}\|_{2 \rightarrow 2} = \sigma_{\max}(\mathbf{A})$ (operator norm, spectral norm, induced $2$-norm).

## Definition 1 [Subdifferentiability]
A function $\varphi : \mathbb{R}^p \to \mathbb{R}$ is said to be subdifferentiable at $\mathbf{x} \in \mathbb{R}^p$ if and only if

$$
\exists \mathbf{z} \in \mathbb{R}^p, \quad \varphi(\mathbf{y}) \ge \varphi(\mathbf{x}) + (\mathbf{y} - \mathbf{x})^\top \mathbf{z} \quad \forall \mathbf{y} \in \mathbb{R}^p
$$

 The set of all such $\mathbf{z}$ at is called the subdifferential of $\varphi$ at $\mathbf{x}$ and is denoted $\partial \varphi(\mathbf{x})$. When $\partial \varphi(\mathbf{x})$ is the singleton set, we say that $\varphi$ is differentiable at $\mathbf{x}$ :

$$
\partial \varphi(\mathbf{x}) = \{ \nabla  \varphi(\mathbf{x})\}
$$

## Definition 2 [Convexity]
A function $\varphi : \mathbb{R}^p \to \mathbb{R}$ is  said to be  convex if and only if

$$
\varphi\left(\lambda \mathbf{x} + (1 - \lambda)\mathbf{y} \right) \le \lambda \varphi(\mathbf{x}) + (1 - \lambda)\varphi(\mathbf{y}) \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p, \quad \forall \lambda \in (0,1)
$$

 If $\varphi$ is subdifferentiable, then this implies

$$
\varphi(\mathbf{y}) \ge \varphi(\mathbf{x}) + (\mathbf{y}-\mathbf{x})^\top \mathbf{z} \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p, \quad \forall \mathbf{z} \in \partial \varphi(\mathbf{x})
$$

 If $\varphi$ is twice-differentiable, this implies

$$
\lambda_{\min}\left( \nabla^2 \varphi(\mathbf{x}) \right) \ge 0 \quad \forall \mathbf{x} \in \mathbb{R}^p
$$

## Definition 3 [Lipschitz Continuity]
For $L > 0$, a function $\Phi : \mathbb{R}^p \to \mathbb{R}^n$ is $L$-Lipschitz continuous if and only if

$$
\begin{equation}
\|\Phi (\mathbf{y}) - \Phi (\mathbf{x}) \|_2 \le L \|\mathbf{y}-\mathbf{x}\|_2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
\end{equation}
$$

When $L=1$, this means that $\Phi$ is nonexpansive.

## Definition 4 [Cocoercivity]
For $L > 0$, a vector field $\Phi : \mathbb{R}^p \to \mathbb{R}^p$ is $1/L$-cocoercive if and only if

$$
\langle \Phi(\mathbf{x}) -  \Phi(\mathbf{y}), \mathbf{x} - \mathbf{y}) \rangle \ge \frac{1}{L} \|  \Phi(\mathbf{x}) - \Phi(\mathbf{y}) \|_2^2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
$$

 The cocoercivity is also often called the Dunn property or Inverse strong monotonicity. When $L=1$, this means that $\Phi$ is firmly nonexpansive.

## Definition 5 [Smoothness]
For $L > 0$, a differentiable function $\varphi : \mathbb{R}^p \to \mathbb{R}$ is $L$-smooth if and only if $\nabla \varphi : \mathbb{R}^p \to \mathbb{R}^p$ is $L$-Lipschitz continuous.

## Lemma 1 [Descent Lemma]
Let $L > 0$, and $\varphi : \mathbb{R}^p \to \mathbb{R}$, differentiable.
* If $\varphi$ is $L$-smooth, then it satisfies the upper-quadratic inequality (UQI)

$$
\varphi(\mathbf{y}) \le \varphi(\mathbf{x}) + (\mathbf{y}-\mathbf{x})^\top \nabla \varphi(\mathbf{x}) + \frac{L}{2} \|\mathbf{y}-\mathbf{x}\|_2^2 \ \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p \quad \text{(UQI)}
$$

* The converse is false in general, but true for convex functions. That mean if $\varphi$ is convex and satisfies the upper-quadratic inequality above, then it is $L$-smooth.

### Remark

This is call the descent Lemma because for any $\alpha \in (0, 2/L]$, moving from any point $\mathbf{x} \in \mathbb{R}^p$ to $\mathbf{x} - \alpha \nabla \varphi(\mathbf{x})$ decreases $\varphi$ (Corollary 1), as well as its gradient (Corollary 2).

### Proof

#### ($\Longrightarrow$) L-Smoothness implies UQI

Let $\mathbf{x}, \mathbf{y} \in \mathbb{R}^p$. We reduce the multivariate statement to the one–variable case by slicing $\varphi$ along the line segment joining $\mathbf{x}$ and $\mathbf{y}$. Define $\psi:[0,1] \longrightarrow \mathbb{R}$ by

$$
\psi(t) = \varphi\left(\mathbf{x}+t(\mathbf{y}-\mathbf{x})\right)
$$

  Since $\psi(1) = \psi(0)+\int_{0}^{1} \psi'(t)dt$ with $\psi'(t) = \langle  \nabla \varphi\left(x+t(\mathbf{y}-\mathbf{x})\right), \mathbf{y}-\mathbf{x} \rangle$, we have

$$
\begin{equation}
\begin{split}
\varphi(\mathbf{y})
&= \varphi(\mathbf{x}) + \int_{0}^{1} \langle\nabla \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x})), \mathbf{y}-\mathbf{x} \rangle dt
\\ &= \varphi(\mathbf{x}) + \langle\nabla \varphi(\mathbf{x}), \mathbf{y}-\mathbf{x} \rangle + \int_{0}^{1} \langle \nabla \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x}))-\nabla \varphi(\mathbf{x}), \mathbf{y}-\mathbf{x} \rangle dt
\\ & \le \varphi(\mathbf{x}) + \langle\nabla \varphi(\mathbf{x}), \mathbf{y}-\mathbf{x} \rangle + \int_{0}^{1} \|\nabla \varphi\left(\mathbf{x}+t(\mathbf{y}-\mathbf{x})\right) - \nabla \varphi(\mathbf{x}) \|_2 \|\mathbf{y}-\mathbf{x}\|_2 dt
\\ & \le \varphi(\mathbf{x}) + \langle\nabla \varphi(\mathbf{x}), \mathbf{y}-\mathbf{x} \rangle + \int_{0}^{1} L \| \mathbf{x}+t(\mathbf{y}-\mathbf{x}) - \mathbf{x} \|_2 \|\mathbf{y}-\mathbf{x}\|_2 dt
\\& = \varphi(\mathbf{x}) +\bigl\langle\nabla \varphi(\mathbf{x}),\, \mathbf{y}-\mathbf{x} \bigr\rangle + L \|\mathbf{y}-\mathbf{x}\|_2^{2} \int_{0}^{1} t dt
\\& = \varphi(\mathbf{x}) + \bigl\langle\nabla \varphi(\mathbf{x}), \mathbf{y}-\mathbf{x} \bigr\rangle + \frac{L}{2} \|\mathbf{y}-\mathbf{x}\|_2^{2}
\end{split}
\end{equation}
$$

#### ($\Longleftarrow$) $L$-UQI + Convex implies $L$-Smoothness

Now let $p=1$ and $\varphi(x) = -a x^{2}$ for some $a>0$. We have $\nabla\varphi(x)=-2a x$, and for every $L>0$,

$$
\begin{equation}
\begin{split}
&  - a (y - x)^2 -\tfrac{L}{2}(y-x)^{2} \le 0 \quad \forall x, y \in \mathbb{R}
\\ & \Longleftrightarrow - a y^{2} + 2a xy - ax^2 -\tfrac{L}{2}(y-x)^{2} \le 0 \quad \forall x, y \in \mathbb{R}
\\ & \Longleftrightarrow - a y^{2} + a x^{2} + 2a xy- 2ax^2 -\tfrac{L}{2}(y-x)^{2} \le 0 \quad \forall x, y \in \mathbb{R}
\\ & \Longleftrightarrow -a y^{2} + a x^{2} +2a x(y-x) -\tfrac{L}{2}(y-x)^{2} \le 0 \quad \forall x, y \in \mathbb{R}
\\ & \Longleftrightarrow  -a y^{2} \le -a x^{2} + (-2a x)(y-x) + \tfrac{L}{2}(y-x)^{2} \quad \forall x, y \in \mathbb{R}
\\ & \Longleftrightarrow \varphi(y) \le \varphi(x) + (y-x)^\top \nabla \varphi(x) + \frac{L}{2} \|y-x\|_2^2 \quad \forall x, y \in \mathbb{R}
\end{split}
\end{equation}
$$

However, $\varphi$ is $L$-smooth if and only if $L \ge 2a$, since $\nabla^2\varphi(x)=-2a$. Therefore, choosing $0<L<2a$ shows that the upper-quadratic inequality alone does not force $\nabla\varphi$ to be $L$-Lipschitz in the general setting.

Let $\varphi : \mathbb{R}^p \to \mathbb{R}$ be a differentiable convex function, i.e.

$$
\varphi(\mathbf{y}) \ge \varphi(\mathbf{x}) + (\mathbf{y}-\mathbf{x})^\top \nabla \varphi(\mathbf{x}) \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
$$

Assume there exists $L>0$ such that

$$
\varphi(\mathbf{y}) \le \varphi(\mathbf{x}) + (\mathbf{y}-\mathbf{x})^\top \nabla \varphi(\mathbf{x}) + \frac{L}{2} \|\mathbf{y}-\mathbf{x}\|_2^2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
$$

We want to show that $\varphi$  is $L$-smooth, i.e.

$$
\| \nabla \varphi(\mathbf{y}) - \nabla \varphi(\mathbf{x}) \|_2 \le L \|\mathbf{y}-\mathbf{x}\|_2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
$$

Fix $\mathbf{x}, \mathbf{y} \in \mathbb{R}^p$ and define the affine shift

$$
h_{\mathbf{x}}(\mathbf{z})
:= \varphi(\mathbf{z})-\varphi(\mathbf{x})
- (\mathbf{z}-\mathbf{x})^\top \nabla\varphi(\mathbf{x}).
$$

By convexity, $h_{\mathbf{x}}(\mathbf{z}) \ge 0$ for every $\mathbf{z} \in \mathbb{R}^p$. Subtracting an affine function preserves the upper-quadratic inequality, so

$$
h_{\mathbf{x}}(\mathbf{v})
\le h_{\mathbf{x}}(\mathbf{u})
+ (\mathbf{v}-\mathbf{u})^\top \nabla h_{\mathbf{x}}(\mathbf{u})
+ \frac{L}{2}\|\mathbf{v}-\mathbf{u}\|_2^2
\quad \forall \mathbf{u},\mathbf{v}\in\mathbb{R}^p.
$$

In particular, the assumed inequality for $\varphi$ at $\mathbf{x}$ and $\mathbf{y}$ gives

$$
h_{\mathbf{x}}(\mathbf{y}) \le \frac{L}{2}\|\mathbf{y}-\mathbf{x}\|_2^2.
$$

Now set

$$
\mathbf{d}
:= \nabla h_{\mathbf{x}}(\mathbf{y})
= \nabla\varphi(\mathbf{y})-\nabla\varphi(\mathbf{x}),
\qquad
\mathbf{w}:=\mathbf{y}-\frac{1}{L}\mathbf{d}.
$$

Because the domain is all of $\mathbb{R}^p$, the point $\mathbf{w}$ is admissible. Applying the upper-quadratic inequality for $h_{\mathbf{x}}$ with base point $\mathbf{y}$ and target point $\mathbf{w}$ yields

$$
\begin{aligned}
0
\le h_{\mathbf{x}}(\mathbf{w})
&\le h_{\mathbf{x}}(\mathbf{y})
+ (\mathbf{w}-\mathbf{y})^\top \mathbf{d}
+ \frac{L}{2}\|\mathbf{w}-\mathbf{y}\|_2^2 \\
&= h_{\mathbf{x}}(\mathbf{y})
- \frac{1}{2L}\|\mathbf{d}\|_2^2 \\
&\le \frac{L}{2}\|\mathbf{y}-\mathbf{x}\|_2^2
- \frac{1}{2L}\|\mathbf{d}\|_2^2.
\end{aligned}
$$

Since $L>0$, rearranging gives

$$
\|\mathbf{d}\|_2^2
\le L^2\|\mathbf{y}-\mathbf{x}\|_2^2.
$$

Taking square roots and recalling the definition of $\mathbf{d}$, we obtain

$$
\|\nabla\varphi(\mathbf{y})-\nabla\varphi(\mathbf{x})\|_2
\le L\|\mathbf{y}-\mathbf{x}\|_2.
$$

Because $\mathbf{x}$ and $\mathbf{y}$ were arbitrary, $\nabla\varphi$ is $L$-Lipschitz continuous, so $\varphi$ is $L$-smooth.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

## Corollary 1 [Descent Lemma]
If a function  $\varphi : \mathbb{R}^p \to [0, \infty)$ is $L$-smooth, then for all $\mathbf{x} \in \mathbb{R}^p$, we have

$$
\begin{equation}
\begin{split}
\varphi( \mathbf{x} - \alpha \nabla \varphi(\mathbf{x}) )
& \le \varphi(\mathbf{x})- \frac{\alpha \left( 2 - L\alpha \right)}{2} \|\nabla \varphi(\mathbf{x})\|_2^{2} \quad \forall \alpha \in \mathbb{R}
\\ & \le \varphi(\mathbf{x}) \quad \text{ for } 0 \le \alpha \le 2/L
\end{split}
\end{equation}
$$

### Proof
Fix any $\mathbf{x} \in\mathbb{R}^p$ and set $\mathbf{y} := \mathbf{x} - \alpha \nabla \varphi(\mathbf{x})$ to have

$$
\begin{equation}
\begin{split}
\varphi(\mathbf{y})
& \le \varphi(\mathbf{x}) + (\mathbf{y}-\mathbf{x})^\top \nabla \varphi(\mathbf{x}) + \frac{L}{2} \|\mathbf{y}-\mathbf{x}\|_2^2 \quad \text{ (Lemma 1) }
\\ & = \varphi(\mathbf{x}) - \alpha \|\nabla \varphi(\mathbf{x})\|_2^{2} +\frac{L \alpha^2}{2}  \|\nabla \varphi(\mathbf{x})\|_2^{2}
\\ & = \varphi(\mathbf{x})- \frac{\alpha}{2} \left( 2 - L\alpha \right) \|\nabla \varphi(\mathbf{x})\|_2^{2}
\end{split}
\end{equation}
$$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

## Corollary 2 [Bounded gradient]
If a non-negative function  $\varphi : \mathbb{R}^p \to [0, \infty)$ is $L$-smooth, then for all $\mathbf{x} \in \mathbb{R}^p$, we have

$$
\begin{equation}
\begin{split}
\|\nabla \varphi (\mathbf{x}) \|_2^{2}
& \le \frac{2}{\alpha \left( 2 - L\alpha \right)} \left( \varphi(\mathbf{x}) - \varphi^* \right) \quad \forall \alpha > 0
\\ & = 2L \left( \varphi(\mathbf{x}) - \varphi^* \right) \quad \text{ for } \alpha = 1/L
\end{split}
\end{equation}
$$

with

$$
\varphi^* = \min_{x \in \mathbb{R}^p} \varphi(x) > -\infty
$$

### Proof
Fix any $\mathbf{x} \in\mathbb{R}^p$ and set $\mathbf{y} := \mathbf{x} - \alpha \nabla \varphi(\mathbf{x})$ to have

$$
\begin{equation}
\begin{split}
& \varphi^* \le \varphi(\mathbf{y}) \le \varphi(\mathbf{x})- \frac{\alpha \left( 2 - L\alpha \right)}{2} \|\nabla \varphi(\mathbf{x})\|_2^{2}
\quad \text{ (Corollary 1) }
\\ & \Longrightarrow \|\nabla \varphi (\mathbf{x}) \|_2^{2} \le \frac{2}{\alpha \left( 2 - L\alpha \right)} \left( \varphi(\mathbf{x}) - \varphi^* \right)
\end{split}
\end{equation}
$$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

## Lemma 2 [Bounds on the curvature]
A twice-differentiable function $\varphi : \mathbb{R}^p \to \mathbb{R}$ is $L$-smooth if and only if

$$
-L \le \lambda\left(\nabla^2 \varphi(\mathbf{x}) \right) \le L
\quad \forall \mathbf{x} \in \mathbb{R}^p
$$

### Proof
$\Longrightarrow$) Assume $\varphi$ is $L$-smooth. Fix $\mathbf{x} \in\mathbb{R}^p$. For a vector $\mathbf{v} \in\mathbb{R}^p$ the directional derivative of $\nabla \varphi$ at $\mathbf{x}$ is defined by

$$
\nabla^2 \varphi(\mathbf{x}) \mathbf{v} = \lim_{t \rightarrow 0} \frac{ \nabla \varphi(\mathbf{x}) (\mathbf{x}+t\mathbf{v}) - \nabla \varphi (\mathbf{x})}{t}
$$

Taking norms and using the $L$-Lipschitz property of $\nabla \varphi$,

$$
\begin{equation}
\begin{split}
\|\nabla^2 \varphi(\mathbf{x}) \mathbf{v} \|_2
& = \lim_{t \rightarrow 0} \frac{\| \nabla \varphi (\mathbf{x}+t\mathbf{v}) - \nabla \varphi(\mathbf{x}) \|_2}{t}
\\ & \le \lim_{t \rightarrow 0} \frac{L\, \| (\mathbf{x}+t\mathbf{v})- \mathbf{x} \|_2}{t}
\\ & = \lim_{t \rightarrow 0} \frac{L t \| \mathbf{v} \|_2}{t}
\\ & = L \| \mathbf{v} \|_2
\end{split}
\end{equation}
$$

If $\mathbf{v}$ is an eigenvector of $\nabla^2 \varphi(\mathbf{x})$ associated with the eigenvalue $\lambda$, we have $\lambda \mathbf{v} = \nabla^2 \varphi(\mathbf{x}) \mathbf{v}$, which implies $| \lambda | \| \mathbf{v} \|_2 =  \|\nabla^2 \varphi(\mathbf{x}) \mathbf{v} \|_2 \le L \| \mathbf{v} \|_2$. Dividing both sides by $\| \mathbf{v} \|_2 \ne 0$ we get $| \lambda |  \le L$.

$\Longleftarrow$) Now, assume that for all $\mathbf{x} \in \mathbb{R}^p$, all the eigenvalues of $\nabla^2 \varphi(\mathbf{x})$ are in $[-L, L]$. We reduce the multivariate statement to the one–variable case by slicing $\nabla \varphi$ along the line segment joining $\mathbf{x}$ and $\mathbf{y}$ using

$$
\psi(t) = \nabla \varphi\left(\mathbf{x}+t(\mathbf{y}-\mathbf{x})\right) \ \forall t\in[0,1]
$$

 Since $\psi(1) - \psi(0) = \int_{0}^{1} \psi'(t)dt$ and $\psi'(t) = \nabla^2 \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x})) \left( \mathbf{y}-\mathbf{x} \right)$, we have

$$
\nabla \varphi(\mathbf{y}) - \nabla \varphi(\mathbf{x}) = \int_{0}^{1} \nabla^2 \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x})) \left( \mathbf{y}-\mathbf{x} \right) dt
$$

 which implies

$$
\begin{equation}
\begin{split}
\| \nabla \varphi(\mathbf{y}) - \nabla\varphi(\mathbf{x}) \|_2
&= \left\|  \int_{0}^{1} \nabla^2 \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x})) \left( \mathbf{y}-\mathbf{x} \right) dt \right\|_2
\\ & \le \int_{0}^{1} \left\|  \nabla^2 \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x})) \left( \mathbf{y}-\mathbf{x} \right) \right\|_2 dt
\\ & \le \int_{0}^{1} \left\|  \nabla^2 \varphi(\mathbf{x}+t(\mathbf{y}-\mathbf{x})) \right\|_{2 \to 2}  \left\| \mathbf{y}-\mathbf{x} \right\|_2 dt
\\ & \le \int_{0}^{1} L \left\| \mathbf{y}-\mathbf{x} \right\|_2 dt = L \left\| \mathbf{y}-\mathbf{x} \right\|_2
\end{split}
\end{equation}
$$

 since $\left\|  \nabla^2 \varphi(\cdot) \right\|_{2 \to 2} = \sigma_{\max}\left(  \nabla^2 \varphi( \cdot ) \right) = \lambda_{\max}\left(  \nabla^2 \varphi(\cdot) \right) \le L$. In fact, since $\varphi \in C^2(\mathbb{R}^p)$, the Hessian is a symmetric matrix, so its spectral norm is equal to the maximum absolute value of its eigenvalues (its spectral radius), which by assumption is less than $L$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

## Lemma 3 [Baillon–Haddad]
Let $\varphi : \mathbb{R}^p \to \mathbb{R}$ be a differentiable function.
* If $\nabla \varphi$ is $1/L$-cocoercive, then $\varphi$ is $L$-smooth. But the converse if false in general.
* If $\varphi$ is convex and $L$-smooth, then $\nabla \varphi$ is $1/L$-cocoercive.

### **Proof**

#### ($\Longrightarrow$) Cocoercivity implies L-Smoothness

Fix $\mathbf{x}, \mathbf{y} \in \mathbb{R}^p$. If $\nabla \varphi(\mathbf{x}) = \nabla \varphi(\mathbf{y})$, then we trivially have $\| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2 = 0 \le L \| \mathbf{x} - \mathbf{y} \|_2$. So we assume  $\nabla \varphi(\mathbf{x}) \ne \nabla \varphi(\mathbf{y})$. If $\nabla \varphi$ is $1/L$-cocoercive, then we have  $\frac{1}{L} \| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2^2 \le \left( \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \right)^\top \left( \mathbf{x} - \mathbf{y} \right)$. This  implies

$$
\| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2^2 \le L \| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2 \| \mathbf{x} - \mathbf{y} \|_2
$$

since $\left( \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \right)^\top \left( \mathbf{x} - \mathbf{y} \right) \le \| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2 \| \mathbf{x} - \mathbf{y} \|_2$. By dividing both sides of the inequality by $\| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2 \ne 0$, we get $\| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2 \le L \| \mathbf{x} - \mathbf{y} \|_2$. So $\varphi$ is $L$-smooth.

The converse if false in general. Consider for example $\varphi(\mathbf{x})= - \frac{1}{2}\|\mathbf{x}\|_2^2$. We have $\nabla \varphi(\mathbf{x})= - \mathbf{x}$ and $\nabla^2 \varphi(\mathbf{x})= - \mathbb{I}$, so $\varphi$ is $L$-smooth if and only if $L>1$. But for all $L > 0$, $\nabla \varphi$ is not $1/L$-cocoercive since

$$
\begin{equation}
\begin{split}
& (\nabla \varphi(\mathbf{x}) -  \nabla \varphi(\mathbf{y}))^\top (\mathbf{x} - \mathbf{y}) \ge \frac{1}{L} \| \nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y}) \|_2^2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
\\ & \Longleftrightarrow (-\mathbf{x} + \mathbf{y})^\top (\mathbf{x} - \mathbf{y}) \ge \frac{1}{L} \| -\mathbf{x} + \mathbf{y} \|_2^2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
\\ & \Longleftrightarrow - \| -\mathbf{x} + \mathbf{y} \|_2^2 \ge \frac{1}{L} \| -\mathbf{x} + \mathbf{y} \|_2^2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^p
\\ & \Longleftrightarrow L \le -1
\end{split}
\end{equation}
$$

#### ($\Longleftarrow$) Convexity and L-Smoothness implies Cocoercivity

Now, we assume that $\varphi$ is a convex and $L$-smooth function. We will show that $\nabla \varphi$ is $1/L$-cocoercive. Fix $\mathbf{x}, \mathbf{y} \in \mathbb{R}^p$. We want to show that

$$
(\mathbf{x}-\mathbf{y})^\top (\nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y})) \ge \frac{1}{L} \|\nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y})\|_2^2
$$

Let define an auxiliary function $\psi(\mathbf{z}) := \varphi(\mathbf{z}) - \mathbf{z}^\top \nabla \varphi(\mathbf{x})$. The gradient of $\psi$ is $\nabla \psi(\mathbf{z}) = \nabla \varphi(\mathbf{z}) - \nabla \varphi(\mathbf{x})$, so $\psi$ is $L$-smooth by the smoothness of $\varphi$ :

$$
\| \nabla \psi(\mathbf{z}) - \nabla \psi(\mathbf{t}) \|_2 = \| \nabla \varphi(\mathbf{z}) - \nabla \varphi(\mathbf{t}) \|_2 \le L \| \mathbf{z} - \mathbf{t} \|_2 \quad \forall \mathbf{z}, \mathbf{t} \in \mathbb{R}^p
$$

 $\psi$ is also convex by the convexity of $\varphi$ since

$$
\begin{equation}
\begin{split}
& \varphi\left(\lambda \mathbf{z} + (1 - \lambda)\mathbf{t} \right) \le \lambda \varphi(\mathbf{z}) + (1 - \lambda)\varphi(\mathbf{t})
\quad \forall \mathbf{z}, \mathbf{t} \in \mathbb{R}^p \quad \forall \lambda \in (0,1)
\\ & \Longleftrightarrow \psi\left(\lambda \mathbf{z} + (1 - \lambda)\mathbf{t} \right) + \left(\lambda \mathbf{z} + (1 - \lambda)\mathbf{t} \right)^\top \nabla \varphi(\mathbf{x}) \le \lambda \left( \psi(\mathbf{z}) + \mathbf{z}^\top \nabla \varphi(\mathbf{x})\right) + (1 - \lambda) \left( \psi(\mathbf{t}) + \mathbf{t}^\top \nabla \varphi(\mathbf{x})\right)
\quad \forall \mathbf{z}, \mathbf{t} \in \mathbb{R}^p \quad \forall \lambda \in (0,1)
\\ & \Longleftrightarrow \psi\left(\lambda \mathbf{z} + (1 - \lambda)\mathbf{t} \right) + \lambda \mathbf{z}^\top \nabla \varphi(\mathbf{x}) + (1 - \lambda)\mathbf{t}^\top \nabla \varphi(\mathbf{x})  \le  \lambda \psi(\mathbf{z}) + \lambda \mathbf{z}^\top \nabla \varphi(\mathbf{x}) + (1 - \lambda) \psi(\mathbf{t}) + (1 - \lambda) \mathbf{t}^\top \nabla \varphi(\mathbf{x})
\quad \forall \mathbf{z}, \mathbf{t} \in \mathbb{R}^p \quad \forall \lambda \in (0,1)
\\ & \Longleftrightarrow \psi\left(\lambda \mathbf{z} + (1 - \lambda)\mathbf{t} \right)  \le  \lambda \psi(\mathbf{z}) + (1 - \lambda) \psi(\mathbf{t})
\quad \forall \mathbf{z}, \mathbf{t} \in \mathbb{R}^p \quad \forall \lambda \in (0,1)
\end{split}
\end{equation}
$$

Because $\psi$ is $L$-smooth, it satisfies the upper-bound inequality (Lemma 1):

$$
\psi(\mathbf{z}) \le \psi(\mathbf{t}) + (\mathbf{z}-\mathbf{t})^\top \nabla \psi(\mathbf{t}) + \frac{L}{2} \|\mathbf{z}-\mathbf{t}\|_2^2 \quad \forall \mathbf{z}, \mathbf{t} \in \mathbb{R}^p
$$

 Let's find the point $\mathbf{z}^*(\mathbf{t})$ that minimizes the right-hand side with respect to $\mathbf{z}$. This occurs when

$$
\begin{equation}
\begin{split}
& \nabla_{\mathbf{z}} \left( \psi(\mathbf{t}) + (\mathbf{z}-\mathbf{t})^\top \nabla \psi(\mathbf{t}) + \frac{L}{2} \|\mathbf{z}-\mathbf{t}\|_2^2 \right) = \nabla \psi(\mathbf{t}) + L (\mathbf{z}-\mathbf{t}) = 0
\\ & \Longrightarrow \mathbf{z}^*(\mathbf{t}) = \mathbf{t} - \frac{1}{L}\nabla \psi(\mathbf{t})
\end{split}
\end{equation}
$$

Plugging this in gives for all $\mathbf{t} \in \mathbb{R}^p$,

$$
\begin{equation}
\begin{split}
\psi(\mathbf{z}^*(\mathbf{t}))
& \le \psi(\mathbf{t}) + (\mathbf{z}^*(\mathbf{t})-\mathbf{t})^\top \nabla \psi(\mathbf{t}) + \frac{L}{2} \|\mathbf{z}^*(\mathbf{t})-\mathbf{t}\|_2^2
\\ & = \psi(\mathbf{t}) - \frac{1}{L} \|\nabla \psi(\mathbf{t})\|_2^2 + \frac{L}{2}\frac{1}{L^2} \|\nabla \psi(\mathbf{t})\|_2^2
\\ & = \psi(\mathbf{t}) - \frac{1}{2L}\|\nabla \psi(\mathbf{t})\|_2^2
\end{split}
\end{equation}
$$

Since $\nabla \psi(\mathbf{x}) = 0$, $\mathbf{x}$ is a minimizer of the convex function $\psi$. So $\mathbf{x}$ is a global minimizer of $\psi$, which implies $\psi(\mathbf{x}) \le \psi(\mathbf{z}^*(\mathbf{t}))$. Combining these gives:

$$
\begin{equation}
\begin{split}
& \psi(\mathbf{x}) \le \psi(\mathbf{t}) - \frac{1}{2L}\|\nabla \psi(\mathbf{t})\|_2^2 \quad \forall \mathbf{t} \in \mathbb{R}^p
\\ & \Longleftrightarrow \varphi(\mathbf{x}) - \mathbf{x}^\top \nabla \varphi(\mathbf{x}) \le \varphi(\mathbf{t}) - \mathbf{t}^\top \nabla \varphi(\mathbf{x}) - \frac{1}{2L}\| \nabla \varphi(\mathbf{t}) - \nabla \varphi(\mathbf{x}) \|_2^2 \quad \forall \mathbf{t} \in \mathbb{R}^p
\\ & \Longleftrightarrow
\frac{1}{2L} \|\nabla \varphi(\mathbf{t}) - \nabla \varphi(\mathbf{x})\|_2^2 \le
\varphi(\mathbf{t}) - \varphi(\mathbf{x}) - (\mathbf{t}-\mathbf{x})^\top \nabla \varphi(\mathbf{x})
\quad \forall \mathbf{t} \in \mathbb{R}^p
\end{split}
\end{equation}
$$

Using $\mathbf{t}=\mathbf{y}$, we get

$$
\varphi(\mathbf{y}) - \varphi(\mathbf{x}) - (\mathbf{y}-\mathbf{x})^\top \nabla \varphi(\mathbf{x}) \ge \frac{1}{2L} \|\nabla \varphi(\mathbf{y}) - \nabla \varphi(\mathbf{x})\|_2^2
$$

We can also similarly show that

$$
\varphi(\mathbf{x}) - \varphi(\mathbf{y}) - (\mathbf{x}-\mathbf{y})^\top \nabla \varphi(\mathbf{y}) \ge \frac{1}{2L} \|\nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y})\|_2^2
$$

Now, we add these two inequalities together to get the desired result:

$$
\begin{equation}
\begin{split}
& - (\mathbf{y}-\mathbf{x})^\top \nabla \varphi(\mathbf{x}) - (\mathbf{x}-\mathbf{y})^\top \nabla \varphi(\mathbf{y}) \ge \frac{2}{2L} \|\nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y})\|_2^2
\\ & \Longleftrightarrow (\mathbf{x}-\mathbf{y})^\top (\nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y})) \ge \frac{1}{L} \|\nabla \varphi(\mathbf{x}) - \nabla \varphi(\mathbf{y})\|_2^2
\end{split}
\end{equation}
$$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

This result is the Baillon-Haddad theorem restricted to Euclidean spaces. We give the general version below. For now on, we let $\mathcal{H}$ be a real Hilbert space  with scalar product $\langle \cdot, \cdot \rangle$ and induced norm $\| \cdot \|$.
## Definition 6

We say that a function $\varphi : \mathcal{H} \to \mathbb{R}$ is Fréchet differentiable at $\mathbf{x} \in \mathcal{H}$ if there exists a bounded linear operator $D\varphi(\mathbf{x}) : \mathcal{H} \to \mathbb{R}$ such that:

$$
\lim_{\|\mathbf{h}\| \to 0} \frac{|\varphi(\mathbf{x} + \mathbf{h}) - \varphi(\mathbf{x}) - D\varphi(\mathbf{x})(\mathbf{h})|}{\|\mathbf{h}\|} = 0
$$

Equivalently, this means:

$$
\varphi(\mathbf{x} + \mathbf{h}) = \varphi(\mathbf{x}) + D\varphi(\mathbf{x})(\mathbf{h}) + o(\|\mathbf{h}\|) \quad \text{as } \mathbf{h} \to 0
$$

Since $D\varphi(\mathbf{x})$ is a bounded linear functional on $\mathcal{H}$, the Riesz representation theorem guarantees that there exists a unique vector $\nabla \varphi(\mathbf{x}) \in \mathcal{H}$ such that:

$$
D\varphi(\mathbf{x})(\mathbf{h}) = \langle \nabla \varphi(\mathbf{x}), \mathbf{h} \rangle \quad \forall \mathbf{h} \in \mathcal{H}
$$

Therefore, the Fréchet differentiability of $\varphi$ at $\mathbf{x}$ can also be written as:

$$
\varphi(\mathbf{x} + \mathbf{h}) = \varphi(\mathbf{x}) + \langle \nabla \varphi(\mathbf{x}), \mathbf{h} \rangle + o(\|\mathbf{h}\|)
$$

## Definition 7
For $L > 0$, a function $\Phi : \mathcal{H} \to \mathcal{H}$ is
* $L$-Lipschitz continuous if and only if

$$
\begin{equation}
\|\Phi (\mathbf{y}) - \Phi (\mathbf{x}) \| \le L \|\mathbf{y}-\mathbf{x}\| \quad \forall \mathbf{x}, \mathbf{y} \in \mathcal{H}
\end{equation}
$$

* $1/L$-cocoercive if and only if

$$
\langle \Phi(\mathbf{x}) -  \Phi(\mathbf{y}), \mathbf{x} - \mathbf{y}) \rangle \ge \frac{1}{L} \|  \Phi(\mathbf{x}) - \Phi(\mathbf{y}) \|^2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathcal{H}
$$

## Theorem 1 [Baillon–Haddad]
Let $\varphi : \mathcal{H} \to \mathbb{R}$ be a Fréchet differentiable function on $\mathcal{H}$.
* If $\nabla \varphi$ is $1/L$-cocoercive, then $\nabla \varphi$ is $L$-Lipschitz continuous. But the converse if false in general.
* If $\varphi$ is convex and $\nabla \varphi$ is $L$-Lipschitz continuous, then $\nabla \varphi$ is $1/L$-cocoercive.
### **Proof**
For a proof, you can check the paper "The Baillon-Haddad Theorem Revisited" by  Heinz H. Bauschke and  Patrick L. Combettes.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$
