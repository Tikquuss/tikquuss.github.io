---
title: "Eigenpairs from Matrix–Vector Products: Power Iteration, Deflation, Lanczos, and Arnoldi"
date: "2026-10-12"
category: "Mathematics"
tags:
  - Linear Algebra
  - Eigenvalues
  - Power Iteration
  - Lanczos Method
  - Arnoldi Method
  - Matrix-Free Methods
excerpt: "What power iteration, deflation, and the Lanczos method recover about a symmetric matrix from matrix–vector products alone: convergence, failure modes, exact termination, and residual certificates, with complete proofs, and what the Arnoldi method recovers about a nonsymmetric one."
---

This document rigorously studies what can be learned about the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="eigenvalues-eigenvectors" aria-label="Recall what eigenvalues and eigenvectors are">eigenpairs</button><span id="eigenvalues-eigenvectors" class="explanation-popover" popover="auto" role="note" aria-label="Eigenvalues and eigenvectors" data-label="Definition">A nonzero vector $\mathbf u$ is an <strong>eigenvector</strong> of $\mathbf A$, with <strong>eigenvalue</strong> $\lambda$, when <span class="explanation-popover__equation">$\mathbf A\mathbf u=\lambda\mathbf u.$</span> Multiplying $\mathbf u$ by $\mathbf A$ only rescales it, by the factor $\lambda$, and the pair $(\lambda,\mathbf u)$ is called an <strong>eigenpair</strong>. For example, $\mathbf A=\left[\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right]$ maps $[1,1]^\top$ to $3[1,1]^\top$ and $[1,-1]^\top$ to itself, so these two vectors are eigenvectors, with the eigenvalues $3$ and $1$. Rewriting the definition as $(\mathbf A-\lambda\,\mathbb I_d)\mathbf u=0$ shows that $\lambda$ is an eigenvalue exactly when the matrix $\mathbf A-\lambda\,\mathbb I_d$ is singular, that is, when $\lambda$ is a root of the polynomial <span class="explanation-popover__equation">$\lambda\mapsto\det(\lambda\,\mathbb I_d-\mathbf A),$</span> of degree $d$, called the <strong>characteristic polynomial</strong> of $\mathbf A$. The eigenvectors of $\lambda$ are then the nonzero vectors of the null space $\ker(\mathbf A-\lambda\,\mathbb I_d)$, called the <strong>eigenspace</strong> of $\lambda$. A real polynomial can have complex roots, so a real matrix can have complex eigenvalues. The rotation $\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$ turns every real vector by a quarter turn, so it has no real eigenvector, and its characteristic polynomial $\lambda^2+1$ has the roots $\pm i$. A real symmetric matrix, in contrast, has only real eigenvalues and an orthonormal basis of eigenvectors, by the [spectral theorem](#spectral-decomposition) used in Sections [1](#1-notation-and-preliminaries) to [4](#4-the-lanczos-method). Computing the characteristic polynomial requires every entry of $\mathbf A$, which the setting of this post does not provide. <span style="color:#2563eb">The definition itself, however, involves $\textcolor{#2563eb}{\mathbf A}$ only through the product $\textcolor{#2563eb}{\mathbf A\mathbf u}$.</span> One product is enough to test a candidate pair $(\lambda,\mathbf u)$, because the residual $\mathbf A\mathbf u-\lambda\mathbf u=(\mathbf A-\lambda\,\mathbb I_d)\mathbf u$ vanishes exactly when the pair is an eigenpair. The methods of this post use $\mathbf A$ in no other way. For a symmetric matrix, [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) and [Remark 11](#remark-11-any-scalar-and-the-best-one) also turn a small residual into a guarantee: if $\|\mathbf u\|=1$, then $\lambda$ is within $\|\mathbf A\mathbf u-\lambda\mathbf u\|$ of an eigenvalue.</span></span> of a real matrix $\mathbf{A}\in\mathbb{R}^{d\times d}$ that is known only through the map $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$. Storing $\mathbf{A}$ takes $d^2$ numbers, and forming it can cost much more. In many applications, however, the product $\mathbf{A}\mathbf{x}$ is cheap, because it follows directly from how $\mathbf{A}$ is defined, whether $\mathbf{A}$ is built from a data matrix, a sparse graph or a discretized differential operator, or is the Hessian of a function whose gradient can be computed ([Section 2](#2-matrices-known-only-through-their-products) gives more examples). Two families of methods need nothing more than such products: **[power iteration](#3-power-iteration)** ([Section 3](#3-power-iteration)), which extracts one eigenpair at a time and reaches further eigenpairs through [shifts](#33-corollary-1-recovering-algebraic-extrema-by-shifting) and [deflation](#34-lemma-1-orthogonal-deflation), and the **[Lanczos method](#4-the-lanczos-method)** ([Section 4](#4-the-lanczos-method)), which builds an orthonormal basis of a [Krylov subspace](#41-definition-1-krylov-subspace) and solves a small tridiagonal eigenvalue problem inside it. Sections [1](#1-notation-and-preliminaries) to [4](#4-the-lanczos-method) assume that $\mathbf{A}$ is symmetric, as Hessians, covariance matrices and the adjacency matrices of undirected graphs are. The **[Arnoldi method](#5-the-arnoldi-method)** ([Section 5](#5-the-arnoldi-method)) extends the Lanczos method to nonsymmetric matrices, such as the transition matrix of a random walk on a directed graph.

The results assume neither positive definiteness nor distinct eigenvalues. Each statement says precisely which eigenpairs are recovered, and the remarks say which are not. All the delicate cases are treated explicitly: a dominant eigenvalue of either sign, opposite eigenvalues of equal magnitude, repeated eigenvalues, and zero eigenvalues. Sections [3](#3-power-iteration), [4](#4-the-lanczos-method) and [5](#5-the-arnoldi-method) each end with an implementable procedure: [Algorithm 1](#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation) for power iteration, [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks) for the Lanczos method and [Algorithm 3](#57-algorithm-3-arnoldi-with-reorthogonalization-and-residual-checks) for the Arnoldi method. All three stop when a residual computed with the original matrix falls below a tolerance. The figures run these procedures on matrices with prescribed spectra and on five matrices from applications: the adjacency matrix of a graph, a covariance matrix and the Hessian of a neural network ([Section 2](#2-matrices-known-only-through-their-products)), and the transition matrix of a random walk on a directed graph and a discretized convection–diffusion operator ([Section 5.1](#51-nonsymmetric-matrices-known-through-their-products)). The code that produces them, with implementations of the three algorithms, is [on GitHub](https://github.com/Tikquuss/tikquuss.github.io/tree/master/agent/scripts/eigenpairs-from-matrix-vector-products).

For a symmetric $\mathbf{A}$, the central object is the decomposition of a vector along the eigenspaces of $\mathbf{A}$. Let $\Lambda(\mathbf{A})$ be the set of distinct eigenvalues of $\mathbf{A}$, and let $\Pi_\lambda$ be the orthogonal projection onto the eigenspace of $\lambda\in\Lambda(\mathbf{A})$. Then, for every vector $\mathbf{v}\in\mathbb{R}^d$ and every integer $k\ge0$,

$$
\mathbf{v}=\sum_{\lambda\in\Lambda(\mathbf{A})}\Pi_\lambda\mathbf{v},
\qquad
\mathbf{A}^k\mathbf{v}=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^k\,\Pi_\lambda\mathbf{v}.
$$

Started from $\mathbf{v}$, power iteration keeps the last vector of the sequence $\mathbf{v},\mathbf{A}\mathbf{v},\mathbf{A}^2\mathbf{v},\dots$ and lets the factors $\lambda^k$ select the eigenvalue of largest magnitude. The Lanczos method keeps the whole sequence and optimizes over its span. Every proof of Sections [3](#3-power-iteration) and [4](#4-the-lanczos-method) follows the components $\Pi_\lambda\mathbf{v}$ through these operations, and [Section 5](#5-the-arnoldi-method) shows what changes when the eigenspaces are no longer orthogonal.

## 1. Notation and preliminaries

### Notation

* $\|\cdot\|$ is the Euclidean norm, $\mathbb{I}_d$ the $d\times d$ identity matrix, $\delta_{a,b}$ the Kronecker delta, equal to $1$ if $a=b$ and to $0$ otherwise, for $a$ and $b$ in any set (we write $\delta_{ij}$, without the comma, when $i$ and $j$ are integers), and $\mathbf{e}_m=[0,\ldots,0,1]^\top\in\mathbb{R}^m$ the last standard basis vector of $\mathbb{R}^m$.
* For a subspace $S\subseteq\mathbb{R}^d$, $S^\perp=\{\mathbf{x}\in\mathbb{R}^d:\mathbf{x}^\top\mathbf{s}=0 \ \forall \mathbf{s}\in S\}$ is its orthogonal complement.
* For a matrix $\mathbf{M}$ with $n$ columns, $\ker(\mathbf{M})=\{\mathbf{x}\in\mathbb{R}^n:\mathbf{M}\mathbf{x}=0\}$ is its null space and $\operatorname{range}(\mathbf{M})=\{\mathbf{M}\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}$ is its column space. 

* In Sections [1](#1-notation-and-preliminaries) to [4](#4-the-lanczos-method), $\mathbf{A}\in\mathbb{R}^{d\times d}$ is symmetric, and [Section 5](#5-the-arnoldi-method) drops this assumption. We let $\Lambda(\mathbf{A})$ be the set of its **distinct** eigenvalues, so that a repeated eigenvalue appears once in $\Lambda(\mathbf{A})$, and we write $\lambda_{\min}(\mathbf{A})=\min\Lambda(\mathbf{A})$ and $\lambda_{\max}(\mathbf{A})=\max\Lambda(\mathbf{A})$.
For each $\lambda\in\Lambda(\mathbf{A})$, $E_\lambda=\ker(\mathbf{A}-\lambda\mathbb{I}_d)$ is the eigenspace of $\lambda$, and $\Pi_\lambda$ is the orthogonal projection onto $E_\lambda$. If the columns $\mathbf{u}_1,\dots,\mathbf{u}_{m_\lambda}$ of $\mathbf{U}_\lambda\in\mathbb{R}^{d\times m_\lambda}$ form an orthonormal basis of $E_\lambda$, where $m_\lambda$ is the multiplicity of $\lambda$, then $\Pi_\lambda=\mathbf{U}_\lambda\mathbf{U}_\lambda^\top=\sum_{i=1}^{m_\lambda}\mathbf{u}_i\mathbf{u}_i^\top$, that is, $\Pi_\lambda\mathbf{x}=\sum_{i=1}^{m_\lambda}(\mathbf{u}_i^\top\mathbf{x})\,\mathbf{u}_i$ keeps the components of $\mathbf{x}$ along these basis vectors. An expression in terms of $\mathbf{A}$ alone is given below.

* For vectors $\mathbf{u}_1,\dots,\mathbf{u}_m\in\mathbb{R}^d$, $[\mathbf{u}_1,\dots,\mathbf{u}_m]\in\mathbb{R}^{d\times m}$ is the matrix with these columns, and $\operatorname{span}\{\mathbf{u}_1,\dots,\mathbf{u}_m\}$ is the set of their linear combinations
* $\operatorname{diag}(c_1,\dots,c_m)$ is the diagonal matrix with diagonal entries $c_1,\dots,c_m \in \mathbb{R}$. 


* $\mathcal{N}(0,\mathbb{I}_d)$ is the standard Gaussian distribution on $\mathbb{R}^d$, and $\operatorname{sgn}(t)\in\{-1,1\}$ is the sign of a real number $t\neq0$.

### Spectral decomposition

When $\mathbf{A}$ is symmetric, the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="spectral-theorem" aria-label="State the spectral theorem used here">spectral theorem</button><span id="spectral-theorem" class="explanation-popover" popover="auto" role="note" aria-label="Spectral theorem" data-label="Theorem">Every real symmetric matrix has an orthonormal basis of eigenvectors. Collecting the basis vectors that share the eigenvalue $\lambda$ as the columns of a matrix $\mathbf U_\lambda$ gives $\Pi_\lambda=\mathbf U_\lambda\mathbf U_\lambda^\top$, and the three identities follow from the orthonormality of the basis.</span></span> gives

$$
\mathbf{A}=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda\,\Pi_\lambda,
\qquad
\mathbb{I}_d=\sum_{\lambda\in\Lambda(\mathbf{A})}\Pi_\lambda,
\qquad
\Pi_\lambda\Pi_\mu=0
\quad\text{when }\lambda\neq\mu .
$$

The powers of $\mathbf{A}$ act on each eigenspace separately, and so does every polynomial:
for every integer $k\ge0$ and every polynomial $p$,
$$
\mathbf{A}^k=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^k\,\Pi_\lambda,
\qquad
p(\mathbf{A})=\sum_{\lambda\in\Lambda(\mathbf{A})}p(\lambda)\,\Pi_\lambda .
$$

In fact, each column $\mathbf{u}_i$ of $\mathbf{U}_\lambda$ is an eigenvector, $\mathbf{A}\mathbf{u}_i=\lambda\mathbf{u}_i$, so multiplying $\Pi_\lambda=\sum_{i=1}^{m_\lambda} \mathbf{u}_i\mathbf{u}_i^\top$ by $\mathbf{A}$ gives $\mathbf{A}\,\Pi_\lambda=\sum_{i=1}^{m_\lambda}(\mathbf{A}\mathbf{u}_i)\,\mathbf{u}_i^\top=\lambda\,\Pi_\lambda.$
The spectral theorem gives the same identity, since $\mathbf{A}\,\Pi_\lambda=\sum_{\mu}\mu\,\Pi_\mu\Pi_\lambda=\lambda\,\Pi_\lambda^2$ and $\Pi_\lambda^2=\Pi_\lambda$. Applying it $k$ times gives $\mathbf{A}^k\Pi_\lambda=\lambda^k\Pi_\lambda$, and multiplying $\mathbb{I}_d=\sum_\lambda\Pi_\lambda$ by $\mathbf{A}^k$ then gives the first identity above. 
For a polynomial $p(t)=\sum_{k=0}^n c_k t^k$, we have $p(\mathbf{A})=\sum_{k=0}^nc_k\mathbf{A}^k=\sum_\lambda\bigl(\sum_{k=0}^nc_k\lambda^k\bigr)\Pi_\lambda=\sum_\lambda p(\lambda)\Pi_\lambda$, which is the second identity. 

Taking for $p$ the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="lagrange-projection" aria-label="Explain the Lagrange polynomial and why it gives the projection">Lagrange polynomial</button><span id="lagrange-projection" class="explanation-popover" popover="auto" role="note" aria-label="Lagrange polynomial" data-label="Definition">Given finitely many distinct numbers, the Lagrange polynomial attached to one of them equals $1$ at that number and $0$ at all the others. For $\lambda\in\Lambda(\mathbf A)$ it is <span class="explanation-popover__equation">$\displaystyle\ell_\lambda(t)=\prod_{\mu\in\Lambda(\mathbf A),\,\mu\neq\lambda}\frac{t-\mu}{\lambda-\mu}.$</span> Each factor vanishes at its own $\mu$ and equals $1$ at $t=\lambda$, so $\ell_\lambda(\lambda)=1$ and $\ell_\lambda(\mu)=0$ for every other $\mu\in\Lambda(\mathbf A)$. Its degree is $|\Lambda(\mathbf A)|-1$. Substituting it for $p$ in $p(\mathbf A)=\sum_{\nu}p(\nu)\,\Pi_\nu$ keeps only the term $\nu=\lambda$, so <span class="explanation-popover__equation">$\ell_\lambda(\mathbf A)=\Pi_\lambda.$</span> For example, if $\Lambda(\mathbf A)=\{1,3\}$, then $\ell_1(t)=(3-t)/2$ and $\Pi_1=(3\,\mathbb I_d-\mathbf A)/2$.</span></span> that equals $1$ at $\lambda$ and $0$ at the other distinct eigenvalues expresses each projection through $\mathbf{A}$ alone:

$$
\Pi_\lambda=\prod_{\substack{\mu\in\Lambda(\mathbf{A})\\\mu\neq\lambda}}\frac{\mathbf{A}-\mu\,\mathbb{I}_d}{\lambda-\mu},
$$

where the empty product, when $\mathbf{A}$ has a single distinct eigenvalue, equals $\mathbb{I}_d$.

These projections are used in the analysis. The algorithms below do not require computing them.

## 2. Matrices known only through their products

The results below use $\mathbf{A}$ only through the products $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$, so they hold whatever routine computes those products. The following cases are typical.

### 2.1 Hessians

In many problems, $\mathbf{A}=\nabla_\theta^2f(\theta)$ is the Hessian of a function $f:\mathbb{R}^d\to\mathbb{R}$ that is twice continuously differentiable in a neighborhood of a point $\theta$. Examples include the following.

* **Training losses.** The training loss of a (deep) neural network with parameters $\theta$ is $f(\theta)=\frac1n\sum_{i=1}^n \ell\bigl(\mathbf{z}_i(\theta),\mathbf{y}_i\bigr)$, where $\mathbf{z}_i(\theta)$ is the output of the network on the $i$-th of $n$ training inputs, $\mathbf{y}_i$ is the target for that input, and $\ell$ is a cost function such as $\ell(\mathbf{z},\mathbf{y})=\frac12\|\mathbf{z}-\mathbf{y}\|^2$. Its curvature is studied to understand optimization (how training progresses) and generalization (how the network performs on new inputs).
* **Likelihoods.** The negative log-likelihood of observations $\mathbf{y}_1,\dots,\mathbf{y}_n$ under a statistical model with density $p(\cdot\,;\theta)$ is $f(\theta)=-\sum_{i=1}^n\log p(\mathbf{y}_i;\theta)$. Its minimizer $\hat\theta$ is the maximum-likelihood estimate, and the Hessian $\nabla^2f(\hat\theta)$ is the observed Fisher information. For large samples, its inverse approximates the covariance of $\hat\theta$, that is, how precisely the data determine the parameters.
* **Newton-type optimization.** To minimize $f(\theta)$, <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="newton-step" aria-label="Explain where the Newton step comes from">Newton's method</button><span id="newton-step" class="explanation-popover" popover="auto" role="note" aria-label="The Newton step" data-label="Derivation">Near $\theta$, Newton's method replaces $f$ by its second-order Taylor expansion, a quadratic function of the step: <span class="explanation-popover__equation">$f(\theta+\Delta\theta)\approx f(\theta)+\nabla f(\theta)^\top\Delta\theta+\tfrac12\,\Delta\theta^\top\nabla^2f(\theta)\,\Delta\theta.$</span> The gradient of this model with respect to $\Delta\theta$ is $\nabla f(\theta)+\nabla^2f(\theta)\,\Delta\theta$, and setting it to zero gives the linear system $\nabla^2f(\theta)\,\Delta\theta=-\nabla f(\theta)$. When $\nabla^2f(\theta)$ is positive definite, the model is a bowl and $\Delta\theta$ goes to its bottom. Otherwise, the model has no minimum, and practical methods modify the step. In one dimension, the step $\Delta\theta=-f'(\theta)/f''(\theta)$ moves to the point where the tangent line of $f'$ at $\theta$ crosses zero, as Newton's method does to solve $f'=0$. Write the Hessian in an orthonormal eigenbasis, $\nabla^2f(\theta)=\sum_{j=1}^d\lambda_j\mathbf{u}_j\mathbf{u}_j^\top$, with every $\lambda_j>0$. Since $\mathbf{u}_i^\top\mathbf{u}_j=\delta_{ij}$, multiplying the sum on the left by $\mathbf{u}_i^\top$ gives $\mathbf{u}_i^\top\nabla^2f(\theta)=\lambda_i\mathbf{u}_i^\top$. Multiplying the linear system on the left by $\mathbf{u}_i^\top$ therefore gives $\lambda_i\,\mathbf{u}_i^\top\Delta\theta=-\mathbf{u}_i^\top\nabla f(\theta)$, that is, $\mathbf{u}_i^\top\Delta\theta=- \mathbf{u}_i^\top\nabla f(\theta)/\lambda_i.$ Expanding the step in the same basis, $\Delta\theta=\sum_{i=1}^d(\mathbf{u}_i^\top\Delta\theta)\,\mathbf{u}_i$, and substituting each coordinate gives <span class="explanation-popover__equation">$\displaystyle\Delta\theta=-\sum_{i=1}^d\frac{\mathbf{u}_i^\top\nabla f(\theta)}{\lambda_i}\,\mathbf{u}_i.$</span> Each component of the gradient is divided by the curvature $\lambda_i$ along $\mathbf{u}_i$, so the step is short where $f$ is steep and long where $f$ is flat. Gradient descent instead takes the step <span class="explanation-popover__equation">$\displaystyle\Delta\theta=-\alpha\nabla f(\theta)=-\sum_{i=1}^d\alpha\,\bigl(\mathbf{u}_i^\top\nabla f(\theta)\bigr)\,\mathbf{u}_i,$</span> which multiplies every coordinate by the same factor $\alpha$.</span></span> takes the step $\Delta\theta$ that solves the linear system $\nabla^2f(\theta)\,\Delta\theta=-\nabla f(\theta)$. Solving it by forming and factorizing the $d\times d$ Hessian would take $d^2$ numbers of storage and about $d^3$ operations. Truncated Newton methods ([Dembo and Steihaug, 1983](#ref-dembo1983)) instead run a few iterations of the conjugate gradient method ([Hestenes and Stiefel, 1952](#ref-hestenes1952)), which uses the Hessian only through products $\nabla^2f(\theta)\,\mathbf{x}$, computed as explained below, and stop once the step is accurate enough. The conjugate gradient method builds the same Krylov subspaces as the [Lanczos method](#4-the-lanczos-method).
* **Molecular vibrations.** The potential energy $f(\theta)$ of a molecule depends on the positions $\theta\in\mathbb{R}^{3N}$ of its $N$ atoms. Near an equilibrium $\theta^\star$, where $\nabla f(\theta^\star)=0$, small displacements obey <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="newton-law-of-motion" aria-label="Explain why small displacements obey this equation">Newton's law of motion</button><span id="newton-law-of-motion" class="explanation-popover" popover="auto" role="note" aria-label="Explain why small displacements obey this equation" data-label="Physics">Newton's second law says that mass times acceleration equals force, and the force on the atoms is minus the gradient of the potential energy, $-\nabla f(\theta)$: the atoms are pushed downhill. Near $\theta^\star$, the first-order Taylor expansion of the gradient is <span class="explanation-popover__equation">$\nabla f(\theta)\approx\nabla f(\theta^\star)+\nabla^2f(\theta^\star)(\theta-\theta^\star)=\nabla^2f(\theta^\star)(\theta-\theta^\star),$</span> because $\nabla f(\theta^\star)=0$ at an equilibrium. The force is therefore a linear restoring force, as for a system of springs, and $\mathbf M\ddot\theta=-\nabla f(\theta)$ becomes the stated equation, where $\ddot\theta$ is the acceleration, the second time derivative of the positions.</span></span> $\mathbf{M}\ddot\theta=-\nabla^2f(\theta^\star)(\theta-\theta^\star)$, where $\mathbf{M}$ is the diagonal matrix of atomic masses, each repeated for the three coordinates of its atom. The eigenvectors of the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="normal-modes" aria-label="Explain why its eigenpairs are the normal modes and squared frequencies">mass-weighted Hessian</button><span id="normal-modes" class="explanation-popover" popover="auto" role="note" aria-label="Explain why its eigenpairs are the normal modes and squared frequencies" data-label="Physics">In the mass-weighted coordinates $\mathbf x=\mathbf M^{1/2}(\theta-\theta^\star)$, substituting $\theta-\theta^\star=\mathbf M^{-1/2}\mathbf x$ into the equation of motion and multiplying it on the left by $\mathbf M^{-1/2}$ gives <span class="explanation-popover__equation">$\ddot{\mathbf x}=-\mathbf K\mathbf x,\qquad\mathbf K=\mathbf M^{-1/2}\nabla^2f(\theta^\star)\mathbf M^{-1/2}.$</span> A motion in which every coordinate oscillates at the same frequency $\omega$ in a fixed pattern $\mathbf v$, $\mathbf x(t)=\cos(\omega t)\,\mathbf v$, has $\ddot{\mathbf x}=-\omega^2\mathbf x$, so it solves the equation exactly when $\mathbf K\mathbf v=\omega^2\mathbf v$. Each eigenvector of $\mathbf K$ is thus a vibration pattern, a normal mode, and its eigenvalue is the squared frequency of that mode. Because $\mathbf K$ is symmetric, its orthonormal eigenvectors split every small motion into independent normal modes, each vibrating at its own frequency, and the weighting by $\mathbf M^{-1/2}$ on both sides is what keeps the matrix symmetric. At a minimum of $f$, the Hessian is positive semidefinite, and so is $\mathbf K$, since $\mathbf x^\top\mathbf K\mathbf x=(\mathbf M^{-1/2}\mathbf x)^\top\nabla^2f(\theta^\star)(\mathbf M^{-1/2}\mathbf x)\ge0$. Its eigenvalues are therefore nonnegative, and each one is the square of a real frequency $\omega$. At a saddle point of $f$, an eigenvalue $-\mu<0$ would have no real square root, and the displacement along its mode would grow like $e^{\sqrt{\mu}\,t}$ instead of oscillating. For example, take two atoms of masses $m_1$ and $m_2$ at positions $\theta_1$ and $\theta_2$ on a line, joined by a spring of stiffness $k$ and rest length $a$. The energy of the spring is $f(\theta)=\tfrac k2(\theta_2-\theta_1-a)^2$, and <span class="explanation-popover__equation">$\nabla^2f(\theta^\star)=k\begin{bmatrix}1&-1\\-1&1\end{bmatrix},\qquad\mathbf K=k\begin{bmatrix}1/m_1&-1/\sqrt{m_1m_2}\\-1/\sqrt{m_1m_2}&1/m_2\end{bmatrix}.$</span> The eigenvalues of $\mathbf K$ are $0$, whose mode moves both atoms together and stretches nothing, and $k(1/m_1+1/m_2)$, the trace of $\mathbf K$. The second mode moves the atoms in opposite directions, the heavier one less, and it is the stretching vibration, with frequency $\omega=\sqrt{k(1/m_1+1/m_2)}$.</span></span> $\mathbf{M}^{-1/2}\nabla^2f(\theta^\star)\mathbf{M}^{-1/2}$ are the vibration patterns of the molecule (its normal modes), and its eigenvalues are the squared vibration frequencies $\omega^2$. A product with this matrix is a Hessian–vector product between two diagonal scalings, $\mathbf{M}^{-1/2}\bigl(\nabla^2f(\theta^\star)(\mathbf{M}^{-1/2}\mathbf{x})\bigr)$, so the lowest-frequency modes of a large molecule, its slowest collective motions, can be computed without forming the $3N\times3N$ Hessian.

Such a Hessian is symmetric by <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="schwarz-theorem" aria-label="Explain why the Hessian is symmetric">Schwarz's theorem</button><span id="schwarz-theorem" class="explanation-popover" popover="auto" role="note" aria-label="Schwarz's theorem" data-label="Theorem">When the second partial derivatives of $f$ are continuous near $\theta$, they do not depend on the order of differentiation: $\partial^2f/\partial\theta_i\partial\theta_j=\partial^2f/\partial\theta_j\partial\theta_i$. The Hessian is therefore a real symmetric matrix.</span></span>. In that case, each product can be evaluated as
$\mathbf{A}\mathbf{x}=\nabla_\theta\bigl(\nabla_\theta f(\theta)^\top\mathbf{x}\bigr),$
with $\mathbf{x}$ held constant during differentiation. The scalar $\nabla_\theta f(\theta)^\top\mathbf{x}$ is the directional derivative of $f$ along $\mathbf{x}$. Its gradient with respect to $\theta$ is $\nabla_\theta^2f(\theta)^\top\mathbf{x}$, which equals $\mathbf{A}\mathbf{x}$ because the Hessian is symmetric. Automatic differentiation therefore returns the product at a <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="hvp-cost" aria-label="Explain the cost of a Hessian-vector product">small multiple of the cost of a gradient</button><span id="hvp-cost" class="explanation-popover" popover="auto" role="note" aria-label="Cost of a Hessian-vector product" data-label="Fact">The product is obtained by differentiating the gradient computation once more, in forward or reverse mode. Each pass costs a small constant multiple of the original computation, so the product costs about as much as a few gradient evaluations.</span></span> ([Pearlmutter, 1994](#ref-pearlmutter1994)), and the $d\times d$ matrix is never formed.

The point $\theta$ and everything that defines $f$ must remain fixed during an eigensolver run. For a training loss, evaluating it on a different random subset of the data (a minibatch) between two products changes the matrix being analyzed, and the statements below then no longer describe the computation.


Every symmetric matrix $\mathbf{A}$ is in fact a Hessian: the quadratic $\mathbf{x}\mapsto\frac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ has the Hessian $\mathbf{A}$ at every point. This does not help to compute products, since the gradient of that quadratic is $\mathbf{A}\mathbf{x}$ itself. The Hessian route pays off when $f$ is given by a program whose gradient is cheap, so that curvature is obtained from first-order code. For the matrices of the following section, the products come instead from the structure that defines $\mathbf{A}$.

### 2.2 Other matrices known through their products

Products are just as cheap in many cases where $\mathbf{A}$ is not presented as a Hessian.

* **Gram and covariance matrices.** For a data matrix $\mathbf{X}\in\mathbb{R}^{n\times d}$, whose $n$ rows are observations of $d$ variables, the product $\mathbf{X}^\top\mathbf{X}\mathbf{x}=\mathbf{X}^\top(\mathbf{X}\mathbf{x})$ <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="gram-product-cost" aria-label="Explain the cost of the product and of forming the matrix">costs $O(nd)$ operations</button><span id="gram-product-cost" class="explanation-popover" popover="auto" role="note" aria-label="Cost of a product with a Gram matrix" data-label="Cost">The parentheses fix the order: two matrix–vector products, and $\mathbf X^\top\mathbf X$ is never formed. First, each of the $n$ entries of $\mathbf y=\mathbf X\mathbf x$ is the inner product of a row of $\mathbf X$ with $\mathbf x$, about $d$ multiplications and additions. Then each of the $d$ entries of $\mathbf X^\top\mathbf y$ is the inner product of a column of $\mathbf X$ with $\mathbf y$, about $n$ of them. Each pass costs about $nd$ operations, and only $\mathbf y$ is stored in between. Forming $\mathbf X^\top\mathbf X$ instead takes $d^2$ inner products of columns of length $n$, about $nd^2$ operations, and $d^2$ numbers of storage. For $n=10^6$ observations of $d=10^4$ variables, one product costs about $2\cdot10^{10}$ operations, whereas forming $\mathbf X^\top\mathbf X$ costs about $10^{14}$ operations and $10^8$ numbers of storage. If $\mathbf X$ is sparse, a product costs on the order of its number of nonzero entries.</span></span>, whereas forming $\mathbf{X}^\top\mathbf{X}$ costs $O(nd^2)$ and storing it costs $O(d^2)$. When each column of $\mathbf{X}$ has zero mean, the sample covariance matrix of the $d$ variables is the average of the outer products of the observations, $\mathbf{X}^\top\mathbf{X}/n=(1/n)\sum_{i=1}^n\mathbf{X}_i\mathbf{X}_i^\top$, where $\mathbf{X}_i\in\mathbb{R}^d$ is the $i$-th row of $\mathbf{X}$. Its leading eigenvector, the first principal component, is the direction of largest variance in the data: it maximizes the variance $\frac1n\|\mathbf{X}\mathbf{u}\|^2$ of the data projected on $\mathbf{u}$ over unit vectors $\mathbf{u}$.
* **Gauss–Newton matrices.** Many objectives are compositions $f(\theta)=\psi\bigl(F(\theta)\bigr)$ of a map $F:\mathbb{R}^d\to\mathbb{R}^p$ and a function $\psi:\mathbb{R}^p\to\mathbb{R}$, both twice differentiable. For the training loss of [Section 2.1](#21-hessians), $F(\theta)$ stacks the outputs $\mathbf{z}_1(\theta),\dots,\mathbf{z}_n(\theta)$ into one vector of $\mathbb{R}^p$, and $\psi(\mathbf{z})=\frac1n\sum_{i=1}^n\ell(\mathbf{z}_i,\mathbf{y}_i)$ for $\mathbf{z}=[\mathbf{z}_1;\dots;\mathbf{z}_n]$. Let $\mathbf{J}\in\mathbb{R}^{p\times d}$ be the Jacobian of $F$ at $\theta$, with entries $J_{kj}=\partial F_k(\theta)/\partial\theta_j$, and let $\mathbf{H}=\nabla^2\psi\bigl(F(\theta)\bigr)\in\mathbb{R}^{p\times p}$. The chain rule gives $\nabla^2f(\theta)=\mathbf{J}^\top\mathbf{H}\mathbf{J}+\sum_{k=1}^p\partial_k\psi\bigl(F(\theta)\bigr)\,\nabla^2F_k(\theta)$, where $\partial_k\psi$ is the partial derivative of $\psi$ with respect to its $k$-th argument. The generalized Gauss–Newton matrix $\mathbf{G}=\mathbf{J}^\top\mathbf{H}\mathbf{J}$ keeps the first term. It is symmetric, and positive semidefinite when $\psi$ is convex. It is applied as $\mathbf{G}\mathbf{x}=\mathbf{J}^\top\bigl(\mathbf{H}(\mathbf{J}\mathbf{x})\bigr)$, where automatic differentiation computes $\mathbf{J}\mathbf{x}$ in forward mode and $\mathbf{J}^\top\mathbf{u}$ in reverse mode without forming $\mathbf{J}$. For least squares, $\psi(\mathbf{z})=\frac12\|\mathbf{z}-\mathbf{y}\|^2$ for a target $\mathbf{y}\in\mathbb{R}^p$, so $\mathbf{H}=\mathbb{I}_p$ and $\mathbf{G}=\mathbf{J}^\top\mathbf{J}$. When $F$ is linear, $F(\theta)=\mathbf{J}\theta$, every $\nabla^2F_k$ vanishes and $\mathbf{G}$ is the exact Hessian. When $F$ is nonlinear, $\mathbf{G}=\mathbf{J}^\top\mathbf{J}$ is the classical Gauss–Newton approximation of the Hessian. For the squared loss and the softmax cross-entropy of a neural network, $\mathbf{G}$ is also the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="fisher-gauss-newton" aria-label="Explain why the Gauss-Newton matrix equals the Fisher information">Fisher information matrix</button><span id="fisher-gauss-newton" class="explanation-popover" popover="auto" role="note" aria-label="Fisher information and Gauss-Newton" data-label="Fact">Suppose the model gives each target a probability density $p(\mathbf y\,|\,\mathbf z)$ that depends on the output $\mathbf z=F(\theta)$, and that $\ell$ is the corresponding negative log-likelihood. The Fisher information matrix is <span class="explanation-popover__equation">$\mathbb E_{\mathbf y\sim p(\cdot\,|\,F(\theta))}\bigl[\nabla_\theta\log p(\mathbf y\,|\,F(\theta))\,\nabla_\theta\log p(\mathbf y\,|\,F(\theta))^\top\bigr].$</span> By the chain rule it equals $\mathbf J^\top\mathbf I_{\mathbf z}\mathbf J$ at $\mathbf z=F(\theta)$, where <span class="explanation-popover__equation">$\mathbf I_{\mathbf z}=\mathbb E_{\mathbf y\sim p(\cdot\,|\,\mathbf z)}\bigl[\nabla_{\mathbf z}\log p(\mathbf y\,|\,\mathbf z)\,\nabla_{\mathbf z}\log p(\mathbf y\,|\,\mathbf z)^\top\bigr]=\mathbb E_{\mathbf y\sim p(\cdot\,|\,\mathbf z)}\bigl[-\nabla_{\mathbf z}^2\log p(\mathbf y\,|\,\mathbf z)\bigr].$</span> When $-\nabla_{\mathbf z}^2\log p(\mathbf y\,|\,\mathbf z)$ does not depend on $\mathbf y$, the last expectation is $\mathbf H$, so the Fisher matrix is $\mathbf G$. This holds for a Gaussian with mean $\mathbf z$ (the squared loss, $\mathbf H=\mathbb I$) and for a categorical distribution with probabilities $s_k=e^{z_k}/\sum_je^{z_j}$ (the softmax cross-entropy, $\mathbf H=\operatorname{diag}(\mathbf s)-\mathbf s\mathbf s^\top$).</span></span> of the model.
* **Graphs.** For an undirected graph on $d$ nodes, the adjacency matrix $\mathbf{W}$ has the entries $W_{ij}=W_{ji}\ge0$, the weight of the edge between nodes $i$ and $j$, with $W_{ij}=0$ when there is no edge. It is symmetric, and so is the Laplacian $\mathbf{L}=\operatorname{diag}(\mathbf{W}\mathbf{1})-\mathbf{W}$, where $\mathbf{1}$ is the all-ones vector. A product with either matrix costs one pass over the edges. Since $\mathbf{x}^\top\mathbf{L}\mathbf{x}=\sum_{i<j}W_{ij}(x_i-x_j)^2$, the eigenvectors of $\mathbf{L}$ with the smallest eigenvalues vary little across heavily weighted edges. They are used to split the graph into well-connected groups, in spectral clustering and graph partitioning ([von Luxburg, 2007](#ref-vonluxburg2007)).
* **Discretized differential operators.** Finite-difference and finite-element discretizations of self-adjoint differential operators, such as those of heat conduction, elasticity or quantum mechanics, are applied point by point or element by element, in time proportional to the number of unknowns and without assembling the matrix. For instance, the operator $u\mapsto-u''$ on $[0,1]$ with $u(0)=u(1)=0$, discretized on the grid $t_i=ih$ with $h=1/(d+1)$, becomes the matrix defined by $(\mathbf{A}\mathbf{x})_i=(2x_i-x_{i-1}-x_{i+1})/h^2$ for $i\in\{1,\dots,d\}$, with $x_0=x_{d+1}=0$. [Lanczos (1950)](#ref-lanczos1950) designed his method for linear differential and integral operators.
* **Structured matrices.** A symmetric Toeplitz matrix is constant along each diagonal, $A_{ij}=a_{|i-j|}$ for numbers $a_0,\dots,a_{d-1}$, as is the covariance matrix of a stationary time series. It is applied in $O(d\log(d))$ operations by embedding it in a circulant matrix of size $2d$, whose rows are cyclic shifts of one another and which the fast Fourier transform ([Cooley and Tukey, 1965](#ref-cooley1965)) diagonalizes. Operators that act separately along the two axes of a grid are Kronecker products. A Kronecker product of symmetric matrices $\mathbf{M}_1\in\mathbb{R}^{d_1\times d_1}$ and $\mathbf{M}_2\in\mathbb{R}^{d_2\times d_2}$ is applied without being formed, through $(\mathbf{M}_1\otimes\mathbf{M}_2)\operatorname{vec}(\mathbf{Y})=\operatorname{vec}(\mathbf{M}_2\mathbf{Y}\mathbf{M}_1^\top)$ for $\mathbf{Y}\in\mathbb{R}^{d_2\times d_1}$, where $\operatorname{vec}(\cdot)$ stacks the columns of a matrix.
* **Shifted inverses.** If linear systems with $\mathbf{A}-\sigma\mathbb{I}_d$ can be solved for a shift $\sigma$ that is not an eigenvalue, for instance with a sparse factorization computed once, then $\mathbf{x}\mapsto(\mathbf{A}-\sigma\mathbb{I}_d)^{-1}\mathbf{x}$ is a product with a symmetric matrix that has the same eigenvectors as $\mathbf{A}$ and the eigenvalues $1/(\lambda-\sigma)$. Power iteration and the Lanczos method applied to it find the eigenvalues of $\mathbf{A}$ closest to $\sigma$. This is the shift-and-invert strategy.

All these matrices are symmetric, and the results of Sections [3](#3-power-iteration) and [4](#4-the-lanczos-method) rely on symmetry through the spectral theorem. For a nonsymmetric matrix accessed through products, such as the transition matrix of a random walk on a directed graph, whose entry $(i,j)$ is the probability of stepping from node $i$ to node $j$, the Arnoldi method ([Arnoldi, 1951](#ref-arnoldi1951)) replaces the Lanczos method. [Section 5](#5-the-arnoldi-method) presents it, together with nonsymmetric examples and what power iteration retains without symmetry.


## 3. Power iteration

In this part, we first establish the convergence conditions for ordinary power iteration, then describe how shifting and deflation recover additional eigenpairs. Starting from a unit vector $\mathbf{v}_0$, power iteration alternates a product with $\mathbf{A}$ and a normalization:

$$
\mathbf{v}_{k+1}=\frac{\mathbf{A}\mathbf{v}_k}{\|\mathbf{A}\mathbf{v}_k\|},
\qquad k\ge0 .
$$

If $\mathbf{A}\mathbf{v}_0=0$, then $\mathbf{v}_0$ is already an eigenvector with eigenvalue zero. In this case, the algorithm should return $(0,\mathbf{v}_0)$ without attempting to normalize $\mathbf{A}\mathbf{v}_0$.

The numerical analyst Alston S. Householder attributed the first treatment of the method to [Müntz (1913)](#ref-muntz1913) in his book [*The Theory of Matrices in Numerical Analysis*](#ref-householder1964) (1964), as recalled in the historical survey of [Golub and van der Vorst (2000)](#ref-golub2000). Deflation schemes built on it were analyzed early on, for example by [Osborne (1958)](#ref-osborne1958). The eigenvalue estimate of power iteration is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="rayleigh-quotient" aria-label="Define the Rayleigh quotient">Rayleigh quotient</button><span id="rayleigh-quotient" class="explanation-popover" popover="auto" role="note" aria-label="Rayleigh quotient" data-label="Definition">For a unit vector $\mathbf v$, the Rayleigh quotient $\mathbf v^\top\mathbf A\mathbf v$ is the curvature of $\mathbf x\mapsto\frac12\mathbf x^\top\mathbf A\mathbf x$ along $\mathbf v$. It lies between $\lambda_{\min}(\mathbf A)$ and $\lambda_{\max}(\mathbf A)$. It equals $\lambda$ for every unit vector $\mathbf v$ of the eigenspace $E_\lambda$, since $\mathbf A\mathbf v=\lambda\mathbf v$ gives $\mathbf v^\top\mathbf A\mathbf v=\lambda\|\mathbf v\|^2=\lambda$. The converse fails: $\mathbf v^\top\mathbf A\mathbf v=\lambda$ does not make $\mathbf v$ an eigenvector, except when $\lambda$ is $\lambda_{\min}(\mathbf A)$ or $\lambda_{\max}(\mathbf A)$.</span></span> $\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k$.

### 3.1 Proposition 1 [Convergence of power iteration]

Let $\mathbf{v}_0\in\mathbb{R}^d$ be a unit vector, and define
$\Lambda_0=\{\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{v}_0\neq0\}.$
Thus, $\Lambda_0$ contains precisely the eigenvalues whose eigenspaces have a nonzero component in the initialization. Suppose that $\Lambda_0$ contains a nonzero eigenvalue $\lambda_\star$ satisfying $|\lambda_\star|>|\lambda|$ for every $\lambda\in\Lambda_0\setminus\{\lambda_\star\}$. The eigenvalue $\lambda_\star$ may have arbitrary multiplicity. Then the iteration

$$
\mathbf{v}_{k+1}:=\frac{\mathbf{A}\mathbf{v}_k}{\|\mathbf{A}\mathbf{v}_k\|}=\frac{\mathbf{A}^{k+1}\mathbf{v}_0}{\|\mathbf{A}^{k+1}\mathbf{v}_0\|},
\qquad k\ge0,
$$

is well defined, and

$$
\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k
\longrightarrow
\mathbf{q}_\star:=\frac{\Pi_{\lambda_\star}\mathbf{v}_0}{\|\Pi_{\lambda_\star}\mathbf{v}_0\|}.
$$

In particular,

$$
\mathbf{A}\mathbf{q}_\star=\lambda_\star\mathbf{q}_\star,
\qquad
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k\longrightarrow\lambda_\star .
$$

#### Remark 1 [Why the dominant component wins]
Recall that for every vector $\mathbf{v}\in\mathbb{R}^d$ and every integer $k\ge0$, 
$\mathbf{v}=\sum_{\lambda\in\Lambda(\mathbf{A})}\Pi_\lambda\mathbf{v}$ and
$\mathbf{A}^k\mathbf{v}=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^k\,\Pi_\lambda\mathbf{v}.$
So, for the iteration above, each application of $\mathbf{A}$ multiplies the component of the current vector in $E_\lambda$ by $\lambda$, so after $k$ steps the component in $E_\lambda$ carries the factor $\lambda^k$:

$$
\mathbf{A}^k\mathbf{v}_0
=\sum_{\lambda\in\Lambda_0}\lambda^k\,\Pi_\lambda\mathbf{v}_0
=\textcolor{#2563eb}{\lambda_\star^k\,\Pi_{\lambda_\star}\mathbf{v}_0}+\textcolor{#ea580c}{\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda^k\,\Pi_\lambda\mathbf{v}_0}
=\lambda_\star^k \left( \textcolor{#2563eb}{\Pi_{\lambda_\star}\mathbf{v}_0}+\textcolor{#ea580c}{\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}} \left(\frac{\lambda}{\lambda_\star}\right)^k\,\Pi_\lambda\mathbf{v}_0} \right).
$$

Among the components present in $\mathbf{v}_0$, the <span style="color:#2563eb">blue</span> one, with the largest $|\lambda|$ (we have $|\lambda_\star|>|\lambda|$ for every $\lambda\in\Lambda_0\setminus\{\lambda_\star\}$ by assumption), grows fastest, and the normalization removes the common factor $|\lambda_\star|^k$, so the <span style="color:#ea580c">orange</span> components fade relative to it. Only the sign of $\lambda_\star^k$ survives, which is why the statement multiplies $\mathbf{v}_k$ by $\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k$. The limit is the normalized projection of $\mathbf{v}_0$ onto the whole eigenspace $E_{\lambda_\star}$: inside a repeated eigenspace, the direction is chosen by the initialization, not by the method.

#### Proof

Set $\mathbf{p}_\lambda=\Pi_\lambda\mathbf{v}_0$. The spectral decomposition gives

$$
\mathbf{v}_0=\sum_{\lambda\in\Lambda_0}\mathbf{p}_\lambda,
\qquad
\mathbf{A}^k\mathbf{v}_0=\sum_{\lambda\in\Lambda_0}\lambda^k\mathbf{p}_\lambda
\quad(k\ge1).
$$

**Well-definedness.** Because $\mathbf{p}_{\lambda_\star}\neq0$ and $\lambda_\star\neq0$, the component
$\Pi_{\lambda_\star}\mathbf{A}^k\mathbf{v}_0=\lambda_\star^k\mathbf{p}_{\lambda_\star}$
is nonzero for every $k\ge0$. Consequently, $\mathbf{A}^k\mathbf{v}_0\neq0$ for every $k\ge0$. Since $\|\mathbf{v}_0\|=1$, repeated normalization satisfies
$
\mathbf{v}_k=\frac{\mathbf{A}^k\mathbf{v}_0}{\|\mathbf{A}^k\mathbf{v}_0\|}.
$
Indeed, this identity holds at $k=0$. If it holds at iteration $k$, then

$$
\begin{aligned}
\mathbf{v}_{k+1}
&=\frac{\mathbf{A}\bigl(\mathbf{A}^k\mathbf{v}_0/\|\mathbf{A}^k\mathbf{v}_0\|\bigr)}{\bigl\|\mathbf{A}\bigl(\mathbf{A}^k\mathbf{v}_0/\|\mathbf{A}^k\mathbf{v}_0\|\bigr)\bigr\|}\\
&=\frac{\mathbf{A}^{k+1}\mathbf{v}_0}{\|\mathbf{A}^{k+1}\mathbf{v}_0\|}.
\end{aligned}
$$
<!-- because the positive scalar $1/\|\mathbf{A}^k\mathbf{v}_0\|$ cancels between the numerator and the denominator.-->

**Convergence of the direction.** For $k\ge1$, define the scaled remainder

$$
\mathbf{r}_k=\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left(\frac{\lambda}{\lambda_\star}\right)^k\mathbf{p}_\lambda .
$$

Then

$$
\mathbf{A}^k\mathbf{v}_0=\lambda_\star^k\bigl(\textcolor{#2563eb}{\mathbf{p}_{\lambda_\star}}+\textcolor{#ea580c}{\mathbf{r}_k}\bigr).
$$

The eigenspaces are orthogonal, so $\mathbf{p}_{\lambda_\star}^\top\mathbf{r}_k=0$ and

$$
\|\mathbf{r}_k\|^2=\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2\longrightarrow0,
$$

since every ratio $|\lambda/\lambda_\star|$ in this sum is smaller than one. Therefore,

$$
\begin{aligned}
\mathbf{v}_k
&= \frac{\mathbf{A}^k\mathbf{v}_0}{\|\mathbf{A}^k\mathbf{v}_0\|}
%=\frac{\lambda_\star^k\bigl(\mathbf{p}_{\lambda_\star}+\mathbf{r}_k\bigr)}{|\lambda_\star|^k\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}}
=\frac{\lambda_\star^k}{|\lambda_\star|^k} \frac{\textcolor{#2563eb}{\mathbf{p}_{\lambda_\star}}+\textcolor{#ea580c}{\mathbf{r}_k}}{\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}}
=\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k \frac{\textcolor{#2563eb}{\mathbf{p}_{\lambda_\star}}+\textcolor{#ea580c}{\mathbf{r}_k}}{\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}},\\
\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k
&=\frac{\textcolor{#2563eb}{\mathbf{p}_{\lambda_\star}}+\textcolor{#ea580c}{\mathbf{r}_k}}{\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}}
\longrightarrow\frac{\mathbf{p}_{\lambda_\star}}{\|\mathbf{p}_{\lambda_\star}\|}.
\end{aligned}
$$

The <span style="color:#2563eb">blue term</span> is the component along the dominant eigenspace, and it keeps its size after rescaling. The <span style="color:#ea580c">orange term</span> collects every other component, and it shrinks geometrically. Since $\mathbf{p}_{\lambda_\star}\in E_{\lambda_\star}$, the limit $\mathbf{q}_\star$ satisfies $\mathbf{A}\mathbf{q}_\star=\lambda_\star\mathbf{q}_\star$.

**Rayleigh quotient.** For $k\ge1$, the symmetry of $\mathbf{A}$ and its spectral decomposition $\mathbf{A}=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda\,\Pi_\lambda$ give<span id="eq-rayleigh-quotient"></span>

$$
\begin{align*}
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k
&=\frac{(\mathbf{A}^k\mathbf{v}_0)^\top\mathbf{A}(\mathbf{A}^k\mathbf{v}_0)}{\|\mathbf{A}^k\mathbf{v}_0\|^2}\\
&=\frac{\mathbf{v}_0^\top \mathbf{A}^{2k+1}\mathbf{v}_0}{\mathbf{v}_0^\top \mathbf{A}^{2k}\mathbf{v}_0}\\
&=\frac{\displaystyle\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^{2k+1}\,\mathbf{v}_0^\top\Pi_\lambda\mathbf{v}_0}{\displaystyle\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^{2k}\,\mathbf{v}_0^\top\Pi_\lambda\mathbf{v}_0}\\
&=\frac{\displaystyle\sum_{\lambda\in\Lambda_0}\lambda^{2k+1}\|\mathbf{p}_\lambda\|^2}{\displaystyle\sum_{\lambda\in\Lambda_0}\lambda^{2k}\|\mathbf{p}_\lambda\|^2}\\
&=\frac{\displaystyle\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2+\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\displaystyle\|\mathbf{p}_{\lambda_\star}\|^2+\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}\tag{1}\\
&\longrightarrow\frac{\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2}=\lambda_\star .
\end{align*}
$$

The second line uses $(\mathbf{A}^k\mathbf{v}_0)^\top=\mathbf{v}_0^\top\mathbf{A}^k$, since $\mathbf{A}$ is symmetric. The third expands $\mathbf{A}^{2k+1}$ and $\mathbf{A}^{2k}$ with $\mathbf{A}^j=\sum_\lambda\lambda^j\,\Pi_\lambda$. The fourth uses $\mathbf{v}_0^\top\Pi_\lambda\mathbf{v}_0=\mathbf{v}_0^\top\Pi_\lambda^\top\Pi_\lambda\mathbf{v}_0=\|\Pi_\lambda\mathbf{v}_0\|^2=\|\mathbf{p}_\lambda\|^2$, which holds because $\Pi_\lambda$ is symmetric and $\Pi_\lambda^2=\Pi_\lambda$, and only the eigenvalues of $\Lambda_0$ remain, since $\mathbf{p}_\lambda=0$ for the others. The fifth divides the numerator and the denominator by $\lambda_\star^{2k}>0$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$


#### Remark 2 [A negative dominant eigenvalue]

If $\lambda_\star>0$, [Proposition 1](#31-proposition-1-convergence-of-power-iteration) gives $\mathbf{v}_k\to\mathbf{q}_\star$. If $\lambda_\star<0$, it gives $(-1)^k\mathbf{v}_k\to\mathbf{q}_\star$, or, equivalently,
$\mathbf{v}_{2k}\longrightarrow\mathbf{q}_\star$ and $\mathbf{v}_{2k+1}\longrightarrow-\mathbf{q}_\star.$
Thus, in the latter case, the even and odd subsequences converge to opposite unit vectors. The raw sequence $\{\mathbf{v}_k\}_{k\ge0}$ does not converge, but $\min\bigl\{\|\mathbf{v}_k-\mathbf{q}_\star\|,\|\mathbf{v}_k+\mathbf{q}_\star\|\bigr\}\longrightarrow0.$
A stopping rule that compares consecutive iterates, such as $\|\mathbf{v}_{k+1}-\mathbf{v}_k\|\le\varepsilon$ with $\varepsilon<2$, is therefore eventually never satisfied when $\lambda_\star<0$, because $\|\mathbf{v}_{k+1}-\mathbf{v}_k\|\to2\|\mathbf{q}_\star\|=2$, even though the eigendirection has converged. The Rayleigh quotient converges in both cases.

### 3.2 Theorem 1 [Rate of convergence]

Assume the hypotheses of [Proposition 1](#31-proposition-1-convergence-of-power-iteration), and write $\mathbf{p}_{\lambda_\star}=\Pi_{\lambda_\star}\mathbf{v}_0$. Let $\rho_2=\max\{|\lambda|:\lambda\in\Lambda_0\setminus\{\lambda_\star\}\},$
with $\rho_2=0$ when $\Lambda_0=\{\lambda_\star\}$, and let $\phi_0\in[0,\pi/2)$ be <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="angle-to-eigenspace" aria-label="Explain the angle between a vector and a subspace">the angle between $\mathbf{v}_0$ and $E_{\lambda_\star}$</button><span id="angle-to-eigenspace" class="explanation-popover" popover="auto" role="note" aria-label="Angle between a vector and a subspace" data-label="Geometry">Write $\mathbf v_0=\mathbf p_{\lambda_\star}+(\mathbf v_0-\mathbf p_{\lambda_\star})$, where $\mathbf p_{\lambda_\star}=\Pi_{\lambda_\star}\mathbf v_0$ lies in $E_{\lambda_\star}$ and the remainder is orthogonal to it. <span class="explanation-popover__equation"><img src="/images/blog/matrix-vector-products/angle-to-eigenspace.svg" width="300" height="164" alt="A right triangle: the unit vector v0 from the origin as hypotenuse, its projection p along the line of the eigenspace, the perpendicular side v0 minus p, a right angle at the foot of the perpendicular, and the angle phi0 at the origin" style="display:block;margin:0 auto;max-width:100%;height:auto"></span> The three vectors form a right triangle: the hypotenuse $\mathbf v_0$ (<span style="color:#27322f">black</span>), a side $\mathbf p_{\lambda_\star}$ along $E_{\lambda_\star}$ (<span style="color:#2563eb">blue</span>), and a side $\mathbf v_0-\mathbf p_{\lambda_\star}$ perpendicular to it (<span style="color:#ea580c">orange</span>). The angle between a vector and a subspace is the angle between the vector and its orthogonal projection, the smallest angle it makes with a vector of the subspace, so $\phi_0$ is the angle between $\mathbf v_0$ and $\mathbf p_{\lambda_\star}$. The tangent is the opposite side over the adjacent side, and since $\|\mathbf v_0\|=1$, also $\cos(\phi_0)=\|\mathbf p_{\lambda_\star}\|$ and $\sin(\phi_0)=\|\mathbf v_0-\mathbf p_{\lambda_\star}\|$. The hypothesis $\mathbf p_{\lambda_\star}\neq0$ keeps $\phi_0$ below $\pi/2$.</span></span>, so that $\tan(\phi_0)=\|\mathbf{v}_0-\mathbf{p}_{\lambda_\star}\|/\|\mathbf{p}_{\lambda_\star}\|$. Then, for every $k\ge1$,
$$
\Bigl\|\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k-\mathbf{q}_\star\Bigr\|
\le\textcolor{#ea580c}{\left(\frac{\rho_2}{|\lambda_\star|}\right)^{k}}\tan(\phi_0),
\qquad
\bigl|\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k-\lambda_\star\bigr|
\le2|\lambda_\star|\textcolor{#ea580c}{\left(\frac{\rho_2}{|\lambda_\star|}\right)^{2k}}\tan^2(\phi_0).
$$

#### Remark 3 [The convergence factor]

Relative to the dominant component, each product multiplies the component in $E_\lambda$ by $\lambda/\lambda_\star$ ([Remark 1](#remark-1-why-the-dominant-component-wins)), so the slowest of them, with $|\lambda|=\rho_2$, sets the pace. The eigenvector error therefore shrinks by the <span style="color:#ea580c">factor</span> $\textcolor{#ea580c}{\rho_2/|\lambda_\star|}$ per product, and the eigenvalue error by its square, because the Rayleigh quotient is stationary at an eigenvector. The factor can be arbitrarily close to one. [Remark 5](#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude) below is the limiting case $\rho_2=|\lambda_\star|$, in which convergence can fail altogether.

This factor is related to the condition number $\kappa(\mathbf{A})$ of $\mathbf{A}$ in the following way.
* For an invertible $\mathbf{A}$, the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="condition-number-symmetric" aria-label="Derive the condition number of a symmetric matrix from its eigenvalues">condition number</button><span id="condition-number-symmetric" class="explanation-popover" popover="auto" role="note" aria-label="Condition number of a symmetric matrix" data-label="Derivation">For a matrix $\mathbf M$, $\|\mathbf M\|=\max_{\|\mathbf z\|=1}\|\mathbf M\mathbf z\|$ is the largest factor by which $\mathbf M$ stretches a vector, and the condition number of an invertible $\mathbf A$ is <span class="explanation-popover__equation">$\kappa(\mathbf A)=\|\mathbf A\|\,\|\mathbf A^{-1}\|.$</span> For a symmetric $\mathbf A$, write a unit vector along the eigenspaces, $\mathbf z=\sum_\lambda\Pi_\lambda\mathbf z$, so that $\mathbf A\mathbf z=\sum_\lambda\lambda\,\Pi_\lambda\mathbf z$. The eigenspaces are orthogonal, so Pythagoras gives <span class="explanation-popover__equation">$\|\mathbf A\mathbf z\|^2=\sum_\lambda\lambda^2\|\Pi_\lambda\mathbf z\|^2\le(\max_\lambda\lambda^2)\sum_\lambda\|\Pi_\lambda\mathbf z\|^2=\max_\lambda\lambda^2,$</span> with equality when $\mathbf z$ is a unit eigenvector of an eigenvalue of largest magnitude. Hence $\|\mathbf A\|=\max_{\lambda\in\Lambda(\mathbf A)}|\lambda|$. If $\mathbf A$ is invertible, $0$ is not an eigenvalue, since $\mathbf A\mathbf u=0$ forces $\mathbf u=0$, and $\mathbf A^{-1}=\sum_\lambda\lambda^{-1}\,\Pi_\lambda$, because multiplying it by $\mathbf A=\sum_\mu\mu\,\Pi_\mu$ gives $\sum_\lambda\Pi_\lambda=\mathbb I_d$. So $\mathbf A^{-1}$ is symmetric, with the same eigenspaces and the eigenvalues $1/\lambda$, and the same argument gives $\|\mathbf A^{-1}\|=\max_\lambda1/|\lambda|=1/\min_{\lambda\in\Lambda(\mathbf A)}|\lambda|$. Multiplying the two norms, <span class="explanation-popover__equation">$\kappa(\mathbf A)=\dfrac{\max_{\lambda\in\Lambda(\mathbf A)}|\lambda|}{\min_{\lambda\in\Lambda(\mathbf A)}|\lambda|}.$</span> Without symmetry, $\|\mathbf A\|$ and $1/\|\mathbf A^{-1}\|$ are the largest and smallest singular values of $\mathbf A$, which can differ greatly from the magnitudes of its eigenvalues.</span></span> $\kappa(\mathbf{A})=\max_{\lambda\in\Lambda(\mathbf{A})}|\lambda|/\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda|$ compares the largest magnitude with the smallest, while the factor compares the largest with the next one. The two are linked only by a lower bound. When $\Lambda_0$ contains an eigenvalue other than $\lambda_\star$, $|\lambda_\star|\le\|\mathbf{A}\|:=\max_{\|\mathbf{z}\|=1}\|\mathbf{A}\mathbf{z}\|$ and $\rho_2\ge\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda|=1/\|\mathbf{A}^{-1}\|$, so
$$\textcolor{#ea580c}{\rho_2/|\lambda_\star|}\ge1/\kappa(\mathbf{A}),$$ 
with equality when $\Lambda_0=\Lambda(\mathbf{A})$ has exactly two elements. A well-conditioned matrix is therefore a slow one for power iteration. When $\kappa(\mathbf{A})$ is close to one, all the eigenvalues have nearly the same magnitude, and the factor is close to one as well. For $\mathbf{A}=\operatorname{diag}(1.01,1)$, $\kappa(\mathbf{A})=1.01$, and each decimal digit of the eigenvector costs about <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="products-per-digit" aria-label="Explain how the number of products per decimal digit is computed">$\ln10/\ln1.01\approx231$ products</button><span id="products-per-digit" class="explanation-popover" popover="auto" role="note" aria-label="Products per decimal digit" data-label="Derivation">One decimal digit of accuracy means dividing the error by $10$. By [Theorem 1](#32-theorem-1-rate-of-convergence), after $k$ products the eigenvector error is at most $(\rho_2/|\lambda_\star|)^k\tan(\phi_0)$, here $(1/1.01)^k\tan(\phi_0)$. Each product multiplies this bound by $1/1.01$, so $k$ products multiply it by $(1/1.01)^k$. The bound has thus gained one digit when <span class="explanation-popover__equation">$\left(\dfrac{1}{1.01}\right)^k=\dfrac{1}{10}\iff k=\dfrac{\ln10}{\ln1.01}\approx\dfrac{2.3026}{0.00995}\approx231.$</span> The logarithm solves for the unknown exponent, and the ratio does not depend on its base, since $\ln10/\ln1.01=1/\log_{10}1.01$. For a factor $1-\delta$, where $\delta=1-\rho_2/|\lambda_\star|$ is the relative gap, the same computation gives <span class="explanation-popover__equation">$k=\dfrac{\ln10}{-\ln(1-\delta)}\approx\dfrac{2.3}{\delta}$</span> products per digit when $\delta$ is small, because $-\ln(1-\delta)\approx\delta$. The cost grows like $1/\delta$. Here $\delta\approx0.0099$, and the approximation gives about $232$.</span></span>. This is the reverse of the conjugate gradient method for $\mathbf{A}\mathbf{x}=\mathbf{b}$ ([Hestenes and Stiefel, 1952](#ref-hestenes1952)), which converges fast when $\kappa(\mathbf{A})$ is small. A large condition number, on the other hand, implies nothing. $\operatorname{diag}(1,10^{-6})$ has $\kappa=10^6$ and the factor $10^{-6}$, so each product gains six digits, whereas $\operatorname{diag}(1.01,1,10^{-6})$ also has $\kappa\approx10^6$ but is as slow as $\operatorname{diag}(1.01,1)$. 
The pace is set by the relative gap $1-\rho_2/|\lambda_\star|$ at the top of the spectrum, which depends only on the two largest magnitudes present.

* For a singular $\mathbf{A}$, $\kappa(\mathbf{A})=\infty$ by convention, and the bound above says nothing. It still holds with the zero eigenvalue left out. The pseudo-inverse $\mathbf{A}^+=\sum_{\lambda\in\Lambda(\mathbf{A})\setminus\{0\}}\lambda^{-1}\,\Pi_\lambda$ inverts $\mathbf{A}$ on $\operatorname{range}(\mathbf{A})$ and vanishes on $E_0=\ker(\mathbf{A})$, and
$\kappa^+(\mathbf{A})=\|\mathbf{A}\|\,\|\mathbf{A}^+\|=\max_{\lambda\in\Lambda(\mathbf{A})}|\lambda|/\min_{\lambda\in\Lambda(\mathbf{A})\setminus\{0\}}|\lambda|$
compares the largest magnitude with the smallest nonzero one. When $\Lambda_0$ contains a nonzero eigenvalue other than $\lambda_\star$, $\rho_2\ge\min_{\lambda\in\Lambda(\mathbf{A})\setminus\{0\}}|\lambda|=1/\|\mathbf{A}^+\|$, so $\rho_2/|\lambda_\star|\ge1/\kappa^+(\mathbf{A})$, and everything said above about $\kappa(\mathbf{A})$ applies to $\kappa^+(\mathbf{A})$. For instance, $\operatorname{diag}(1.01,1,0)$ has $\kappa^+=1.01$ and is as slow as $\operatorname{diag}(1.01,1)$. Otherwise, the only other eigenvalue that can be present is $0$, so $\rho_2=0$, and a single product gives the eigenvector exactly. In both cases, a zero eigenvalue never slows power iteration. Its magnitude is below every other, and the first product removes the component along $E_0$.

#### Proof

Recall from the proof of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) the scaled remainder and the normalized iterate:

$$
\mathbf{r}_k=\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left(\frac{\lambda}{\lambda_\star}\right)^k\mathbf{p}_\lambda,\qquad
\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k=\frac{\mathbf{p}_{\lambda_\star}+\mathbf{r}_k}{\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}} 
\longrightarrow
\mathbf{q}_\star=\frac{\Pi_{\lambda_\star}\mathbf{v}_0}{\|\Pi_{\lambda_\star}\mathbf{v}_0\|} = \frac{\mathbf{p}_{\lambda_\star}}{\|\mathbf{p}_{\lambda_\star}\|}.
$$

The vectors $\mathbf{p}_\lambda$ lie in mutually orthogonal eigenspaces, so the squared norm of a combination of them is the sum of the squared norms of its terms (Pythagoras). Every $|\lambda|$ in the sum defining $\mathbf{r}_k$ is at most $\rho_2$, hence

$$
\|\mathbf{r}_k\|^2=\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2
\le\left(\frac{\rho_2}{|\lambda_\star|}\right)^{2k}\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\|\mathbf{p}_\lambda\|^2
=\left(\frac{\rho_2}{|\lambda_\star|}\right)^{2k}\|\mathbf{v}_0-\mathbf{p}_{\lambda_\star}\|^2,
$$

where the last equality is Pythagoras again, applied to $\mathbf{v}_0-\mathbf{p}_{\lambda_\star}=\sum_{\lambda\neq\lambda_\star}\mathbf{p}_\lambda$. 
Write $t_k=\|\mathbf{r}_k\|/\|\mathbf{p}_{\lambda_\star}\|$, so that $\boldsymbol{t_k\le(\rho_2/|\lambda_\star|)^k\tan(\phi_0)}$. 
Since $\mathbf{r}_k$ is orthogonal to $\mathbf{p}_{\lambda_\star}$ and $\mathbf{q}_\star=\mathbf{p}_{\lambda_\star}/\|\mathbf{p}_{\lambda_\star}\|$, the inner product of the unit vector $\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k$ with $\mathbf{q}_\star$ is

$$
\bigl(\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k\bigr)^\top\mathbf{q}_\star=\frac{\|\mathbf{p}_{\lambda_\star}\|^2}{\|\mathbf{p}_{\lambda_\star}\|\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}}=\frac{1}{\sqrt{1+t_k^2}} .
$$

For unit vectors $\mathbf{w}$ and $\mathbf{q}$, $\|\mathbf{w}-\mathbf{q}\|^2=\|\mathbf{w}\|^2-2\mathbf{w}^\top\mathbf{q}+\|\mathbf{q}\|^2=2-2\mathbf{w}^\top\mathbf{q}$, so

$$
\Bigl\|\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k-\mathbf{q}_\star\Bigr\|^2
=2-\frac{2}{\sqrt{1+t_k^2}}
\le t_k^2,
$$

because $1/\sqrt{1+t}\ge1-t/2$ for all $t\ge0$ (the convex function $t\mapsto1/\sqrt{1+t}$ lies above its tangent line at $0$). Taking square roots and using the bound on $t_k$ gives the first inequality. For the Rayleigh quotient, recall [equation (1)](#eq-rayleigh-quotient) from the proof of [Proposition 1](#31-proposition-1-convergence-of-power-iteration), whose denominator equals $\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2$ by the expression of $\|\mathbf{r}_k\|^2$ above:

$$
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k
=\frac{\displaystyle\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2+\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2} .
$$

Multiplying the expression of $\|\mathbf{r}_k\|^2$ above by $\lambda_\star$ gives $\lambda_\star\|\mathbf{r}_k\|^2=\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda_\star\left|\lambda/\lambda_\star\right|^{2k}\|\mathbf{p}_\lambda\|^2$, so $\lambda_\star$ can be written over the same denominator as

$$
\lambda_\star=\frac{\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2+\lambda_\star\|\mathbf{r}_k\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}
=\frac{\displaystyle\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2+\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda_\star\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2} .
$$

Subtracting it from the Rayleigh quotient, the terms $\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2$ cancel and the two sums combine term by term, which gives

$$
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k-\lambda_\star
=\frac{\displaystyle\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}(\lambda-\lambda_\star)\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2} .
$$

Taking absolute values,

$$
\begin{aligned}
\bigl|\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k-\lambda_\star\bigr|
&\le\frac{\displaystyle\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}|\lambda-\lambda_\star|\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}\\
&\le\frac{\displaystyle2|\lambda_\star|\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2}
=\frac{2|\lambda_\star|\,\|\mathbf{r}_k\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2}
=2|\lambda_\star|\,t_k^2\\
&\le2|\lambda_\star|\left(\frac{\rho_2}{|\lambda_\star|}\right)^{2k}\tan^2(\phi_0).
\end{aligned}
$$

The first inequality moves the absolute value inside the sum (triangle inequality). The second uses $|\lambda-\lambda_\star|\le|\lambda|+|\lambda_\star|<2|\lambda_\star|$, since every other eigenvalue present satisfies $|\lambda|<|\lambda_\star|$, and makes the denominator smaller by dropping $\|\mathbf{r}_k\|^2$. The two equalities are the expression of $\|\mathbf{r}_k\|^2$ above and the definition of $t_k$, and the last inequality squares the bound on $t_k$. This is the second inequality of the theorem.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

#### Remark 4 [Which eigenvalue is found]

The hypothesis of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) concerns distinct eigenvalues present in the initialization. It does not require the dominant eigenvalue to be <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="simple-eigenvalue" aria-label="Explain what a simple eigenvalue is">simple</button><span id="simple-eigenvalue" class="explanation-popover" popover="auto" role="note" aria-label="Simple eigenvalue" data-label="Definition">A simple eigenvalue is an eigenvalue that appears only once as a root of a matrix's characteristic polynomial $\lambda\mapsto\det(\lambda\,\mathbb I_d-\mathbf A)$, that is, with <strong>algebraic multiplicity</strong> one. For a symmetric matrix, the number of times $\lambda$ appears as a root equals the dimension $m_\lambda$ of its eigenspace (<strong>geometric multiplicity</strong>), so $\lambda$ is simple exactly when $m_\lambda=1$. Its eigenspace $E_\lambda$ is then a line, and its unit eigenvector is unique up to sign. For example, in $\operatorname{diag}(3,3,1)$, the eigenvalue $1$ is simple, with $E_1=\operatorname{span}\{[0,0,1]^\top\}$, whereas $3$ is a double root, with $E_3=\operatorname{span}\{[1,0,0]^\top,[0,1,0]^\top\}$. Power iteration started from $\mathbf v_0=[a,b,c]^\top$, with $(a,b)\neq(0,0)$, converges to <span class="explanation-popover__equation">$\dfrac{[a,b,0]^\top}{\sqrt{a^2+b^2}},$</span> an eigenvector of $3$ that depends on $\mathbf v_0$, so different initializations give different eigenvectors. When the dominant eigenvalue is simple, every admissible initialization gives the same eigenvector, up to sign.</span></span>, since the limit is the normalized projection of $\mathbf{v}_0$ onto its entire eigenspace. To obtain an eigenvalue of globally largest absolute value, the initialization must have a nonzero projection onto a corresponding eigenspace. For example, if
$\mathbf{v}_0=\mathbf{g}/\|\mathbf{g}\|$ with
$\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$,
then $\Pi_\lambda\mathbf{v}_0\neq0$ for every $\lambda\in\Lambda(\mathbf{A})$ with probability one. Indeed, for each $\lambda$, the event $\Pi_\lambda\mathbf{g}=0$ means that $\mathbf{g}$ belongs to the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="proper-subspace" aria-label="Explain why this subspace is proper">proper linear subspace</button><span id="proper-subspace" class="explanation-popover" popover="auto" role="note" aria-label="Proper subspace" data-label="Definition">A subspace of $\mathbb R^d$ is proper when it is not all of $\mathbb R^d$, that is, when its dimension is less than $d$. Here $E_\lambda$ contains an eigenvector $\mathbf u\neq0$ of $\lambda$, and $\mathbf u^\top\mathbf u=\|\mathbf u\|^2>0$, so $\mathbf u$ is not orthogonal to itself and $\mathbf u\notin E_\lambda^\perp$. Hence $E_\lambda^\perp\neq\mathbb R^d$, and indeed <span class="explanation-popover__equation">$\dim E_\lambda^\perp=d-m_\lambda\le d-1<d.$</span></span></span> $E_\lambda^\perp$, which has <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="gaussian-null-subspace" aria-label="Explain why a proper subspace has Gaussian probability zero">Gaussian probability zero</button><span id="gaussian-null-subspace" class="explanation-popover" popover="auto" role="note" aria-label="Gaussian measure of a proper subspace" data-label="Fact">A proper subspace $S$ lies in a hyperplane $\{\mathbf x:\mathbf c^\top\mathbf x=0\}$ with $\mathbf c\neq0$. For $\mathbf g\sim\mathcal N(0,\mathbb I_d)$, the number $\mathbf c^\top\mathbf g$ is Gaussian with variance $\|\mathbf c\|^2\gt0$, so it equals zero with probability zero.</span></span>. There are only finitely many eigenspaces. Consequently, such an initialization satisfies the hypotheses of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) whenever $\mathbf{A}\neq0$ and exactly one distinct eigenvalue of $\mathbf{A}$ has maximal absolute value.

#### Remark 5 [Failure for opposite eigenvalues of equal dominant magnitude]

The condition examined here is the strict dominance assumption of [Proposition 1](#31-proposition-1-convergence-of-power-iteration). In words, some nonzero eigenvalue present in the initialization is strictly larger in absolute value than every other eigenvalue present. With $\Lambda_0=\{\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{v}_0\neq0\}$, the set of eigenvalues present, it reads

$$
\exists\,\lambda_\star\in\Lambda_0\setminus\{0\}\quad\forall\,\lambda\in\Lambda_0\setminus\{\lambda_\star\}:\qquad|\lambda_\star|>|\lambda| .
$$

The condition fails when <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="dominance-failure-examples" aria-label="Show matrices for which the strict dominance condition fails"> no eigenvalue present stands out in this way</button><span id="dominance-failure-examples" class="explanation-popover" popover="auto" role="note" aria-label="Examples of the failing condition" data-label="Example">The matrix $\mathbf A=\left[\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right]$ swaps the two coordinates of a vector. Its eigenvalues are $1$, with the eigenvector $[1,1]^\top$, and $-1$, with the eigenvector $[1,-1]^\top$. From $\mathbf v_0=[1,0]^\top=\tfrac12[1,1]^\top+\tfrac12[1,-1]^\top$, both are present, and each is matched in absolute value by the other. Power iteration gives <span class="explanation-popover__equation">$\mathbf v_1=[0,1]^\top,\quad\mathbf v_2=[1,0]^\top,\quad\mathbf v_3=[0,1]^\top,\quad\dots$</span> The iterates alternate between two vectors, neither of which is an eigenvector, and the Rayleigh quotient $\mathbf v_k^\top\mathbf A\mathbf v_k=0$ is constant but is not an eigenvalue. The condition also fails for $\operatorname{diag}(2,-2,1)$ when the first two coordinates of $\mathbf v_0$ are nonzero: $1$ is exceeded by $2$, and $2$ and $-2$ match each other. It holds for the same matrix when $\mathbf v_0=[1,0,1]^\top/\sqrt2$, because $-2$ is then absent and $2$ dominates $1$. It also holds for $\operatorname{diag}(2,2,1)$, because $2$ counts as one distinct eigenvalue, however many times it is repeated.</span></span>: every nonzero eigenvalue present is matched or exceeded in absolute value by another eigenvalue present, i.e.
$$
\forall\,\mu\in\Lambda_0\setminus\{0\}\quad\exists\,\lambda\in\Lambda_0\setminus\{\mu\}:\qquad|\lambda|\ge|\mu| .
$$

Suppose that the condition fails, while some nonzero eigenvalue is present in the initialization. Let $\rho=\max\{|\lambda|:\lambda\in\Lambda_0\}$ be the largest absolute value among the eigenvalues present, so that $\rho>0$. If only one element of $\Lambda_0$ had absolute value $\rho$, it would be strictly larger in absolute value than every other eigenvalue present, and the condition would hold with $\lambda_\star$ equal to that element. Since the condition fails, at least two distinct elements of $\Lambda_0$ have absolute value $\rho$, $+\rho$ and $-\rho$, and both are present in the initialization. We show that ordinary power iteration can then fail to converge to any eigendirection, even while its Rayleigh quotient converges.

Suppose that 
$\mathbf{p}_+=\Pi_\rho\mathbf{v}_0\neq0,$
$\mathbf{p}_-=\Pi_{-\rho}\mathbf{v}_0\neq0$;
and that every other eigenvalue present in $\mathbf{v}_0$ has absolute value smaller than $\rho$. For $k\ge1$, define the scaled remainder

$$
\mathbf{r}_k=\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\left(\frac{\lambda}{\rho}\right)^k\mathbf{p}_\lambda .
$$

Then

$$
\|\mathbf{r}_k\|^2=\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\left|\frac{\lambda}{\rho}\right|^{2k}\|\mathbf{p}_\lambda\|^2\longrightarrow0,
$$

since every ratio $|\lambda/\rho|$ in this sum is smaller than one, and

$$
\begin{aligned}
\mathbf{v}_k
&=\frac{\mathbf{A}^k\mathbf{v}_0}{\|\mathbf{A}^k\mathbf{v}_0\|}\\
&=\frac{\rho^k\bigl(\textcolor{#2563eb}{\mathbf{p}_+}+\textcolor{#ea580c}{(-1)^k\mathbf{p}_-}+\mathbf{r}_k\bigr)}{\rho^k\sqrt{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2+\|\mathbf{r}_k\|^2}}\\
&=\frac{\textcolor{#2563eb}{\mathbf{p}_+}+\textcolor{#ea580c}{(-1)^k\mathbf{p}_-}+\mathbf{r}_k}{\sqrt{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2+\|\mathbf{r}_k\|^2}} .
\end{aligned}
$$

The second line uses the spectral decomposition $\mathbf{A}^k\mathbf{v}_0=\rho^k\mathbf{p}_++(-\rho)^k\mathbf{p}_-+\sum_{|\lambda|<\rho}\lambda^k\mathbf{p}_\lambda$, and its denominator follows from Pythagoras, since the three terms in the parentheses lie in $E_\rho$, in $E_{-\rho}$ and in the sum of the other eigenspaces, which are mutually orthogonal. It is positive because $\mathbf{p}_+\neq0$. 
The <span style="color:#2563eb">blue</span> component keeps its sign, the <span style="color:#ea580c">orange</span> component changes sign at every step, and the remainder vanishes. Hence

$$
\mathbf{v}_{2k}\longrightarrow\frac{\textcolor{#2563eb}{\mathbf{p}_+}+\textcolor{#ea580c}{\mathbf{p}_-}}{\sqrt{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2}},
$$

whereas

$$
\mathbf{v}_{2k+1}\longrightarrow\frac{\textcolor{#2563eb}{\mathbf{p}_+}-\textcolor{#ea580c}{\mathbf{p}_-}}{\sqrt{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2}} .
$$

These two limits are not collinear. Indeed, if $\mathbf{p}_++\mathbf{p}_-=c(\mathbf{p}_+-\mathbf{p}_-)$ for some scalar $c$, projecting onto $E_\rho$ gives $c=1$, whereas projecting onto $E_{-\rho}$ gives $c=-1$. Thus no choice of an overall factor in $\{-1,1\}$ can make the iterates converge to a single unit eigenvector.
Nevertheless, the Rayleigh quotient satisfies

$$
\begin{aligned}
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k
&=\frac{\displaystyle\rho\|\mathbf{p}_+\|^2-\rho\|\mathbf{p}_-\|^2+\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\lambda\left|\frac{\lambda}{\rho}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\displaystyle\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2+\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\left|\frac{\lambda}{\rho}\right|^{2k}\|\mathbf{p}_\lambda\|^2}
%\\&
\longrightarrow\rho\,\frac{\|\mathbf{p}_+\|^2-\|\mathbf{p}_-\|^2}{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2}.
\end{aligned}
$$

This limit need not be an eigenvalue. For example, let

$$
\mathbf{A}=\begin{bmatrix}1&0\\0&-1\end{bmatrix},
\qquad
\mathbf{v}_0=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}.
$$

Then, at every iteration,

$$
\mathbf{v}_k=\frac{1}{\sqrt{2}}\begin{bmatrix}\textcolor{#2563eb}{1}\\\textcolor{#ea580c}{(-1)^k}\end{bmatrix},
\qquad
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k=0 .
$$

However,

$$
0\notin\Lambda(\mathbf{A}),
\qquad
\bigl\|\mathbf{A}\mathbf{v}_k-(\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k)\mathbf{v}_k\bigr\|=\|\mathbf{A}\mathbf{v}_k\|=1 .
$$

Thus a constant or convergent Rayleigh quotient does not by itself establish convergence to an eigenpair. This obstruction motivates the squared-power construction of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) below and the use of an eigenpair residual as a stopping criterion.

<figure class="article-figure" id="figure-power-iteration">
  <img src="/images/blog/matrix-vector-products/power-iteration.svg" width="1040" height="430" alt="Two panels of error curves against the number of matrix-vector products. Left: geometric decay of the sign-corrected eigenvector error and the faster decay of the Rayleigh-quotient error for a negative dominant eigenvalue, with the raw distance alternating near 2. Right: the residual of plain power iteration stalls when +1 and -1 tie, while squared power iteration with sign separation drives it to zero geometrically." loading="lazy" />
  <figcaption><strong>Figure 1.</strong> Power iteration on two symmetric matrices of size $d=60$ with random eigenvectors. Each curve is the median over 200 Gaussian initializations, and its band contains the middle half of the runs (the interquartile range). (a) The dominant eigenvalue $\lambda_\star=-1$ has multiplicity 3 and the next largest magnitude is $\rho_2=0.8$. The sign-corrected error $\|(-1)^k\mathbf v_k-\mathbf q_\star\|$ (<span style="color:#2563eb">blue</span>) decays like $0.8^k$ and the Rayleigh-quotient error (<span style="color:#ea580c">orange</span>) like $0.8^{2k}$, below the bounds of <a href="#32-theorem-1-rate-of-convergence">Theorem 1</a> (<span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>), while the raw distance $\|\mathbf v_k-\mathbf q_\star\|$ (<span style="color:#7d8783">grey</span>) alternates between those small values and nearly 2 (<a href="#remark-2-a-negative-dominant-eigenvalue">Remark 2</a>). (b) The eigenvalues $+1$ and $-1$ tie for the largest magnitude. The residual $\|\mathbf A\mathbf v-(\mathbf v^\top\mathbf A\mathbf v)\mathbf v\|$ of plain power iteration (<span style="color:#ea580c">orange</span>) stalls at order one (<a href="#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude">Remark 5</a>), whereas squared power iteration with sign separation (<span style="color:#2563eb">blue</span>; <a href="#35-proposition-2-power-iteration-with-sign-separation">Proposition 2</a>, run as in <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a>) drives the residual of its candidate to zero, by a factor of about $0.8$ per product. Both horizontal axes count matrix–vector products: $\mathbf v_k$ costs $k$ products and its Rayleigh quotient one more, and each squared step costs two.</figcaption>
</figure>

### 3.3 Corollary 1 [Recovering algebraic extrema by shifting]

Let $\lambda_{\min}=\min\Lambda(\mathbf{A})$ and $\lambda_{\max}=\max\Lambda(\mathbf{A})$.

If $\alpha\ge\lambda_{\max}$ and $\alpha>\lambda_{\min}$, then power iteration applied to $\mathbf{B}=\alpha\mathbb{I}_d-\mathbf{A}$ converges to a unit vector in $E_{\lambda_{\min}}$, provided its initialization has a nonzero projection onto that eigenspace. Its Rayleigh quotient converges to $\mu_{\max}=\alpha-\lambda_{\min}$, so $\lambda_{\min}=\alpha-\mu_{\max}$.

Similarly, if $\beta\le\lambda_{\min}$ and $\beta<\lambda_{\max}$, then power iteration applied to $\mathbf{C}=\mathbf{A}-\beta\mathbb{I}_d$ converges to a unit vector in $E_{\lambda_{\max}}$, provided its initialization has a nonzero projection onto that eigenspace. Its Rayleigh quotient converges to $\nu_{\max}=\lambda_{\max}-\beta$, so $\lambda_{\max}=\beta+\nu_{\max}$.

#### Remark 6 [What a shift does to the spectrum]

Power iteration returns the eigenvalue of largest magnitude, which is $\lambda_{\max}$ or $\lambda_{\min}$ depending on the signs. A scalar shift moves the whole spectrum without changing the eigenvectors, and the shift $\alpha\mathbb{I}_d-\mathbf{A}$ also reverses its order. With $\alpha\ge\lambda_{\max}$, the spectrum of $\mathbf{B}$ lies in $[0,\alpha-\lambda_{\min}]$, so the eigenvalue of $\mathbf{A}$ farthest below $\alpha$ becomes the dominant eigenvalue of $\mathbf{B}$.

#### Proof

For any eigenpair $(\lambda,\mathbf{u})$ of $\mathbf{A}$,
$\mathbf{B}\mathbf{u}=(\alpha\mathbb{I}_d-\mathbf{A})\mathbf{u}=(\alpha-\lambda)\mathbf{u}.$
Since an orthonormal eigenbasis of $\mathbf{A}$ is also an eigenbasis of $\mathbf{B}$,
$\Lambda(\mathbf{B})=\{\alpha-\lambda:\lambda\in\Lambda(\mathbf{A})\}.$
The assumptions on $\alpha$ give
$0\le\alpha-\lambda\le\textcolor{#2563eb}{\alpha-\lambda_{\min}}$ and
$\textcolor{#2563eb}{\alpha-\lambda_{\min}}>0.$
Whenever $\lambda>\lambda_{\min}$, $\alpha-\lambda<\alpha-\lambda_{\min}.$
Therefore, $\mathbf{B}$ has a positive dominant eigenvalue $\textcolor{#2563eb}{\alpha-\lambda_{\min}}$, whose eigenspace is exactly $E_{\lambda_{\min}}$. [Proposition 1](#31-proposition-1-convergence-of-power-iteration) applies, including when this eigenvalue is repeated.

For $\mathbf{C}$, 
$\mathbf{C}\mathbf{u}=(\mathbf{A}-\beta\mathbb{I}_d)\mathbf{u}=(\lambda-\beta)\mathbf{u}$
and $\Lambda(\mathbf{C})=\{\lambda-\beta:\lambda\in\Lambda(\mathbf{A})\}.$
The assumptions on $\beta$ give 
$0\le\lambda-\beta\le\textcolor{#2563eb}{\lambda_{\max}-\beta}$
and $\textcolor{#2563eb}{\lambda_{\max}-\beta}>0.$
Whenever $\lambda<\lambda_{\max}$, $\lambda-\beta<\lambda_{\max}-\beta.$
Thus $\mathbf{C}$ has a positive dominant eigenvalue $\textcolor{#2563eb}{\lambda_{\max}-\beta}$, with eigenspace $E_{\lambda_{\max}}$. [Proposition 1](#31-proposition-1-convergence-of-power-iteration) again applies.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

In particular, choosing $\alpha=\lambda_{\max}$ recovers the minimum eigenvalue from the maximum, unless $\mathbf{A}=\lambda_{\max}\mathbb{I}_d$. In that exceptional case, every eigenvalue already equals $\lambda_{\max}$. The analogous statement holds for $\beta=\lambda_{\min}$.

#### Remark 7 [The shift only needs to pass the midpoint]

The condition $\alpha\ge\lambda_{\max}$ is convenient but stronger than needed. [Proposition 1](#31-proposition-1-convergence-of-power-iteration) only requires that $\alpha-\lambda_{\min}$ strictly dominate $|\alpha-\lambda|$ for every other eigenvalue $\lambda$ present in the initialization. For $\lambda\le\alpha$, this holds because $\lambda>\lambda_{\min}$. For $\lambda>\alpha$, it reads $\lambda-\alpha<\alpha-\lambda_{\min}$, that is, $\lambda+\lambda_{\min}<2\alpha$. Hence every shift $\alpha>(\lambda_{\min}+\lambda_{\max})/2$ works. In particular, an underestimate of $\lambda_{\max}$, such as a Rayleigh quotient
$$
\frac{\mathbf{x}^\top\mathbf{A}\mathbf{x}}{\|\mathbf{x}\|^2}=\frac{\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda\,\|\Pi_\lambda\mathbf{x}\|^2}{\sum_{\lambda\in\Lambda(\mathbf{A})}\|\Pi_\lambda\mathbf{x}\|^2}\le\lambda_{\max}
\qquad(\mathbf{x}\neq0),
$$
is a valid shift as long as it exceeds the midpoint of the spectrum. Symmetrically, every $\beta<(\lambda_{\min}+\lambda_{\max})/2$ works for $\mathbf{C}$.

The midpoint is unknown when $\lambda_{\min}$ and $\lambda_{\max}$ are, so $\mathbf{x}$ must be chosen without it. A random vector is not enough. For $\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$, the unit vector $\mathbf{u}=\mathbf{g}/\|\mathbf{g}\|$ is uniformly distributed on the sphere, so $\mathbb{E}[\mathbf{u}\mathbf{u}^\top]=\mathbb{I}_d/d$ and
$\mathbb{E}\left[\mathbf{g}^\top\mathbf{A}\mathbf{g} / \|\mathbf{g}\|^2\right]=\operatorname{tr}\bigl(\mathbf{A}\,\mathbb{E}[\mathbf{u}\mathbf{u}^\top]\bigr)=\operatorname{tr}(\mathbf{A})/d,$
the mean of the eigenvalues counted with multiplicity, which can lie far below the midpoint. For the spectrum $\{10,0,\dots,0\}$, the midpoint is $5$, while the mean is $10/d$.

Power iteration itself supplies a suitable $\mathbf{x}$. Assume that one of the two extremes is strictly larger in absolute value than the other, so that, with a random initialization ([Remark 4](#remark-4-which-eigenvalue-is-found)), power iteration on $\mathbf{A}$ converges to it. Take for $\mathbf{x}$ the iterate $\mathbf{v}_k$, and for the shift its Rayleigh quotient $\theta_k=\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k$. By [Theorem 1](#32-theorem-1-rate-of-convergence), $\theta_k$ converges to the dominant eigenvalue $\lambda_\star\in\{\lambda_{\min},\lambda_{\max}\}$, with
$|\theta_k-\lambda_\star|\le2|\lambda_\star|\left(\rho_2/|\lambda_\star|\right)^{2k}\tan^2(\phi_0).$
Both extremes lie at the distance $(\lambda_{\max}-\lambda_{\min})/2$ from the midpoint. As soon as $|\theta_k-\lambda_\star|$ is smaller than this distance, $\theta_k$ lies on the same side of the midpoint as $\lambda_\star$, and $\theta_k$ is a valid shift. The sign of $\lambda_\star$ tells which extreme was found. If $\lambda_\star>0$, every other eigenvalue satisfies $\lambda\le|\lambda|<\lambda_\star$, so $\lambda_\star=\lambda_{\max}$. If $\lambda_\star<0$, every other eigenvalue satisfies $\lambda\ge-|\lambda|>\lambda_\star$, so $\lambda_\star=\lambda_{\min}$. This sign can be read from $\theta_k$. In both cases, the midpoint lies strictly between $0$ and $\lambda_\star$. For instance, if $\lambda_\star=\lambda_{\max}>0$, then $|\lambda_{\min}|<\lambda_{\max}$ gives $\lambda_{\min}+\lambda_{\max}>0$. Hence a $\theta_k$ on the same side of the midpoint as $\lambda_\star$ has the sign of $\lambda_\star$. The sign also shows in the iterates: $\mathbf{v}_k^\top\mathbf{v}_{k+1}=\theta_k/\|\mathbf{A}\mathbf{v}_k\|$, so consecutive iterates point to opposite sides exactly when $\theta_k<0$, which is the alternation of [Remark 2](#remark-2-a-negative-dominant-eigenvalue). If $\theta_k>0$, the matrix to use is $\mathbf{B}=\theta_k\mathbb{I}_d-\mathbf{A}$ of [Corollary 1](#33-corollary-1-recovering-algebraic-extrema-by-shifting), with $\alpha=\theta_k$. If $\theta_k<0$, it is $\mathbf{C}=\mathbf{A}-\theta_k\mathbb{I}_d$, with $\beta=\theta_k$. Even this distinction is a convenience. The two matrices differ only by their sign, so they have the same eigenvectors, and power iteration on either, from a random initialization, converges, up to sign ([Remark 2](#remark-2-a-negative-dominant-eigenvalue)), to an eigenvector of the eigenvalue of $\mathbf{A}$ farthest from $\theta_k$, which is the other extreme.

Products alone cannot certify that $\theta_k$ has passed the midpoint, because the spread $\lambda_{\max}-\lambda_{\min}$ is unknown. A failure is nevertheless visible. If $\theta_k$ is still on the wrong side of the midpoint, the eigenvalue of $\mathbf{A}$ farthest from $\theta_k$ is $\lambda_\star$ itself, so the second run converges to an eigenvector of $\lambda_\star$, the eigenvalue already found by the first run, instead of the other extreme, and comparing the eigenvalues returned by the two runs reveals it. The remedy is to continue the first run, which brings $\theta_k$ closer to $\lambda_\star$, and to repeat the second. Once $|\theta_k-\lambda_\star|<(\lambda_{\max}-\lambda_{\min})/2$, the second run reaches the other extreme. This can only fail if the half-spread is below the accuracy reached, that is, if every eigenvalue of $\mathbf{A}$ is within twice that accuracy of $\lambda_\star$, and then the other extreme coincides with $\lambda_\star$ to that accuracy anyway. [Algorithm 1](#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation) avoids the question altogether: it uses no shift, reaches the remaining eigenpairs by deflation, and separates eigenvalues of opposite signs by [sign separation](#35-proposition-2-power-iteration-with-sign-separation).

Beyond the two extremes, however, a scalar shift does not help. It changes every eigenvalue, and the dominant eigenvalue of $\mathbf{A}-\sigma\mathbb{I}_d$ comes from the eigenvalue of $\mathbf{A}$ farthest from $\sigma$, which is $\lambda_{\min}$ or $\lambda_{\max}$ whatever $\sigma$ is. **No shift therefore makes an interior eigenvalue dominant. Deflation does.** Power iteration finds one eigenpair, the dominant one, and running it again on $\mathbf{A}$ would only find the same pair. To reach the next pair, deflation removes from $\mathbf{A}$ the eigendirections already found and leaves every other eigenpair untouched. In an orthonormal eigenbasis $\mathbf{q}_1,\dots,\mathbf{q}_d$ of $\mathbf{A}$, with $\mathbf{A}\mathbf{q}_i=\lambda_i\mathbf{q}_i$, the spectral decomposition writes $\mathbf{A}$ as a sum of rank-one layers, one per eigenvector, and deflating the first $r$ eigenvectors means deleting their layers, which gives a new matrix, the deflated matrix $\mathbf{B}$:

$$
\mathbf{A}=\textcolor{#ea580c}{\sum_{i=1}^r\lambda_i\mathbf{q}_i\mathbf{q}_i^\top}+\textcolor{#2563eb}{\sum_{i=r+1}^d\lambda_i\mathbf{q}_i\mathbf{q}_i^\top}
\qquad\longrightarrow\qquad
\mathbf{B}:=\textcolor{#2563eb}{\sum_{i=r+1}^d\lambda_i\mathbf{q}_i\mathbf{q}_i^\top} .
$$

This matrix has the same eigenvectors as $\mathbf{A}$. The ones already found (<span style="color:#ea580c">orange</span>) now have the eigenvalue $0$, and all the others (<span style="color:#2563eb">blue</span>) keep their eigenvalues. The largest remaining eigenvalue of $\mathbf{A}$ in absolute value is therefore the dominant eigenvalue of $\mathbf{B}$, and power iteration on $\mathbf{B}$ finds it. <!-- For example, if $\mathbf{A}=\operatorname{diag}(3,2,1)$, power iteration finds the eigenvalue $3$ with the eigenvector $\mathbf{e}_1$. Deleting the layer $3\,\mathbf{e}_1\mathbf{e}_1^\top$ leaves $\mathbf{B}=\operatorname{diag}(0,2,1)$, on which power iteration finds $2$ with $\mathbf{e}_2$, and deleting that layer as well leaves $\operatorname{diag}(0,0,1)$. --> The layer picture uses every eigenvector, but computing $\mathbf{B}$ does not: the lemma below shows that it only needs the eigenvectors already found, through the projection $\mathbf{P}=\mathbb{I}_d-\mathbf{Q}\mathbf{Q}^\top$ that removes the components along them.

### 3.4 Lemma 1 [Orthogonal deflation]

Let $\mathbf{q}_1,\dots,\mathbf{q}_r$ be exact orthonormal eigenvectors of $\mathbf{A}$, with associated eigenvalues $\lambda_1,\dots,\lambda_r$:

$$
\mathbf{A}\mathbf{q}_i=\lambda_i\mathbf{q}_i,
\qquad
\mathbf{q}_i^\top\mathbf{q}_\ell=\delta_{i\ell},
\qquad
i,\ell\in\{1,\dots,r\}.
$$

Define

$$
\mathbf{Q}=[\mathbf{q}_1,\dots,\mathbf{q}_r],
\qquad
\mathbf{D}=\operatorname{diag}(\lambda_1,\dots,\lambda_r),
\qquad
\mathbf{P}=\mathbb{I}_d-\mathbf{Q}\mathbf{Q}^\top .
$$

Then

$$
\mathbf{B}=\mathbf{P}\mathbf{A}\mathbf{P}=\mathbf{A}-\mathbf{Q}\mathbf{D}\mathbf{Q}^\top=\mathbf{A}-\sum_{i=1}^r\lambda_i\mathbf{q}_i\mathbf{q}_i^\top .
$$

Moreover,

1. **The eigenvectors already found have the eigenvalue $0$ for $\mathbf{B}$.** For each column $\mathbf{q}_i$ of $\mathbf{Q}$, $\mathbf{B}\mathbf{q}_i=0$, for $i\in\{1,\dots,r\}$.
2. **On the vectors orthogonal to the eigenvectors already found, $\mathbf{B}$ acts exactly as $\mathbf{A}$.** That is, $\mathbf{B}\mathbf{x}=\mathbf{A}\mathbf{x}$ for every $\mathbf{x}$ with $\mathbf{Q}^\top\mathbf{x}=0$, which is to say for every $\mathbf{x}\in\operatorname{range}(\mathbf{P})=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_r\}^\perp$.
3. **The other eigenpairs of $\mathbf{A}$ are eigenpairs of $\mathbf{B}$.** Every eigenvector of $\mathbf{A}$ orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_r$ is an eigenvector of $\mathbf{B}$ with the same eigenvalue. Hence the restriction of $\mathbf{B}$ to $\operatorname{range}(\mathbf{P}) = \operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_r\}^\perp$ has the remaining eigenvalues of $\mathbf{A}$, counted with multiplicity, and $\mathbf{B}$ has the eigenvalue $0$ on $\operatorname{range}(\mathbf{P})^\perp = \operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_r\}$.

#### Remark 8 [Two views of the deflated matrix]

The two formulas for $\mathbf{B}$ are two ways of computing a product $\mathbf{B}\mathbf{x}$ with one product by $\mathbf{A}$.

* **Projection,** $\mathbf{B}\mathbf{x}=\mathbf{P}\bigl(\mathbf{A}(\mathbf{P}\mathbf{x})\bigr)$: remove from $\mathbf{x}$ its components along the stored eigenvectors, apply $\mathbf{A}$, and remove those components again. A vector orthogonal to every stored eigenvector passes through unchanged (item 2), and a stored eigenvector is sent to $0$ (item 1).
* **Subtraction,** $\mathbf{B}\mathbf{x}=\mathbf{A}\mathbf{x}-\sum_{i=1}^r\lambda_i(\mathbf{q}_i^\top\mathbf{x})\,\mathbf{q}_i$: apply $\mathbf{A}$, then subtract the layers of the stored eigenpairs, exactly as in the picture above.

The two agree because the $\mathbf{q}_i$ are exact eigenvectors, and the proof below uses $\mathbf{A}\mathbf{Q}=\mathbf{Q}\mathbf{D}$. For computed, approximate eigenvectors they differ. The projection still returns vectors exactly orthogonal to the stored ones, so the eigenvectors found later stay orthogonal to them, which is why the implementation below applies $\mathbf{P}$ explicitly.

#### Proof

Since

$$
\mathbf{A}\mathbf{Q}=\mathbf{Q}\mathbf{D},
\qquad
\mathbf{Q}^\top\mathbf{A}=\mathbf{D}\mathbf{Q}^\top,
\qquad
\mathbf{Q}^\top\mathbf{Q}=\mathbb{I}_r,
$$

we obtain

$$
\begin{aligned}
\mathbf{P}\mathbf{A}\mathbf{P}
&=(\mathbb{I}_d-\mathbf{Q}\mathbf{Q}^\top)\mathbf{A}(\mathbb{I}_d-\mathbf{Q}\mathbf{Q}^\top)\\
&=\mathbf{A}-\mathbf{Q}\mathbf{Q}^\top\mathbf{A}-\mathbf{A}\mathbf{Q}\mathbf{Q}^\top+\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\mathbf{Q}\mathbf{Q}^\top\\
&=\mathbf{A}-\mathbf{Q}\mathbf{D}\mathbf{Q}^\top\textcolor{#2563eb}{-\mathbf{Q}\mathbf{D}\mathbf{Q}^\top+\mathbf{Q}\mathbf{D}\mathbf{Q}^\top}\\
&=\mathbf{A}-\mathbf{Q}\mathbf{D}\mathbf{Q}^\top .
\end{aligned}
$$

In the third line, each of the three products reduces to $\mathbf{Q}\mathbf{D}\mathbf{Q}^\top$, and the two <span style="color:#2563eb">blue terms</span> cancel. For each $i\in\{1,\dots,r\}$,

$$
\begin{aligned}
\mathbf{B}\mathbf{q}_i
&=\mathbf{A}\mathbf{q}_i-\sum_{\ell=1}^r\lambda_\ell\mathbf{q}_\ell(\mathbf{q}_\ell^\top\mathbf{q}_i)\\
&=\lambda_i\mathbf{q}_i-\lambda_i\mathbf{q}_i\\
&=0 .
\end{aligned}
$$

If $\mathbf{Q}^\top\mathbf{x}=0$, then

$$
\mathbf{B}\mathbf{x}=\mathbf{A}\mathbf{x}-\mathbf{Q}\mathbf{D}(\mathbf{Q}^\top\mathbf{x})=\mathbf{A}\mathbf{x}.
$$

Also,

$$
\mathbf{Q}^\top\mathbf{A}\mathbf{x}=\mathbf{D}\mathbf{Q}^\top\mathbf{x}=0,
$$

so $\operatorname{range}(\mathbf{P})$ is invariant under $\mathbf{A}$.

Extend $\mathbf{q}_1,\dots,\mathbf{q}_r$ to an orthonormal eigenbasis $\mathbf{q}_1,\dots,\mathbf{q}_d$ of $\mathbf{A}$. For every $i>r$,

$$
\mathbf{Q}^\top\mathbf{q}_i=0,
\qquad
\mathbf{B}\mathbf{q}_i=\mathbf{A}\mathbf{q}_i=\lambda_i\mathbf{q}_i .
$$

Thus the eigenpairs associated with $\mathbf{q}_{r+1},\dots,\mathbf{q}_d$ are preserved on $\operatorname{range}(\mathbf{P})$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

When an eigenvalue is repeated, deflating one unit eigenvector removes only that direction. Other directions in the same eigenspace that are orthogonal to it retain their eigenvalue.

Ordinary power iteration can be applied repeatedly to the deflated operator whenever the hypotheses of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) hold at each stage. The following modification also handles opposite eigenvalues of equal magnitude, without requiring a spectral bound, such as a lower bound $\beta\le\lambda_{\min}$ that would make $\mathbf{A}-\beta\mathbb{I}_d$ positive semidefinite and thus free of opposite eigenvalues. It squares the operator instead, which makes it positive semidefinite without any bound, and then separates the signs.

### 3.5 Proposition 2 [Power iteration with sign separation]

Let $\mathbf{B}\in\mathbb{R}^{d\times d}$ be symmetric, and let $\mathbf{w}_0$ be any unit vector. If $\mathbf{B}\mathbf{w}_0=0$, then $(0,\mathbf{w}_0)$ is an eigenpair. Otherwise, the iteration
$$
\mathbf{w}_{k+1}=\frac{\mathbf{B}^2\mathbf{w}_k}{\|\mathbf{B}^2\mathbf{w}_k\|}
$$
is well defined and converges to a unit vector $\mathbf{w}$ satisfying
$\mathbf{B}^2\mathbf{w}=\rho^2\mathbf{w}$
for some $\rho>0$. This value can be recovered as
$\rho=\|\mathbf{B}\mathbf{w}\|.$
Define
$$
\textcolor{#2563eb}{\mathbf{u}_+}=\frac12\left(\mathbf{w}+\frac{\mathbf{B}\mathbf{w}}{\rho}\right),
\qquad
\textcolor{#ea580c}{\mathbf{u}_-}=\frac12\left(\mathbf{w}-\frac{\mathbf{B}\mathbf{w}}{\rho}\right).
$$

Then

$$
\mathbf{B}\textcolor{#2563eb}{\mathbf{u}_+}=\rho\,\textcolor{#2563eb}{\mathbf{u}_+},
\qquad
\mathbf{B}\textcolor{#ea580c}{\mathbf{u}_-}=-\rho\,\textcolor{#ea580c}{\mathbf{u}_-} .
$$

At least one of $\mathbf{u}_+$ and $\mathbf{u}_-$ is nonzero. Normalizing either nonzero vector gives an eigenvector of $\mathbf{B}$.

#### Remark 9 [Sign separation is a reflection]

Squaring maps $+\rho$ and $-\rho$ to the same eigenvalue $\rho^2$ of $\mathbf{B}^2$, so the obstruction of [Remark 5](#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude) disappears: $\mathbf{B}^2$ is positive semidefinite, and its largest eigenvalue present in $\mathbf{w}_0$ is strictly dominant. The price is that the limit $\mathbf{w}$ may mix the two eigenspaces of $\mathbf{B}$ for $+\rho$ and $-\rho$. One more product separates them. The limit lies in the eigenspace of $\mathbf{B}^2$ for $\rho^2$, which is the sum of these two eigenspaces, so we can write $\mathbf{w}=\textcolor{#2563eb}{\mathbf{w}_+}+\textcolor{#ea580c}{\mathbf{w}_-}$ with $\mathbf{B}\mathbf{w}_\pm=\pm\rho\,\mathbf{w}_\pm$. Then $\mathbf{B}\mathbf{w}/\rho=\textcolor{#2563eb}{\mathbf{w}_+}-\textcolor{#ea580c}{\mathbf{w}_-}$ is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="sign-reflection" aria-label="Explain the reflection picture of sign separation">reflection</button><span id="sign-reflection" class="explanation-popover" popover="auto" role="note" aria-label="Sign separation as a reflection" data-label="Geometry">On the sum of the two eigenspaces, $\mathbf B/\rho$ fixes every vector of the $+\rho$ eigenspace and reverses every vector of the $-\rho$ eigenspace: it is the mirror reflection across the $+\rho$ eigenspace. Averaging a vector with its mirror image keeps its part on the mirror; averaging it with minus its mirror image keeps the other part.</span></span> of $\mathbf{w}$ that fixes the <span style="color:#2563eb">blue part</span> and reverses the <span style="color:#ea580c">orange part</span>, so $\mathbf{u}_+=\textcolor{#2563eb}{\mathbf{w}_+}$ and $\mathbf{u}_-=\textcolor{#ea580c}{\mathbf{w}_-}$ are exactly the orthogonal projections of $\mathbf{w}$ onto the two eigenspaces. The orthogonality and norm identities in the proof below are this observation written out.

#### Proof

Let $\Pi^{\mathbf{B}}_\lambda$ denote the orthogonal projection onto the eigenspace of $\mathbf{B}$ associated with $\lambda$. Since $\mathbf{B}\mathbf{w}_0\neq0$, the initialization has a nonzero component in at least one nonzero eigenspace of $\mathbf{B}$. Define
$$
\rho=\max\bigl\{|\lambda|:\lambda\in\Lambda(\mathbf{B}),\;\Pi^{\mathbf{B}}_\lambda\mathbf{w}_0\neq0\bigr\}>0 .
$$
The eigenvalues of $\mathbf{B}^2$ are the numbers $\lambda^2$. Its largest eigenvalue present in $\mathbf{w}_0$ is therefore $\rho^2>0$, and every other distinct eigenvalue present is strictly smaller than $\rho^2$.
[Proposition 1](#31-proposition-1-convergence-of-power-iteration), applied to $\mathbf{B}^2$, shows that the iterates are well defined and converge to a unit vector $\mathbf{w}$ in the eigenspace associated with $\rho^2$:
$\mathbf{B}^2\mathbf{w}=\rho^2\mathbf{w}.$

Furthermore,
$$
\begin{aligned}
\|\mathbf{B}\mathbf{w}\|^2
&=\mathbf{w}^\top\mathbf{B}^\top\mathbf{B}\mathbf{w}\\
&=\mathbf{w}^\top\mathbf{B}^2\mathbf{w}\\
&=\rho^2\|\mathbf{w}\|^2\\
&=\rho^2 .
\end{aligned}
$$

Thus $\rho=\|\mathbf{B}\mathbf{w}\|$.
For either choice of the sign,
$$
\begin{aligned}
\mathbf{B}\mathbf{u}_\pm
&=\frac12\left(\mathbf{B}\mathbf{w}\pm\frac{\mathbf{B}^2\mathbf{w}}{\rho}\right)\\
&=\frac12\bigl(\mathbf{B}\mathbf{w}\pm\rho\,\mathbf{w}\bigr)\\
&=\pm\rho\,\frac12\left(\mathbf{w}\pm\frac{\mathbf{B}\mathbf{w}}{\rho}\right)\\
&=\pm\rho\,\mathbf{u}_\pm .
\end{aligned}
$$

Also, 
$\textcolor{#2563eb}{\mathbf{u}_+}+\textcolor{#ea580c}{\mathbf{u}_-}=\mathbf{w}\neq0,$
so the two vectors cannot both be zero. More precisely,
$$
\textcolor{#2563eb}{\mathbf{u}_+}^\top\textcolor{#ea580c}{\mathbf{u}_-}=\frac14\left(\|\mathbf{w}\|^2-\frac{\|\mathbf{B}\mathbf{w}\|^2}{\rho^2}\right)=0,
$$
and
$\|\textcolor{#2563eb}{\mathbf{u}_+}\|^2+\|\textcolor{#ea580c}{\mathbf{u}_-}\|^2=1.$
Therefore,
$\max\bigl\{\|\mathbf{u}_+\|,\|\mathbf{u}_-\|\bigr\}\ge 1/\sqrt{2}.$
Choosing the vector with larger norm avoids normalizing a potentially very small component.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

This construction does not require forming $\mathbf{B}^2$. Its action is obtained through $\mathbf{B}^2\mathbf{x}=\mathbf{B}(\mathbf{B}\mathbf{x})$, using two applications of $\mathbf{B}$. Squaring removes the distinction between the eigenvalues $+\rho$ and $-\rho$ during power iteration. The subsequent construction of $\mathbf{u}_+$ and $\mathbf{u}_-$ separates their eigenspaces.

### 3.6 Corollary 2 [Recovery of a complete eigenbasis]

For every real symmetric matrix $\mathbf{A}$, orthogonal deflation combined with [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) recovers a complete orthonormal eigenbasis, including repeated and zero eigenvalues, when each inner power iteration is taken to its limit. The procedure is as follows.

1. Initialize $\mathbf{Q}_0\in\mathbb{R}^{d\times0}$ as an empty matrix, that is, a matrix with no columns, so that $\mathbf{Q}_0\mathbf{Q}_0^\top=0_{d\times d}$ (each entry is an empty sum).
2. At stage $j\in\{1,\dots,d\}$, define $\mathbf{P}_{j-1}=\mathbb{I}_d-\mathbf{Q}_{j-1}\mathbf{Q}_{j-1}^\top$ and $\mathbf{B}_j=\mathbf{P}_{j-1}\mathbf{A}\mathbf{P}_{j-1}$. Choose $\mathbf{z}$ such that $\mathbf{P}_{j-1}\mathbf{z}\neq0$, and set $\mathbf{w}_0=\mathbf{P}_{j-1}\mathbf{z}/\|\mathbf{P}_{j-1}\mathbf{z}\|$. For example, a fresh draw $\mathbf{z}\sim\mathcal{N}(0,\mathbb{I}_d)$ gives a nonzero denominator with probability one.
3. If $\mathbf{B}_j\mathbf{w}_0=0$, set $\mathbf{q}_j=\mathbf{w}_0$ and $\lambda_j=0$. Otherwise, apply [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) to $\mathbf{B}_j$, choose the larger of the two sign-separated components in norm, and normalize it to obtain $\mathbf{q}_j$. Recover the eigenvalue as $\lambda_j=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$.
4. Append the new eigenvector, $\mathbf{Q}_j=[\mathbf{Q}_{j-1},\mathbf{q}_j]$, and continue to the next stage.

#### Proof

The proof is by induction on the stage $j$. Assume that the columns of $\mathbf{Q}_{j-1}$ are orthonormal eigenvectors of $\mathbf{A}$, which holds trivially for $j=1$, since $\mathbf{Q}_0$ has no columns. Let

$$
S_j=\{\mathbf{x}\in\mathbb{R}^d:\mathbf{Q}_{j-1}^\top\mathbf{x}=0\}
$$

be the orthogonal complement of the eigenvectors already found, with $S_1=\mathbb{R}^d$. We show that $\mathbf{q}_j$ is a unit eigenvector of $\mathbf{A}$ that lies in $S_j$, so that the columns of $\mathbf{Q}_j=[\mathbf{Q}_{j-1},\mathbf{q}_j]$ are again orthonormal eigenvectors of $\mathbf{A}$.

**On $S_j$, $\mathbf{B}_j$ acts as $\mathbf{A}$.** By [Lemma 1](#34-lemma-1-orthogonal-deflation), $\mathbf{B}_j=\mathbf{A}-\mathbf{Q}_{j-1}\mathbf{D}_{j-1}\mathbf{Q}_{j-1}^\top$, where $\mathbf{D}_{j-1}$ is the diagonal matrix of the eigenvalues already found. Hence $\mathbf{B}_j\mathbf{x}=\mathbf{A}\mathbf{x}$ for every $\mathbf{x}\in S_j$.

**$\mathbf{B}_j$ maps every vector into $S_j$.** Since $\mathbf{Q}_{j-1}^\top\mathbf{Q}_{j-1}=\mathbb{I}_{j-1}$,
$\mathbf{Q}_{j-1}^\top\mathbf{P}_{j-1}=\mathbf{Q}_{j-1}^\top-\mathbf{Q}_{j-1}^\top\mathbf{Q}_{j-1}\mathbf{Q}_{j-1}^\top=0,$
so $\mathbf{P}_{j-1}\mathbf{y}\in S_j$ for every $\mathbf{y}\in\mathbb{R}^d$. In particular, $\mathbf{B}_j\mathbf{x}=\mathbf{P}_{j-1}(\mathbf{A}\mathbf{P}_{j-1}\mathbf{x})\in S_j$ for every $\mathbf{x}$.

**Every vector computed at stage $j$ lies in $S_j$.** The initialization $\mathbf{w}_0=\mathbf{P}_{j-1}\mathbf{z}/\|\mathbf{P}_{j-1}\mathbf{z}\|$ does, by the previous step. Each iterate $\mathbf{w}_{k+1}$ is a multiple of $\mathbf{B}_j(\mathbf{B}_j\mathbf{w}_k)$, so it does too. The limit $\mathbf{w}$ does, because a subspace is closed. Finally, $\mathbf{u}_\pm=\frac12(\mathbf{w}\pm\mathbf{B}_j\mathbf{w}/\rho)$ are linear combinations of the two vectors $\mathbf{w}$ and $\mathbf{B}_j\mathbf{w}$ of $S_j$.

**Conclusion.** The matrix $\mathbf{B}_j$ is symmetric, so [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) applies to it. If $\mathbf{B}_j\mathbf{w}_0\neq0$, the vector $\mathbf{q}_j$ is a normalized $\mathbf{u}_+$ or $\mathbf{u}_-$. It lies in $S_j$, that is, $\mathbf{Q}_{j-1}^\top\mathbf{q}_j=0$, and it satisfies $\mathbf{B}_j\mathbf{q}_j=\mu\,\mathbf{q}_j$ with $\mu=\rho$ or $\mu=-\rho$. The first step turns this into
$\mathbf{A}\mathbf{q}_j=\mathbf{B}_j\mathbf{q}_j=\mu\,\mathbf{q}_j,$
so $\mathbf{q}_j$ is an eigenvector of $\mathbf{A}$, and $\lambda_j=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j=\mu$. If $\mathbf{B}_j\mathbf{w}_0=0$, then $\mathbf{q}_j=\mathbf{w}_0$ lies in $S_j$, and the first step gives
$\mathbf{A}\mathbf{w}_0=\mathbf{B}_j\mathbf{w}_0=0,$
so $\mathbf{w}_0$ is an eigenvector of $\mathbf{A}$ with the eigenvalue $\lambda_j=0$.

Thus each stage adds one unit eigenvector orthogonal to all previously stored eigenvectors.
After $d$ stages,
$\mathbf{Q}_d^\top\mathbf{Q}_d=\mathbb{I}_d$ and
$\mathbf{A}\mathbf{Q}_d=\mathbf{Q}_d\operatorname{diag}(\lambda_1,\dots,\lambda_d).$
Since the square matrix $\mathbf{Q}_d$ is orthogonal, $\mathbf{Q}_d\mathbf{Q}_d^\top=\mathbb{I}_d$, and hence
$\mathbf{A}=\mathbf{Q}_d\operatorname{diag}(\lambda_1,\dots,\lambda_d)\mathbf{Q}_d^\top.$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

Any admissible nonzero initialization suffices to recover some eigenpair at each stage. With fresh Gaussian initializations, the procedure selects an eigenvalue of largest remaining absolute value with probability one. Consequently, the eigenvalues are obtained in nonincreasing absolute value, with arbitrary ordering within ties. Algebraic ordering can be obtained by sorting afterward.

The exact test $\mathbf{B}_j\mathbf{w}_0=0$ identifies a zero eigenvector. It does not require concluding that the entire remaining operator is zero.

[Figure 2](#figure-bipartite-graph) below shows the adjacency matrix of a bipartite graph, whose nonzero eigenvalues come in opposite pairs and whose eigenvalue $0$ is repeated. [Figure 3](#figure-complete-eigenbasis) runs this procedure on it and compares it with deflation combined with ordinary power iteration, with and without a shift.

<figure class="article-figure article-figure-wide" id="figure-bipartite-graph">
  <img src="/images/blog/matrix-vector-products/bipartite-graph.svg" width="1400" height="460" alt="Three panels. Left: a bipartite graph with 30 nodes on the left and 20 on the right, each side split into two communities drawn in teal and gold, with most edges joining nodes of the same community and a few grey dashed edges across communities. Middle: its 50 by 50 adjacency matrix, with zero diagonal blocks and dense blocks where the communities match. Right: its 50 eigenvalues in decreasing order, symmetric about zero, with ten zeros in the middle." loading="lazy" />
  <figcaption><strong>Figure 2.</strong> The bipartite graph used in Figures <a href="#figure-complete-eigenbasis">3</a> and <a href="#figure-lanczos-vs-deflation">7</a>. (a) The graph has 30 nodes on one side and 20 on the other, each side split into two planted communities (<span style="color:#0d9488">teal</span> and <span style="color:#ca8a04">gold</span>). Each of the $30\times20$ possible edges between the two sides is present with probability 0.45 when the communities match and 0.08 otherwise, and no edge joins two nodes of the same side. The draw has 148 edges, 128 within communities (colored) and 20 across (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>). (b) Its adjacency matrix $\mathbf W$, with the nodes ordered side by side and community by community. The diagonal blocks are zero, and the off-diagonal blocks are the $30\times20$ biadjacency matrix $\mathbf B$ and its transpose, so a product $\mathbf W\mathbf x$ is one pass over the 148 edges. (c) The 50 eigenvalues of $\mathbf W$ in decreasing order. They are symmetric about $0$. Writing a vector of $\mathbb R^{50}$ as $(\mathbf x,\mathbf y)$ with $\mathbf x\in\mathbb R^{30}$ and $\mathbf y\in\mathbb R^{20}$, $\mathbf W(\mathbf x,\mathbf y)=(\mathbf B\mathbf y,\mathbf B^\top\mathbf x)$, so a pair of singular vectors with $\mathbf B\mathbf v=\sigma\mathbf u$ and $\mathbf B^\top\mathbf u=\sigma\mathbf v$ gives the eigenvector $(\mathbf u,\mathbf v)$ for $\sigma$ and $(\mathbf u,-\mathbf v)$ for $-\sigma$. The 20 singular values of $\mathbf B$ give the 40 nonzero eigenvalues, and $0$ is repeated $30-20=10$ times.</figcaption>
</figure>

<figure class="article-figure article-figure-wide" id="figure-complete-eigenbasis">
  <img src="/images/blog/matrix-vector-products/complete-eigenbasis.svg" width="1400" height="460" alt="Three panels on the adjacency matrix of a bipartite graph with 50 nodes. Left: the eigenvalue found at each of the 50 deflation stages; Algorithm 1 alternates between opposite eigenvalues of decreasing magnitude and ends with ten zeros, while shifted deflation descends from the largest to the smallest eigenvalue. Middle: eigenpairs found against matrix-vector products on a log scale; Algorithm 1 finishes after about fifteen thousand products, shifted deflation after about sixty thousand, and plain deflation never finds one. Right: residuals, eigenvalue errors and inner products of Algorithm 1, all below one trillionth." loading="lazy" />
  <figcaption><strong>Figure 3.</strong> A complete eigenbasis of the adjacency matrix of the bipartite graph of <a href="#figure-bipartite-graph">Figure 2</a>, applied through its edge list (<a href="#22-other-matrices-known-through-their-products">Section 2.2</a>). Because the graph is bipartite, every nonzero eigenvalue $\sigma$ comes with $-\sigma$, so plain power iteration meets the tie of <a href="#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude">Remark 5</a> at once, and $0$ is an eigenvalue of multiplicity 10. Every stage starts from a fresh Gaussian vector and, as in <a href="#36-corollary-2-recovery-of-a-complete-eigenbasis">Corollary 2</a>, is taken to its limit: it runs until the residual of its candidate for the deflated matrix $\mathbf B_j$ is below $10^{-13}$. (a) Eigenvalue found at each stage. <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> (<span style="color:#2563eb">blue</span>) finds the eigenvalues in nonincreasing absolute value, the two members of each pair at consecutive stages, and the ten zero eigenvalues last. Power iteration on $\mathbf A-\lambda_{\min}(\mathbf A)\,\mathbb I_d$ with deflation (<span style="color:#7c3aed">purple</span>; <a href="#33-corollary-1-recovering-algebraic-extrema-by-shifting">Corollary 1</a> and <a href="#34-lemma-1-orthogonal-deflation">Lemma 1</a>) finds them in decreasing order. <span style="color:#7d8783">Grey lines</span> mark the 41 distinct eigenvalues. (b) Number of eigenpairs found against the number of products. <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> uses 15,296 products in all, two for each zero eigenvalue. The shifted iteration uses 61,453, even with the most favorable shift $\lambda_{\min}(\mathbf A)=-6.48$: adding $6.48$ to every eigenvalue brings the ratios that set the speed of convergence (<a href="#32-theorem-1-rate-of-convergence">Theorem 1</a>) closer to one, and each zero eigenvalue takes a full stage. Plain deflated power iteration (<span style="color:#ea580c">orange</span>) never completes its first stage, and after 20,000 products its residual is still above 1.9. (c) Accuracy of the eigenbasis of <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a>, far below the tolerance $10^{-10}\|\mathbf A\|$ (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>): residuals (<span style="color:#2563eb">blue</span>) below $3.3\cdot10^{-13}$, eigenvalues within $6.3\cdot10^{-15}$ of the spectrum (<span style="color:#7c3aed">purple</span>; exact zeros drawn at $10^{-17}$), inner products between distinct eigenvectors (<span style="color:#7d8783">grey</span>) below $7.3\cdot10^{-14}$, and $\|\mathbf A-\mathbf Q_{50}\operatorname{diag}(\lambda_1,\dots,\lambda_{50})\mathbf Q_{50}^\top\|=2.2\cdot10^{-13}$.</figcaption>
</figure>

### 3.7 Lemma 2 [Residual certificate for an approximate eigenpair]

For any unit vector $\mathbf{q}$, define 
$\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$
and
$\mathbf{r}
=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q} 
%=(\mathbf{A}-\widehat{\lambda}\mathbb{I})\mathbf{q}
.$
Then $\min_{\lambda\in\Lambda(\mathbf{A})}|\widehat{\lambda}-\lambda|\le\|\mathbf{r}\|.$

#### Proof

Expand $\mathbf{q}$ in an orthonormal eigenbasis $\mathbf{u}_1,\dots,\mathbf{u}_d$ of $\mathbf{A}$:
$$
\mathbf{q}=\sum_{i=1}^dc_i\mathbf{u}_i,
\qquad
\mathbf{A}\mathbf{u}_i=\lambda_i\mathbf{u}_i,
\qquad
\sum_{i=1}^dc_i^2=1 .
$$

Then $\mathbf{r}=\sum_{i=1}^dc_i(\lambda_i-\widehat{\lambda})\mathbf{u}_i,$
and therefore
$$
\begin{aligned}
\|\mathbf{r}\|^2
&=\sum_{i=1}^d\textcolor{#2563eb}{c_i^2}\,\textcolor{#ea580c}{(\lambda_i-\widehat{\lambda})^2}\\
&\ge\textcolor{#ea580c}{\left(\min_{1\le i\le d}|\lambda_i-\widehat{\lambda}|^2\right)}\textcolor{#2563eb}{\sum_{i=1}^dc_i^2}\\
&=\textcolor{#ea580c}{\min_{1\le i\le d}|\lambda_i-\widehat{\lambda}|^2} .
\end{aligned}
$$

The squared residual is thus an average of the <span style="color:#ea580c">squared distances</span> from $\widehat\lambda$ to the eigenvalues, with <span style="color:#2563eb">weights</span> $c_i^2$ that sum to one, so it is at least the smallest of them. Taking square roots gives the result.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

#### Remark 10 [A computable bound on an unknown distance]

The residual $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$ is computable with one product, and it bounds an unknown quantity: the distance from the estimate $\widehat\lambda=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ to the nearest true eigenvalue. If $\mathbf{q}$ is exactly an eigenvector, the residual is zero, and the bound says that $\widehat\lambda$ is exactly an eigenvalue. The lemma has three consequences.

**A stopping rule with a guarantee.** The eigenvalues of $\mathbf{A}$ are unknown, so the error of $\widehat\lambda$ cannot be measured directly. The lemma converts the computable residual into a guarantee: the interval $[\widehat\lambda-\|\mathbf{r}\|,\widehat\lambda+\|\mathbf{r}\|]$ contains an eigenvalue of $\mathbf{A}$. Stopping when $\|\mathbf{r}\|\le\varepsilon$ therefore ensures that $\widehat\lambda$ is within $\varepsilon$ of an eigenvalue, and this is how [Algorithm 1](#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation) and [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks) stop. [Lemma 7](#56-lemma-7-residual-certificate-without-symmetry) is the counterpart for nonsymmetric matrices.

**Independence from the method.** The lemma involves only $\mathbf{A}$, $\mathbf{q}$ and one product, not the way $\mathbf{q}$ was obtained. It applies equally to an unconverged power iterate, an approximate eigenvector produced by the [Lanczos method of Section 4](#4-the-lanczos-method), or a vector from a late deflation stage whose earlier eigenvectors carry rounding errors. This is why the algorithms compute the residual with the original matrix $\mathbf{A}$ rather than with the matrix iterated at stage $j$ of the deflation in [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis), $\mathbf{B}_j=\mathbf{P}_{j-1}\mathbf{A}\mathbf{P}_{j-1}$, where $\mathbf{P}_{j-1}=\mathbb{I}_d-\mathbf{Q}_{j-1}\mathbf{Q}_{j-1}^\top$ removes the directions of the eigenvectors $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$ already found. Errors in these eigenvectors change $\mathbf{B}_j$, but not the certificate.

**What it does not say.** It certifies some eigenvalue, not a particular one, such as the largest remaining one. Nor does a small residual alone make $\mathbf{q}$ close to an eigenvector. The expansion in the proof above also bounds the angle between $\mathbf{q}$ and an eigenspace. Fix an eigenvalue $\lambda$, and let $\delta=\min\{|\mu-\widehat\lambda|:\mu\in\Lambda(\mathbf{A})\setminus\{\lambda\}\}$ be the distance from $\widehat\lambda$ to the other eigenvalues, assumed positive. Then
$$
\begin{aligned}
\|\mathbf{r}\|^2
&=\sum_{i=1}^dc_i^2(\lambda_i-\widehat\lambda)^2\\
&\ge\sum_{i:\,\lambda_i\neq\lambda}c_i^2(\lambda_i-\widehat\lambda)^2\\
&\ge\delta^2\sum_{i:\,\lambda_i\neq\lambda}c_i^2\\
&=\delta^2\,\|\mathbf{q}-\Pi_\lambda\mathbf{q}\|^2\\
&=\delta^2\sin^2\angle(\mathbf{q},E_\lambda).
\end{aligned}
$$
The first line is the expansion of the proof. The second drops the terms with $\lambda_i=\lambda$, which are nonnegative. The third uses $|\lambda_i-\widehat\lambda|\ge\delta$ for every $\lambda_i\neq\lambda$, which is the definition of $\delta$. For the fourth, the $\mathbf{u}_i$ with $\lambda_i=\lambda$ form an orthonormal basis of $E_\lambda$, so $\Pi_\lambda\mathbf{q}=\sum_{i:\,\lambda_i=\lambda}c_i\mathbf{u}_i$ and $\mathbf{q}-\Pi_\lambda\mathbf{q}=\sum_{i:\,\lambda_i\neq\lambda}c_i\mathbf{u}_i$, whose squared norm is $\sum_{i:\,\lambda_i\neq\lambda}c_i^2$ by Pythagoras. The last is the sine of the angle between the unit vector $\mathbf{q}$ and $E_\lambda$, as in [Theorem 1](#32-theorem-1-rate-of-convergence): the side of the right triangle opposite the angle, over the hypotenuse $\|\mathbf{q}\|=1$. Dividing by $\delta^2$ and taking square roots gives
$$
\sin\angle(\mathbf{q},E_\lambda)
\le\frac{\|\mathbf{r}\|}{\delta} 
= \frac{\|\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|}{\min\{|\mu-\widehat\lambda|:\mu\in\Lambda(\mathbf{A})\setminus\{\lambda\}\}}.
%= \frac{\|(\mathbf{A} - \mathbf{q}^\top\mathbf{A}\mathbf{q} \mathbb{I}_d) \mathbf{q}\|}{\min\{|\mu-\mathbf{q}^\top\mathbf{A}\mathbf{q}|:\mu\in\Lambda(\mathbf{A})\setminus\{\lambda\}\}}.
$$

The eigenvector is accurate only when $\widehat\lambda = \mathbf{q}^\top\mathbf{A}\mathbf{q}$ is well separated from the rest of the spectrum. When another eigenvalue is close, a small residual still allows $\mathbf{q}$ to mix the two eigenvectors.

For example, take $\mathbf{A}=\operatorname{diag}(1,3)$ and the unit vector $\mathbf{q}=[\cos(t),\sin(t)]^\top$, at the angle $t$ from the eigenvector $[1,0]^\top$ of $1$. Then
$$
\begin{aligned}
&\widehat\lambda=\cos^2(t)+3\sin^2(t)=1+2\sin^2(t),\\
&\mathbf{r} = [\cos(t) - \widehat\lambda\cos(t), 3\sin(t) - \widehat\lambda\sin(t)]^\top = [-2\sin^2(t)\cos(t),2 \sin(t) - 2\sin^3(t)]^\top = 2\sin(t)\cos(t)\,[-\sin(t),\cos(t)]^\top\\
&\|\mathbf{r}\|= 2|\sin(t)\cos(t)| \sqrt{\sin^2(t)+\cos^2(t)} =|\sin(2t)| .
\end{aligned}
$$
For $\sin(t)=0.1$, $\widehat\lambda=1.02$ and $\|\mathbf{r}\|\approx0.199$, so the lemma guarantees an eigenvalue in $[0.821,1.219]$, and $1$ is there. The eigenvector bound gives $\sin(t)\le0.199/1.98\approx0.1005$, close to the true $0.1$. The actual eigenvalue error, $0.02$, is ten times smaller than the residual: it is quadratic in the angle $t$, while the residual is linear in it.

This holds in general. Since the $c_i^2$ sum to one, $\widehat\lambda=\sum_ic_i^2\lambda_i$, so $\widehat\lambda-\lambda=\sum_ic_i^2(\lambda_i-\lambda)$, where the terms with $\lambda_i=\lambda$ vanish. Hence
$$
|\widehat\lambda-\lambda|
\le\max_{1\le i\le d}|\lambda_i-\lambda|\sum_{i:\,\lambda_i\neq\lambda}c_i^2
=\max_{1\le i\le d}|\lambda_i-\lambda|\,\sin^2\angle(\mathbf{q},E_\lambda)
\le\max_{1\le i\le d}|\lambda_i-\lambda|\,\frac{\|\mathbf{r}\|^2}{\delta^2},
$$
where the equality combines the last two lines of the chain above and the last step is the eigenvector bound. In the example, the right-hand side is $2\cdot0.199^2/1.98^2\approx0.0202$, against the true error $0.02$. Once the residual is small compared with the gap $\delta$, the eigenvalue error is of the order of $\|\mathbf{r}\|^2$, far below the certificate $\|\mathbf{r}\|$. The certificate is therefore valid but pessimistic: it guarantees roughly the square root of the accuracy actually reached, as [Figure 4](#figure-residual-certificate) shows.

#### Remark 11 [Any scalar, and the best one]

The proof never uses the definition of $\widehat\lambda$. Replacing $\widehat\lambda$ by any scalar $\mu$, the same expansion gives $\|\mathbf{A}\mathbf{q}-\mu\mathbf{q}\|^2=\sum_{i=1}^dc_i^2(\lambda_i-\mu)^2\ge\min_i(\lambda_i-\mu)^2$, that is,

$$
\min_{\lambda\in\Lambda(\mathbf{A})}|\mu-\lambda|\le\|\mathbf{A}\mathbf{q}-\mu\mathbf{q}\|
\qquad\text{for every unit vector }\mathbf{q}\text{ and every scalar }\mu .
$$

No relation between $\mathbf{q}$ and $\mu$ is required. They can be chosen independently, and a poor match only makes the bound loose, never wrong. For $\mathbf{A}=\operatorname{diag}(1,3)$, $\mathbf{q}=[1,0]^\top$ and $\mu=10$, it reads $7\le9$. Hence the certificate applies to an eigenvalue estimate computed by any means, as long as it is paired with a unit vector. The Rayleigh quotient is the best choice of $\mu$. Indeed, expanding the square with $\|\mathbf{q}\|=1$ and $\widehat\lambda=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ gives

$$
\begin{aligned}
\|\mathbf{A}\mathbf{q}-\mu\mathbf{q}\|^2
&=\|\mathbf{A}\mathbf{q}\|^2-2\mu\widehat\lambda+\mu^2\\
&=\textcolor{#ea580c}{(\mu-\widehat\lambda)^2}+\textcolor{#2563eb}{\|\mathbf{A}\mathbf{q}\|^2-\widehat\lambda^2}\\
&=\textcolor{#ea580c}{(\mu-\widehat\lambda)^2}+\textcolor{#2563eb}{\|\mathbf{r}\|^2},
\end{aligned}
$$

where the last line is the second one at $\mu=\widehat\lambda$. The <span style="color:#2563eb">blue term</span> does not depend on $\mu$, so the certificate is smallest at $\mu=\widehat\lambda$, where it equals the residual of [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair), and the <span style="color:#ea580c">orange term</span> is the price of any other estimate. As a function of $\mu$, the certificate $\sqrt{(\mu-\widehat\lambda)^2+\|\mathbf{r}\|^2}$ is a hyperbola centered at the Rayleigh quotient $\widehat\lambda$, and it lies above the distance $\min_{\lambda\in\Lambda(\mathbf{A})}|\mu-\lambda|$ from $\mu$ to the spectrum, as panel (a) of [Figure 4](#figure-residual-certificate) below shows.
This distance is the pointwise minimum of the functions $\mu\mapsto|\mu-\lambda|$, one for each eigenvalue $\lambda$, whose graphs are V shapes with their vertex on the axis at $\lambda$. Its graph is therefore the lower envelope of these V shapes: a zigzag, piecewise linear with slopes $\pm1$, that vanishes at each eigenvalue. Between two consecutive eigenvalues $\lambda<\lambda'$, the nearest eigenvalue is $\lambda$ up to the midpoint $(\lambda+\lambda')/2$ and $\lambda'$ after it, so the distance equals $\mu-\lambda$, which rises with slope $1$, and then $\lambda'-\mu$, which falls with slope $-1$. Each gap of the spectrum thus carries a triangle whose height $(\lambda'-\lambda)/2$ is half the gap, and beyond the extreme eigenvalues the distance grows along half-lines of slope $\pm1$. The hyperbola has the same slopes far from its center, since it approaches $|\mu-\widehat\lambda|$ as $|\mu-\widehat\lambda|$ grows, and it is lowest at $\mu=\widehat\lambda$, where its height is $\|\mathbf{r}\|$. The bound says that the hyperbola never goes below this envelope. At its lowest point, this is [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair): the height $\|\mathbf{r}\|$ is at least the distance from $\widehat\lambda$ to the spectrum, so a small residual places $\widehat\lambda$ near a zero of the envelope, that is, near an eigenvalue.

The remark has three consequences.

**Certifying an estimate obtained elsewhere.** An eigenvalue estimate $\mu$ that does not come from a Rayleigh quotient, such as a value known from theory, the value $\alpha-\mu_{\max}$ recovered from a shifted run in [Corollary 1](#33-corollary-1-recovering-algebraic-extrema-by-shifting), or the value $\pm\rho$ of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation), is certified by pairing it with a unit vector $\mathbf{q}$ and computing one product: some eigenvalue of $\mathbf{A}$ lies within $\|\mathbf{A}\mathbf{q}-\mu\mathbf{q}\|=\sqrt{(\mu-\widehat\lambda)^2+\|\mathbf{r}\|^2}$ of $\mu$, $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat\lambda\mathbf{q}$.

**Reporting the Rayleigh quotient.** For a given $\mathbf{q}$, replacing $\mu$ by $\widehat\lambda$ never weakens the certificate, and strictly improves it when $\mu\neq\widehat\lambda$. It costs no extra product, since $\widehat\lambda=\mathbf{q}^\top(\mathbf{A}\mathbf{q})$ reuses the product already needed for the residual. This is why [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis) recovers each eigenvalue as $\lambda_j=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$ rather than reporting $\pm\rho$: the two coincide in exact arithmetic, but in practice $\mathbf{q}_j$ is only approximately an eigenvector. Geometrically, $\mathbf{q}^\top\mathbf{r}=\widehat\lambda-\widehat\lambda=0$, so $\widehat\lambda\mathbf{q}$ is the orthogonal projection of $\mathbf{A}\mathbf{q}$ onto the line spanned by $\mathbf{q}$, the point of that line closest to $\mathbf{A}\mathbf{q}$. The certificate measures how well the pair $(\mu,\mathbf{q})$ fits together, not the error of $\mu$ alone. In the example of [Remark 10](#remark-10-a-computable-bound-on-an-unknown-distance), the exact eigenvalue $\mu=1$ receives the certificate $\|\mathbf{A}\mathbf{q}-\mathbf{q}\|=\|[0,\,0.2]^\top\|=0.2$, slightly worse than the $0.199$ of $\widehat\lambda=1.02$, because $\mathbf{q}$ is not its eigenvector.

**Testing a given value.** For a fixed $\mu$, every unit vector $\mathbf{q}$ gives an upper bound on the distance from $\mu$ to the spectrum, so a single well-chosen $\mathbf{q}$ suffices to show that $\mu$ is nearly an eigenvalue. With $\mu=0$, the bound reads $\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda|\le\|\mathbf{A}\mathbf{q}\|$.
Any unit vector $\mathbf{q}$ that $\mathbf{A}$ nearly annihilates proves that $\mathbf{A}$ has an eigenvalue near zero, that is, that $\mathbf{A}$ is nearly singular. Combined with $\|\mathbf{A}\|=\max_{\|\mathbf z\|=1}\|\mathbf A\mathbf z\|\ge\|\mathbf{A}\mathbf{q}'\|$ for any other unit vector $\mathbf{q}'$, it gives a lower bound on the [condition number](#remark-3-the-convergence-factor) computed from two products,
$\kappa(\mathbf{A})
=\max_{\lambda\in\Lambda(\mathbf{A})}|\lambda|/\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda|
=\textcolor{#2563eb}{\|\mathbf{A}\|} / \textcolor{#ea580c}{\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda|}
\ge\textcolor{#2563eb}{\|\mathbf{A}\mathbf{q}'\|}/\textcolor{#ea580c}{\|\mathbf{A}\mathbf{q}\|}$. The inequality bounds the two parts of the fraction separately. The <span style="color:#2563eb">numerator</span> can only decrease, from $\|\mathbf{A}\|$ to $\|\mathbf{A}\mathbf{q}'\|$. The <span style="color:#ea580c">denominator</span> can only increase, from $\min_{\lambda}|\lambda|$ to $\|\mathbf{A}\mathbf{q}\|$, by the bound with $\mu=0$. Both changes make the fraction smaller. A vector $\mathbf{q}'$ that $\mathbf{A}$ stretches and a vector $\mathbf{q}$ that it nearly annihilates therefore give a large ratio, which proves that $\mathbf{A}$ is ill conditioned.

<figure class="article-figure article-figure-wide" id="figure-residual-certificate">
  <img src="/images/blog/matrix-vector-products/residual-certificate.svg" width="1400" height="460" alt="Three panels on a 60 by 60 sample covariance matrix. Left: for several power-iteration iterates, the norm of A v minus mu v as a function of mu is a hyperbola lying above the saw-tooth distance from mu to the spectrum, with its minimum at the Rayleigh quotient. Middle: the residual decreases geometrically and the distance to the nearest eigenvalue twice as fast, down to rounding level. Right: a log-log scatter of distance against residual for many iterates, all below the diagonal." loading="lazy" />
  <figcaption><strong>Figure 4.</strong> The residual certificate on the sample covariance matrix $\mathbf A=\frac1n\mathbf X^\top\mathbf X$ of $n=400$ centered observations of $d=60$ variables, applied as $\mathbf X^\top(\mathbf X\mathbf v)/n$ (<a href="#22-other-matrices-known-through-their-products">Section 2.2</a>). Its three largest eigenvalues, the variances along the first three principal components, are 9.72, 4.92 and 3.20, and the other 57 lie in $[0.40,1.80]$. (a) For the iterates $\mathbf v_k$ of power iteration from one Gaussian start, with Rayleigh quotients $\widehat\lambda_k=\mathbf v_k^\top\mathbf A\mathbf v_k$ and residuals $\mathbf r_k=\mathbf A\mathbf v_k-\widehat\lambda_k\mathbf v_k$, the certificate $\mu\mapsto\|\mathbf A\mathbf v_k-\mu\mathbf v_k\|=\sqrt{(\mu-\widehat\lambda_k)^2+\|\mathbf r_k\|^2}$ of <a href="#remark-11-any-scalar-and-the-best-one">Remark 11</a> (<span style="background:linear-gradient(90deg,#b0dd2f,#481d6f);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span> as $k$ grows) lies above the distance $\min_{\lambda\in\Lambda(\mathbf A)}|\mu-\lambda|$ from $\mu$ to the spectrum (<span style="color:#27322f">black</span>, with the eigenvalues marked in <span style="color:#ea580c">orange</span> on the axis) for every scalar $\mu$. This distance is the lower envelope of the V shapes $|\mu-\lambda|$ (<a href="#remark-11-any-scalar-and-the-best-one">Remark 11</a>), a zigzag with one triangle over each gap of the spectrum, of height half the gap. The tallest, of height $2.40$, spans the gap from $4.92$ to $9.72$, and those over the $57$ closely spaced eigenvalues in $[0.40,1.80]$ are too small to see. The minimum of each hyperbola, marked by a dot, is attained at $\mu=\widehat\lambda_k$, whose value is written under the axis at the foot of the dashed line, and equals the residual norm $\|\mathbf r_k\|$ of <a href="#37-lemma-2-residual-certificate-for-an-approximate-eigenpair">Lemma 2</a>. (b) The same quantities along the whole run rather than for a few iterates. Each curve is the median over 200 Gaussian starts of a quantity computed from $\mathbf v_k$, and its band contains the middle half of the runs. The residual $\|\mathbf r_k\|$ (<span style="color:#2563eb">blue</span>) shrinks by the factor $\lambda_2/\lambda_1\approx0.51$ per product, and the distance from $\widehat\lambda_k$ to the nearest eigenvalue (<span style="color:#ea580c">orange</span>) by its square (<a href="#32-theorem-1-rate-of-convergence">Theorem 1</a>), until rounding errors of order $10^{-15}$ stop it. The certificate is valid but pessimistic: it bounds the error by the residual, whereas the error of a Rayleigh quotient is of the order of the squared residual (<a href="#remark-10-a-computable-bound-on-an-unknown-distance">Remark 10</a>). After 20 products, the median residual is $1.2\cdot10^{-5}$ and the median distance $3.2\cdot10^{-11}$, about its square divided by the gap $4.81$ between the two largest eigenvalues. In the first steps, $\widehat\lambda_k$ lies among the small eigenvalues, and its distance to the nearest of them (<span style="color:#ea580c">orange</span>) is much smaller than its distance to $\lambda_{\max}(\mathbf A)$ (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dotted;text-decoration-thickness:0.12em;text-underline-offset:0.2em">dotted</span>): the certificate concerns some eigenvalue, not necessarily the largest. (c) The pairs formed by the residual and the distance, one point for each iterate $\mathbf v_k$, $0\le k\le39$, of 50 runs, colored by $k$ from <span style="background:linear-gradient(90deg,#fde725,#440154);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span>. None lies above the diagonal, as <a href="#37-lemma-2-residual-certificate-for-an-approximate-eigenpair">Lemma 2</a> requires, and over all 200 runs and 40 steps the largest ratio of the distance to the residual is 0.995. Distances below $10^{-15}$, at rounding level, are drawn at $10^{-15}$.</figcaption>
</figure>

### 3.8 Algorithm 1 [Deflated squared power iteration with sign separation]

[Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis) recovers a complete eigenbasis, but each of its stages runs the iteration of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) to its limit, which takes infinitely many products. The algorithm below is its finite version. It keeps the two loops of [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis), an outer loop over deflation stages ([Lemma 1](#34-lemma-1-orthogonal-deflation)) and an inner loop of squared power iteration on the deflated matrix $\mathbf{B}=\mathbf{P}\mathbf{A}\mathbf{P}$ of the current stage ([Proposition 2](#35-proposition-2-power-iteration-with-sign-separation)), with three changes.

* **No matrix is formed.** Neither $\mathbf{A}$, $\mathbf{P}$ nor $\mathbf{B}$ is built. A product with $\mathbf{B}$ is a product with $\mathbf{A}$ followed by the projection $\mathbf{P}\mathbf{x}=\mathbf{x}-\mathbf{Q}(\mathbf{Q}^\top\mathbf{x})$, as in the projection view of [Remark 8](#remark-8-two-views-of-the-deflated-matrix).
* **The signs are separated at every step.** Each inner step turns the current iterate into a candidate eigenvector by the construction of $\mathbf{u}_\pm$ in [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation), without waiting for the limit.
* **The inner loop stops on a certificate.** The candidate is accepted as soon as its residual with $\mathbf{A}$ is below a tolerance, which by [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) places its Rayleigh quotient near an eigenvalue of $\mathbf{A}$.

At a given deflation stage, let $\mathbf{Q}$ contain the previously computed orthonormal vectors. At the first stage, no vector has been computed yet: $\mathbf{Q}$ is the empty matrix with $d$ rows and no columns, as $\mathbf{Q}_0$ in [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis), so that $\mathbf{Q}\mathbf{Q}^\top=0_{d\times d}$ and $\mathbf{P}=\mathbb{I}_d$, and the steps below are plain squared power iteration with sign separation. Each stage ends by appending the vector it accepts to $\mathbf{Q}$ (step 4), and the next stage starts with the enlarged $\mathbf{Q}$, until the desired number of eigenpairs has been found.

Choose tolerances $\varepsilon_{\mathrm{abs}}>0$ and $\varepsilon_{\mathrm{rel}}\ge0$. Step 4 accepts a candidate when $\|\mathbf{r}\|\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$, and by [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) the accepted $\widehat\lambda$ then lies within this threshold of an eigenvalue of $\mathbf{A}$. 
* In the simplest case $\varepsilon_{\mathrm{rel}}=0$, every eigenvalue is found to within the same distance $\varepsilon_{\mathrm{abs}}$. This is the recommended choice, used in [Figures 3](#figure-complete-eigenbasis) and [7](#figure-lanczos-vs-deflation) and in the constant-threshold runs of [Figure 5](#figure-hessian-deflation) with $\varepsilon_{\mathrm{abs}}=10^{-10}\|\mathbf{A}\|$. This tolerance must be positive, since rounding errors keep the computed residual of the order of the unit roundoff times $\|\mathbf{A}\|$, never exactly zero.
* A positive $\varepsilon_{\mathrm{rel}}$ asks instead for a number of correct digits. The single tolerance $\varepsilon_{\mathrm{abs}}$ of the first case treats large and small eigenvalues unequally, since it accepts the same error on all of them. With $\varepsilon_{\mathrm{abs}}=10^{-10}$, an error of $10^{-10}$ on the eigenvalue $\lambda=10$ gives the estimate $10.000\,000\,000\,1$, which agrees with $\lambda$ in its first eleven significant digits, but the same error on $\lambda=0.001$ gives $0.001\,000\,000\,1$, which agrees in only seven. The relative part removes this imbalance by accepting an error proportional to the eigenvalue. With a negligible $\varepsilon_{\mathrm{abs}}$, the test accepts a residual up to $\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$, and $\|\mathbf{A}\mathbf{q}\|$ is nearly $|\widehat\lambda|$: since $\mathbf{A}\mathbf{q}=\widehat\lambda\mathbf{q}+\mathbf{r}$ with $\mathbf{q}^\top\mathbf{r}=0$ and $\|\mathbf{q}\|=1$, we have $\|\mathbf{A}\mathbf{q}\|^2=\widehat\lambda^2+\|\mathbf{r}\|^2$, and so $|\widehat\lambda|\le\|\mathbf{A}\mathbf{q}\|\le|\widehat\lambda|+\|\mathbf{r}\|$. With $\varepsilon_{\mathrm{rel}}=10^{-10}$, the accepted residual is thus about $10^{-10}|\widehat\lambda|$, that is, about $10^{-9}$ near $\lambda=10$ and about $10^{-13}$ near $\lambda=0.001$. By [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair), the estimates are then within these distances of $\lambda$, for instance $10.000\,000\,001$ and $0.001\,000\,000\,000\,1$, and both agree with $\lambda$ in about ten significant digits. In general, the upper bound $|\widehat\lambda|+\|\mathbf{r}\|$ on $\|\mathbf{A}\mathbf{q}\|$ turns the test $\|\mathbf{r}\|\le\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$ into $\|\mathbf{r}\|\le\varepsilon_{\mathrm{rel}}|\widehat\lambda|/(1-\varepsilon_{\mathrm{rel}})$, and [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) bounds the relative error by $|\widehat\lambda-\lambda|/|\widehat\lambda|\le\|\mathbf{r}\|/|\widehat\lambda|\le\varepsilon_{\mathrm{rel}}/(1-\varepsilon_{\mathrm{rel}})\approx\varepsilon_{\mathrm{rel}}$, whatever the eigenvalue. Using $\|\mathbf{A}\mathbf{q}\|$ rather than $|\widehat\lambda|$ in the test thus changes little, and it gives the test a geometric meaning: $\mathbf{r}$ is the part of $\mathbf{A}\mathbf{q}$ orthogonal to $\mathbf{q}$, so $\|\mathbf{r}\|/\|\mathbf{A}\mathbf{q}\|$ is the sine of the angle between $\mathbf{q}$ and $\mathbf{A}\mathbf{q}$, and the test asks that $\mathbf{A}$ map $\mathbf{q}$ nearly onto its own line. The absolute part remains as a floor. For the eigenvalue $\lambda=0$, the relative part accepts no residual at all, rounding errors never bring the residual to zero, and without $\varepsilon_{\mathrm{abs}}$ the stage would never stop. In deflation, this balance has a price. The eigenvalues are found in decreasing absolute value, so the accepted residual shrinks from stage to stage, from about $10^{-9}$ to about $10^{-13}$ in the example, and later stages can stall, for a reason explained [below](#tolerance-stall).

At each stage, start from a unit vector $\mathbf{w}$ satisfying $\mathbf{Q}^\top\mathbf{w}=0$, for instance $\mathbf{w}=\mathbf{P}\mathbf{z}/\|\mathbf{P}\mathbf{z}\|$ for a fresh draw $\mathbf{z}\sim\mathcal{N}(0,\mathbb{I}_d)$ (at the first stage, any unit vector qualifies), and repeat the following steps.

1. **<span style="color:#2563eb">First product.</span>** Compute $\textcolor{#2563eb}{\mathbf{h}_1}=\mathbf{A}\mathbf{w}$, $\mathbf{a}=\mathbf{P}\textcolor{#2563eb}{\mathbf{h}_1}$ and $\rho=\|\mathbf{a}\|$. If $\rho=0$, go to step 5.
2. **<span style="color:#ea580c">Second product.</span>** Set $\mathbf{y}=\mathbf{a}/\rho$, then compute $\textcolor{#ea580c}{\mathbf{h}_2}=\mathbf{A}\mathbf{y}$ and $\mathbf{b}=\mathbf{P}\textcolor{#ea580c}{\mathbf{h}_2}$.
3. **Candidate.** Choose the scalar sign $s=+1$ if $\mathbf{w}^\top\mathbf{y}\ge0$ and $s=-1$ if $\mathbf{w}^\top\mathbf{y}<0$. This selects whichever of $\mathbf{w}+\mathbf{y}$ and $\mathbf{w}-\mathbf{y}$ has larger norm. Form $n=\|\mathbf{w}+s\mathbf{y}\|$ and $\mathbf{q}=(\mathbf{w}+s\mathbf{y})/n$. The product of the candidate with the original matrix is available by linearity: $\mathbf{A}\mathbf{q}=(\textcolor{#2563eb}{\mathbf{h}_1}+s\textcolor{#ea580c}{\mathbf{h}_2})/n$.
4. **Check or update.** Compute $\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ and $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$. If $\|\mathbf{r}\|\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$, accept $(\widehat{\lambda},\mathbf{q})$, append $\mathbf{q}$ to $\mathbf{Q}$, and end the stage. Otherwise, update $\mathbf{w}\leftarrow\mathbf{b}/\|\mathbf{b}\|$ and return to step 1.
5. **Zero projected product.** If $\rho=0$, use $\mathbf{q}=\mathbf{w}$ as the candidate, compute $\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ and $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$, where $\mathbf{A}\mathbf{q}=\textcolor{#2563eb}{\mathbf{h}_1}$, and apply the test of step 4. Either way, the stage ends here and does not return to step 1. The vector $\mathbf{w}$ satisfies $\mathbf{Q}^\top\mathbf{w}=0$: the starting vector is chosen so, and each update $\mathbf{b}/\|\mathbf{b}\|$ is a projected vector, with $\mathbf{Q}^\top\mathbf{P}=\mathbf{Q}^\top-(\mathbf{Q}^\top\mathbf{Q})\mathbf{Q}^\top=0$ since $\mathbf{Q}^\top\mathbf{Q}=\mathbb{I}$. Hence $\mathbf{P}\mathbf{w}=\mathbf{w}-\mathbf{Q}(\mathbf{Q}^\top\mathbf{w})=\mathbf{w}$, and the condition $\rho=0$ means $\mathbf{B}\mathbf{w}=\mathbf{P}\mathbf{A}\mathbf{P}\mathbf{w}=0$, so there is no direction $\mathbf{y}$ to iterate on. If the test passes, accept $(\widehat{\lambda},\mathbf{q})$ and append $\mathbf{q}$ to $\mathbf{Q}$, as in step 4. In exact arithmetic it always passes, with $\widehat\lambda=0$, by the zero case of [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis). If it fails, which can only come from errors in the stored vectors or from rounding, report nonconvergence or restart the stage from a new $\mathbf{w}$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

#### How the steps implement the theory

Let $\mathbf{B}=\mathbf{P}\mathbf{A}\mathbf{P}$ be the deflated matrix of the current stage. The vectors $\mathbf{w}$ and $\mathbf{y}=\mathbf{P}\textcolor{#2563eb}{\mathbf{h}_1}/\|\mathbf{P}\textcolor{#2563eb}{\mathbf{h}_1}\|$ are orthogonal to the stored vectors (the columns of $\mathbf{Q}$): $\mathbf{w}$ by the choice of the start and because each update is a projected vector, and $\mathbf{y}$ because $\mathbf{Q}^\top\mathbf{P}=0$. Hence $\mathbf{P}\mathbf{w}=\mathbf{w}-\mathbf{Q}\mathbf{Q}^\top\mathbf{w}=\mathbf{w}$ and $\mathbf{P}\mathbf{y}=\mathbf{y}-\mathbf{Q}\mathbf{Q}^\top\mathbf{y}=\mathbf{y}$, and the quantities computed in steps 1 and 2 are products with $\mathbf{B}$:

$$
\mathbf{a}=\mathbf{P}\mathbf{A}\mathbf{w} = \mathbf{P}\mathbf{A}\mathbf{P}\mathbf{w}=\mathbf{B}\mathbf{w},
\quad
\rho=\|\mathbf{B}\mathbf{w}\|,
\quad
\mathbf{y}=\frac{\mathbf{a}}{\rho}=\frac{\mathbf{B}\mathbf{w}}{\rho},
\quad
\mathbf{b}=\mathbf{P}\mathbf{A}\mathbf{y}=\mathbf{P}\mathbf{A}\mathbf{P}\mathbf{y}=\mathbf{B}\mathbf{y}=\frac{\mathbf{B}^2\mathbf{w}}{\rho}.
$$

The table lists what each step computes in terms of $\mathbf{B}$ and the result it implements. The paragraphs after it justify each line.

| Step | In terms of $\mathbf{B}$ | Implements |
| --- | --- | --- |
| 1. First product | $\mathbf{a}=\mathbf{B}\mathbf{w}$ and $\rho=\lVert\mathbf{B}\mathbf{w}\rVert$ | the first half of a step of squared power iteration, by the projection view of [Remark 8](#remark-8-two-views-of-the-deflated-matrix) |
| 2. Second product | $\mathbf{b}=\mathbf{B}^2\mathbf{w}/\rho$ | the second half |
| 3. Candidate | $\mathbf{q}=\mathbf{u}_s/\lVert\mathbf{u}_s\rVert$, with $\mathbf{u}_\pm=\frac12(\mathbf{w}\pm\mathbf{B}\mathbf{w}/\rho)$ and $s=\pm1$ the sign of $\mathbf{w}^\top\mathbf{y}$ ($+1$ if it is zero), which picks the longer of $\mathbf{u}_+$ and $\mathbf{u}_-$ | the sign separation of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation), applied to the current iterate |
| 4. Check | $\lVert\mathbf{A}\mathbf{q}-\widehat\lambda\mathbf{q}\rVert\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\lVert\mathbf{A}\mathbf{q}\rVert$ | the certificate of [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair), with the Rayleigh quotient of [Remark 11](#remark-11-any-scalar-and-the-best-one) |
| 4. Update | $\mathbf{w}\leftarrow\mathbf{B}^2\mathbf{w}/\lVert\mathbf{B}^2\mathbf{w}\rVert$ | the iteration of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) |
| 5. Zero projected product | $\mathbf{B}\mathbf{w}=0$ | the case $\mathbf{B}\mathbf{w}_0=0$ of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) and of [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis) |

**The update is the iteration of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation).** Normalizing $\mathbf{b}$ gives $\mathbf{w}\leftarrow\mathbf{B}^2\mathbf{w}/\|\mathbf{B}^2\mathbf{w}\|$, the iteration of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation) on $\mathbf{B}$, at the cost of two products with $\mathbf{A}$ per step. It is well defined: $\rho>0$ implies $\mathbf{b}\neq0$, because

$$
\mathbf{w}^\top\mathbf{B}^2\mathbf{w}=\|\mathbf{B}\mathbf{w}\|^2=\rho^2>0 .
$$

By [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation), the iterates converge to a unit vector $\mathbf{w}$ with $\mathbf{B}^2\mathbf{w}=\rho^2\mathbf{w}$, where $\rho$ is the largest absolute value among the eigenvalues of $\mathbf{B}$ present in the starting vector. Squaring turns the eigenvalues $+\rho$ and $-\rho$ of $\mathbf{B}$ into the same eigenvalue $\rho^2$ of $\mathbf{B}^2$, so the tie of [Remark 5](#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude) cannot stall the iteration.

**The candidate is the sign separation of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation), before the limit.** Since $\mathbf{w}+s\mathbf{y}=\mathbf{w}+s\mathbf{B}\mathbf{w}/\rho=2\mathbf{u}_s$, the candidate $\mathbf{q}=\mathbf{u}_s/\|\mathbf{u}_s\|$ is the normalized $\mathbf{u}_+$ when $s=+1$ and the normalized $\mathbf{u}_-$ when $s=-1$, computed from the current iterate instead of the limit. The sign picks the longer of the two, since $\|\mathbf{u}_+\|^2-\|\mathbf{u}_-\|^2=\mathbf{w}^\top\mathbf{B}\mathbf{w}/\rho=\mathbf{w}^\top\mathbf{y}$, and the normalization is never small:

$$
n^2=\|\mathbf{w}+s\mathbf{y}\|^2=\|\mathbf{w}\|^2+\|\mathbf{y}\|^2+2s\,\mathbf{w}^\top\mathbf{y}=2+2|\mathbf{w}^\top\mathbf{y}|\ge2,
$$

which is the bound $\max\{\|\mathbf{u}_+\|,\|\mathbf{u}_-\|\}\ge1/\sqrt2$ of the proof of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation). Forming the candidate before the limit is justified by a direct calculation:

$$
\begin{aligned}
(\mathbf{B}-s\rho\,\mathbb{I}_d)\mathbf{q}
&=\frac{\textcolor{#2563eb}{\mathbf{B}\mathbf{w}}+s\mathbf{B}^2\mathbf{w}/\rho-s\rho\,\mathbf{w}\textcolor{#2563eb}{-\mathbf{B}\mathbf{w}}}{n}\\
&=\frac{s\bigl(\mathbf{B}^2\mathbf{w}-\rho^2\mathbf{w}\bigr)}{\rho\,n}.
\end{aligned}
$$

The <span style="color:#2563eb">blue terms</span> cancel because $s^2=1$. Since $n\ge\sqrt2$, the residual of $(s\rho,\mathbf{q})$ as an eigenpair of $\mathbf{B}$ is at most $\|\mathbf{B}^2\mathbf{w}-\rho^2\mathbf{w}\|/(\sqrt2\,\rho)$, which measures how far the current iterate is from an eigenvector of $\mathbf{B}^2$. As squared power iteration converges, the candidate therefore becomes an eigenvector of $\mathbf{B}$, for the eigenvalue $\rho$ or $-\rho$, and nothing is lost by separating the signs at every step.

**The check uses $\mathbf{A}$, not $\mathbf{B}$.** With exact stored eigenvectors, an eigenvector of $\mathbf{B}$ orthogonal to them is an eigenvector of $\mathbf{A}$ with the same eigenvalue, by the first step of the proof of [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis), and its Rayleigh quotient $\widehat\lambda=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ is then $s\rho$. Computed eigenvectors are only approximate, so $\mathbf{B}$ is not exactly the deflated matrix of [Lemma 1](#34-lemma-1-orthogonal-deflation), and an eigenvector of $\mathbf{B}$ need not be one of $\mathbf{A}$. The test therefore computes the residual with $\mathbf{A}$ itself, which [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) turns into a guarantee however $\mathbf{q}$ was obtained ([Remark 10](#remark-10-a-computable-bound-on-an-unknown-distance)), and it reports the Rayleigh quotient, the scalar with the smallest certificate ([Remark 11](#remark-11-any-scalar-and-the-best-one)). For the same reason, the exact recovery of [Corollary 2](#36-corollary-2-recovery-of-a-complete-eigenbasis) does not carry over to finite computations: convergence for $\mathbf{B}$ need not drive the residual for $\mathbf{A}$ below an arbitrary tolerance, as the [end of this section](#tolerance-stall) shows.

**The zero case.** If $\rho=0$, then $\mathbf{B}\mathbf{w}=0$, and $\mathbf{w}$ is an eigenvector of $\mathbf{B}$ for the eigenvalue $0$, the first case of [Proposition 2](#35-proposition-2-power-iteration-with-sign-separation). In exact arithmetic, this can only happen at the first step of a stage: if $\mathbf{B}\mathbf{w}\neq0$, then $\mathbf{w}$ has a component in an eigenspace of $\mathbf{B}$ with a nonzero eigenvalue, and so does every later iterate, since it is a multiple of $\mathbf{B}^{2k}\mathbf{w}$. A small computed $\rho$ does not establish that $\mathbf{w}$ is an eigenvector for $0$, still less that every remaining eigenvalue is zero, which is why the zero case is also accepted only through the residual test of step 4.

**Floating-point safeguards.** The implementation should reorthogonalize projected vectors, impose a maximum number of inner iterations, and report nonconvergence if the original residual does not reach the requested tolerance. No vector with computed norm zero should be normalized. A numerical breakdown should instead trigger a residual check, a restart, or a report of nonconvergence.

<span id="tolerance-stall"></span>**Why a stage can stall.** The residual for $\mathbf{A}$ splits exactly into a part that the inner iteration reduces and a part that it inherits. Let $\widehat\lambda_1,\dots,\widehat\lambda_r$ be the eigenvalue estimates of the stored pairs, and let the columns of $\mathbf{R}=\mathbf{A}\mathbf{Q}-\mathbf{Q}\operatorname{diag}(\widehat\lambda_1,\dots,\widehat\lambda_r)$ be their residuals. For a unit candidate $\mathbf{q}$ with $\mathbf{Q}^\top\mathbf{q}=0$ and $\widehat\lambda=\mathbf{q}^\top\mathbf{A}\mathbf{q}$, we have $\mathbf{P}\mathbf{q}=\mathbf{q}$, hence $\mathbf{A}\mathbf{q}=\mathbf{B}\mathbf{q}+\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\mathbf{q}$, where $\mathbf{B}\mathbf{q}-\widehat\lambda\mathbf{q}$ lies in $\operatorname{range}(\mathbf{P})$ and $\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\mathbf{q}$ in $\operatorname{range}(\mathbf{Q})$. Since $\mathbf{Q}^\top\mathbf{q}=0$, we also have $\mathbf{Q}^\top\mathbf{A}\mathbf{q}=(\mathbf{A}\mathbf{Q})^\top\mathbf{q}=\mathbf{R}^\top\mathbf{q}$. The two parts are orthogonal, so

$$
\textcolor{#2563eb}{\|\mathbf{A}\mathbf{q}-\widehat\lambda\mathbf{q}\|^2}=\textcolor{#ea580c}{\|\mathbf{B}\mathbf{q}-\widehat\lambda\mathbf{q}\|^2}+\textcolor{#7c3aed}{\|\mathbf{R}^\top\mathbf{q}\|^2}.
$$

The inner iteration drives the <span style="color:#ea580c">orange term</span> to zero. The <span style="color:#7c3aed">purple term</span> is inherited from the stored pairs: each entry of $\mathbf{R}^\top\mathbf{q}$ is the inner product of $\mathbf{q}$ with a stored residual, so the term is at most the sum of their squared norms, but nothing in the inner iteration makes it small. A stage therefore stalls when its threshold $\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$ is below the inherited part. This happens readily with $\varepsilon_{\mathrm{rel}}>0$, because the threshold shrinks with $|\widehat\lambda|$, whereas the inherited part comes from earlier stages, whose eigenvalues are larger in absolute value and whose accepted residuals can be nearly as large as their own thresholds. A threshold that is the same at every stage ($\varepsilon_{\mathrm{rel}}=0$) removes this systematic mismatch, but over many stages the inherited parts can still exceed it. Continuing each stage until the residual for $\mathbf{B}$ reaches rounding level, and only then checking the residual for $\mathbf{A}$, keeps every inherited part far below the threshold, at the price of more products. [Figure 5](#figure-hessian-deflation) below shows the stall on the Hessian of a neural network. The complete eigenbasis in [Figure 3](#figure-complete-eigenbasis) was computed in the last way. With the same constant threshold but without the continuation, that run stalls at stage 20 of 50, where the inherited part is $6.89\cdot10^{-10}$ and the threshold $6.48\cdot10^{-10}$.

<figure class="article-figure article-figure-wide" id="figure-hessian-deflation">
  <img src="/images/blog/matrix-vector-products/hessian-deflation.svg" width="1600" height="880" alt="Six panels on the Hessian of a small neural network. Top left: the training and test losses decreasing over 8,000 gradient-descent steps, with the point where the Hessian is taken. Top middle: the Hessian eigenvalues during training on a scale that is logarithmic away from zero; the eight largest are lines, five of them rising to between 3.0 and 6.5, and a grey band holds the others, its lower edge, the smallest eigenvalue, rising from about -0.8 to nearly zero. Top right: the eigenvalues at the last step counted by absolute value on a log scale from 1e-8 to 10, positive ones as bars above the axis and negative ones below, with ticks marking the eight largest. Bottom left: eight deflation stages, each with a residual decreasing geometrically until a constant threshold. Bottom middle: at a stalled second stage, the residual for the deflated matrix keeps decreasing to rounding level while the residual for the original matrix levels off at the inherited part, just above the threshold. Bottom right: the fraction of twenty runs certifying j pairs stays at one with the constant threshold and falls to 0.3 with the relative threshold." loading="lazy" />
  <figcaption><strong>Figure 5.</strong> <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> on the Hessian $\mathbf A=\nabla^2f(\theta)$ of the training loss $f$ of a small neural network, applied by automatic differentiation (<a href="#21-hessians">Section 2.1</a>). The network is a fully connected tanh network with layer widths 4, 16, 16 and 1, so that $\theta\in\mathbb R^{369}$, trained by 8,000 steps of gradient descent on $n=256$ Gaussian inputs labelled by a random teacher network. The top row describes the matrix, and the bottom row runs the algorithm on it. The 8 eigenvalues of largest absolute value at the last step, which the algorithm computes, are colored from <span style="background:linear-gradient(90deg,#b0dd2f,#481d6f);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span> in panels (b), (c) and (d). (a) The squared losses along gradient descent, on the training inputs (<span style="color:#2563eb">blue</span>) and on 1,024 fresh inputs labelled by the same teacher (<span style="color:#ea580c">orange</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>). The Hessian $\mathbf A$ of panels (c) to (f) is taken at the last step (<span style="color:#27322f">black</span> dot), where the training loss is $4.8\cdot10^{-4}$ and the test loss reaches its lowest value, $1.2\cdot10^{-3}$. (b) The eigenvalues of the Hessian along the same run, computed from the dense $369\times369$ matrix every 80 steps, on a vertical scale that is logarithmic beyond $\pm0.01$ and linear in between. The 8 largest are drawn as lines, and the other 361 lie in the <span style="color:#7d8783">grey</span> band, whose lower edge is the smallest eigenvalue. Training sharpens the top of the spectrum: the five largest eigenvalues rise from between 1.07 and 2.32 at initialization to between 3.04 and 6.50, and they separate from the next three, which settle between 0.25 and 0.34. The negative curvature nearly disappears: the smallest eigenvalue rises from $-0.83$ to $-0.0014$, and the number of negative eigenvalues falls from 191 to 139. (c) The 369 eigenvalues at the last step, counted by absolute value in bins of half a decade, the positive ones above the axis (<span style="color:#2563eb">blue</span>) and the negative ones below it (<span style="color:#ea580c">orange</span>). Their absolute values span more than eight orders of magnitude, from $1.1\cdot10^{-8}$ to 6.50, and 269 of them lie between $10^{-5}$ and $10^{-2}$. The five largest stand far above the rest, and every negative eigenvalue lies above $-0.0015$. The ticks at the top mark the 8 largest, which the bottom row computes. The bottom row uses two threshold rules (<a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Section 3.8</a>): a constant one, $\varepsilon_{\mathrm{abs}}=10^{-10}\|\mathbf A\|$ and $\varepsilon_{\mathrm{rel}}=0$, where $\|\mathbf A\|=6.50$ is the absolute value of the first eigenvalue found, and a relative one, $\varepsilon_{\mathrm{abs}}=\varepsilon_{\mathrm{rel}}=10^{-10}$. (d) One run with the constant threshold: the residual of the candidate against the number of Hessian–vector products, counted over all stages, with each stage labelled by the eigenvalue it finds. Each stage decreases geometrically, by a factor per product equal to the ratio of the next largest remaining absolute value to its own, until it reaches the threshold (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>), and the eight stages take 1,072 products. (e) A run with the relative threshold, at its second stage. By the <a href="#tolerance-stall">splitting of Section 3.8</a>, the squared residual for $\mathbf A$ (<span style="color:#2563eb">blue</span>), which the algorithm checks, is the sum of the squares of the residual for the deflated matrix $\mathbf B$ (<span style="color:#ea580c">orange</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>) and of the inherited part $\|\mathbf Q^\top\mathbf A\mathbf q\|$ (<span style="color:#7c3aed">purple</span>, <span style="text-decoration:underline dotted;text-decoration-thickness:0.12em;text-underline-offset:0.2em">dotted</span>). The inner iteration drives the first down to $5\cdot10^{-16}$, but the second stays at $7.31\cdot10^{-10}$, the residual accepted at the first stage. The threshold of the second stage, $10^{-10}(1+5.90)=6.90\cdot10^{-10}$, is just below it, so the stage never stops. Stage 2 is where stalls are most common, and the run shown is the first of the 20 to stall there. (f) Fraction of 20 runs, each with its own Gaussian starting vectors, that certify the $j$ largest pairs. With the relative threshold (<span style="color:#ea580c">orange</span>), 14 runs stall: 8 at stage 2, 3 at stage 4, and one each at stages 5, 6 and 7. With the constant threshold (<span style="color:#2563eb">blue</span>), all 20 runs certify the 8 pairs, and in the run of panel (d) every eigenvalue is within $5\cdot10^{-15}$ of the spectrum.</figcaption>
</figure>

The operator must stay fixed for the whole run. For a Hessian $\mathbf{A}=\nabla_\theta^2f(\theta)$, each product is computed as in [Section 2.1](#21-hessians), with $\theta$ and everything that defines $f$ held fixed.

Each nondegenerate inner iteration above uses two matrix–vector products, $\textcolor{#2563eb}{\mathbf{h}_1}$ and $\textcolor{#ea580c}{\mathbf{h}_2}$, since $\mathbf{A}\mathbf{q}$ comes for free by linearity. After $j-1$ eigenvectors have been stored, projecting a vector costs $O(d(j-1))$ arithmetic operations. Retaining $m$ eigenvectors requires $O(dm)$ storage. Thus computing and storing a complete eigenbasis still requires $O(d^2)$ storage, even though $\mathbf{A}$ itself is never formed.

## 4. The Lanczos method

The Lanczos method constructs an orthonormal basis of a Krylov subspace and projects the original eigenvalue problem onto that subspace. This produces a smaller symmetric tridiagonal matrix whose eigenpairs provide approximations to those of the original matrix. The construction requires only matrix–vector products ([Lanczos, 1950](#ref-lanczos1950)). Unless stated otherwise, all identities and termination results below concern exact arithmetic and a fixed matrix $\mathbf{A}$.

### 4.1 Definition 1 [Krylov subspace]

Given a unit vector $\mathbf{q}_1$, the Krylov subspace of order $m$ is

$$
\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{m-1}\mathbf{q}_1\}.
$$

Equivalently,

$$
\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\bigl\{p(\mathbf{A})\mathbf{q}_1:p\text{ is a polynomial of degree at most }m-1\bigr\}.
$$

These subspaces are nested:

$$
\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\subseteq\mathcal{K}_{m+1}(\mathbf{A},\mathbf{q}_1).
$$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

Power iteration started at $\mathbf{q}_1$ keeps only the last vector $\mathbf{A}^{m-1}\mathbf{q}_1$ of this list. The Krylov subspace keeps all of them, so it contains every polynomial filter of degree at most $m-1$ applied to $\mathbf{q}_1$. By the spectral decomposition, $p(\mathbf{A})\mathbf{q}_1=\sum_\lambda p(\lambda)\Pi_\lambda\mathbf{q}_1$, so **a polynomial that is large at one eigenvalue and small at the others can isolate that eigencomponent much better than the monomial $t^{m-1}$ used by power iteration.**

### 4.2 The Gram–Schmidt process

The Gram–Schmidt process turns linearly independent vectors $\mathbf{v}_1,\dots,\mathbf{v}_m$ of $\mathbb{R}^d$ into orthonormal vectors $\mathbf{q}_1,\dots,\mathbf{q}_m$, that is, vectors with
$\mathbf{q}_i^\top\mathbf{q}_j=\delta_{ij}$
that span the same subspaces: $\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_j\}=\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_j\}$ for every $j$. It builds them one at a time. The first is $\mathbf{q}_1=\mathbf{v}_1/\|\mathbf{v}_1\|$. At step $j\ge2$, it subtracts from $\mathbf{v}_j$ its components along the vectors already built, and normalizes the remainder:

$$
\mathbf{r}_j=\mathbf{v}_j-\sum_{i=1}^{j-1}(\mathbf{q}_i^\top\mathbf{v}_j)\,\mathbf{q}_i,
\qquad
\mathbf{q}_j=\frac{\mathbf{r}_j}{\|\mathbf{r}_j\|}.
$$

Both properties are proved together, by induction on $j$. Suppose that $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$ are orthonormal and span the same subspace as $\mathbf{v}_1,\dots,\mathbf{v}_{j-1}$. This holds for $j=2$, since $\mathbf{q}_1$ is a unit multiple of $\mathbf{v}_1$. Three facts carry it over to $j$.

* First, the remainder $\mathbf{r}_j$ is not zero, so the division by $\|\mathbf{r}_j\|$ is allowed. If $\mathbf{r}_j$ were zero, its definition would give $\mathbf{v}_j=\sum_{i<j}(\mathbf{q}_i^\top\mathbf{v}_j)\,\mathbf{q}_i$, a vector of $\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{j-1}\}$. By the hypothesis, this subspace is $\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_{j-1}\}$, so $\mathbf{v}_j$ would be a combination of $\mathbf{v}_1,\dots,\mathbf{v}_{j-1}$, which linear independence forbids. Without independence, $\mathbf{r}_j=0$ is exactly the signal that $\mathbf{v}_j$ adds no new direction.

* Second, $\mathbf{q}_j$ is orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$. For $k<j$, every term of the sum but $i=k$ vanishes in the inner product with $\mathbf{q}_k$, since $\mathbf{q}_k^\top\mathbf{q}_i=\delta_{ki}$, i.e.
$\mathbf{q}_k^\top\mathbf{r}_j=\mathbf{q}_k^\top\mathbf{v}_j-\sum_{i=1}^{j-1}(\mathbf{q}_i^\top\mathbf{v}_j)\,\mathbf{q}_k^\top\mathbf{q}_i
=\mathbf{q}_k^\top\mathbf{v}_j-\mathbf{q}_k^\top\mathbf{v}_j
=0$.
Dividing by $\|\mathbf{r}_j\|$ keeps this orthogonality and gives $\mathbf{q}_j=\mathbf{r}_j/\|\mathbf{r}_j\|$ norm one, so $\mathbf{q}_1,\dots,\mathbf{q}_j$ are orthonormal.

* Third, the spans agree. The vector $\mathbf{q}_j$ is a combination of $\mathbf{v}_j$ and $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$, hence of $\mathbf{v}_1,\dots,\mathbf{v}_j$ by the hypothesis. Conversely, $\mathbf{v}_j=\|\mathbf{r}_j\|\,\mathbf{q}_j+\sum_{i<j}(\mathbf{q}_i^\top\mathbf{v}_j)\,\mathbf{q}_i$ is a combination of $\mathbf{q}_1,\dots,\mathbf{q}_j$.

For example, $\mathbf{v}_1=[1,1,0]^\top$ and $\mathbf{v}_2=[1,0,1]^\top$ give $\mathbf{q}_1=[1,1,0]^\top/\sqrt2$, then $\mathbf{q}_1^\top\mathbf{v}_2=1/\sqrt2$, $\mathbf{r}_2=\mathbf{v}_2-\mathbf{q}_1/\sqrt2=[\tfrac12,-\tfrac12,1]^\top$ and $\mathbf{q}_2=[1,-1,2]^\top/\sqrt6$, and indeed $\mathbf{q}_1^\top\mathbf{q}_2=(1-1+0)/\sqrt{12}=0$. Step $j$ costs $j-1$ inner products and subtractions, so the cost per step grows with $j$. The following section applies the process to the spanning vectors $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$ of the Krylov subspace.

### 4.3 How the Lanczos method works

> <span class="article-kicker article-kicker--theorem">The idea</span> The Lanczos method runs the [Gram–Schmidt process](#42-the-gramschmidt-process) on the Krylov subspace $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1) = \operatorname{span}\{\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{m-1}\mathbf{q}_1\}$, one matrix–vector product per step. 
> The numbers that Gram–Schmidt computes along the way form a small symmetric matrix, and the eigenvalues of that matrix approximate eigenvalues of $\mathbf{A}$.

This section unpacks these sentences. It first explains why the search for eigenvalues needs an orthonormal basis of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, then runs Gram–Schmidt on the Krylov subspace by hand for four steps, states the general step, and finally shows how eigenvalues and eigenvectors come out of the coefficients. The proofs are in Lemmas [3](#46-lemma-3-orthogonality-and-tridiagonal-projection) to [5](#48-lemma-5-variational-characterization-and-monotonicity) and [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run).

After $m-1$ products, every vector of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ is within reach. It is a combination $c_0\mathbf{q}_1+c_1\mathbf{A}\mathbf{q}_1+\dots+c_{m-1}\mathbf{A}^{m-1}\mathbf{q}_1=p(\mathbf{A})\mathbf{q}_1$, and by the spectral decomposition, $p(\mathbf{A})\mathbf{q}_1=\sum_{\lambda}p(\lambda)\,\Pi_\lambda\mathbf{q}_1$ reweights each eigencomponent of $\mathbf{q}_1$ by the value $p(\lambda)$. Power iteration only ever uses the polynomial $t \mapsto t^{m-1}$. The Lanczos method considers all of them at once.

To turn these vectors into eigenvalue estimates, recall that the Rayleigh quotient of a unit vector never exceeds $\lambda_{\max}(\mathbf{A})=\max_{\|\mathbf{x}\|=1}\mathbf{x}^\top\mathbf{A}\mathbf{x}$ and never falls below $\lambda_{\min}(\mathbf{A})=\min_{\|\mathbf{x}\|=1}\mathbf{x}^\top\mathbf{A}\mathbf{x}$ ([Remark 7](#remark-7-the-shift-only-needs-to-pass-the-midpoint)), with equality at the corresponding eigenvectors. 
These optimizations run over all unit vectors of $\mathbb{R}^d$, which are out of reach. Running them over the unit vectors of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ only gives two numbers $\vartheta_{\max}^{(m)}\le\lambda_{\max}(\mathbf{A})$ and $\vartheta_{\min}^{(m)}\ge\lambda_{\min}(\mathbf{A})$. They are the best estimates of the extreme eigenvalues that Rayleigh quotients of vectors of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ can give, and they improve as $m$ grows, because the subspaces are nested. The power iterate $\mathbf{v}=\mathbf{A}^{m-1}\mathbf{q}_1/\|\mathbf{A}^{m-1}\mathbf{q}_1\|$ is a unit vector of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, so the estimate of power iteration, $\mathbf{v}^\top\mathbf{A}\mathbf{v}$, lies between $\vartheta_{\min}^{(m)}$ and $\vartheta_{\max}^{(m)}$: whichever extreme eigenvalue power iteration approaches, the corresponding estimate of the Lanczos method is at least as close. [Lemma 5](#48-lemma-5-variational-characterization-and-monotonicity) states these facts precisely.

Searching a subspace needs coordinates, and the convenient ones come from an orthonormal basis $\mathbf{q}_1,\dots,\mathbf{q}_m$ of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, the columns of $\mathbf{Q}_m=[\mathbf{q}_1,\dots,\mathbf{q}_m]$. Every vector of the subspace is then $\mathbf{x}=\mathbf{Q}_m\mathbf{y}=y_1\mathbf{q}_1+\dots+y_m\mathbf{q}_m$ for a unique $\mathbf{y}\in\mathbb{R}^m$, lengths are preserved, $\|\mathbf{x}\|=\|\mathbf{y}\|$, and

$$
\mathbf{x}^\top\mathbf{A}\mathbf{x}=\mathbf{y}^\top\mathbf{T}_m\mathbf{y},
\qquad
\mathbf{T}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m\in\mathbb{R}^{m\times m}.
$$

The search over $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ thus becomes the eigenvalue problem of the $m\times m$ matrix $\mathbf{T}_m$, whose entry $(\mathbf{T}_m)_{ij}=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$ is the coordinate of $\mathbf{A}\mathbf{q}_j$ along $\mathbf{q}_i$. Gram–Schmidt applied to the spanning vectors $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$ of the Krylov subspace produces the orthonormal basis, and, as the following steps show, it computes these coordinates on the way.

**Step 0: Gram–Schmidt on $\mathcal{K}_1(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1\}=\{c_0\mathbf{q}_1:c_0\in\mathbb{R}\}$.** There is nothing to do. The starting vector $\mathbf{q}_1$ is a unit vector, so on its own it is already an orthonormal basis of $\mathcal{K}_1(\mathbf{A},\mathbf{q}_1)$: it is the first basis vector, obtained without any product.

**Step 1: Gram–Schmidt on $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1, \mathbf{A}\mathbf{q}_1\}=\{(c_0\mathbb{I}_d+c_1\mathbf{A})\mathbf{q}_1:c_0,c_1\in\mathbb{R}\}$.** The first product gives $\mathbf{A}\mathbf{q}_1$, the vector that $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$ adds. Gram–Schmidt removes its component along $\mathbf{q}_1$, whose coefficient $\textcolor{#7c3aed}{\alpha_1}=\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_1$ is the Rayleigh quotient of $\mathbf{q}_1$. It is the first eigenvalue estimate, the same as that of power iteration. The remainder $\mathbf{r}_1=\mathbf{A}\mathbf{q}_1-\alpha_1\mathbf{q}_1$ is orthogonal to $\mathbf{q}_1$, and dividing it by its length $\textcolor{#16a34a}{\beta_1}=\|\mathbf{r}_1\|$ gives the second basis vector $\mathbf{q}_2=\mathbf{r}_1/\textcolor{#16a34a}{\beta_1}$. Rearranged, the step reads

$$
\mathbf{A}\mathbf{q}_1=\alpha_1\mathbf{q}_1+\mathbf{r}_1=\textcolor{#7c3aed}{\alpha_1\mathbf{q}_1}+\textcolor{#16a34a}{\beta_1\mathbf{q}_2},
\qquad
\mathbf{q}_2
=\frac{\mathbf{r}_1}{\beta_1}
=\frac{\mathbf{A}-\alpha_1\mathbb{I}_d}{\beta_1} \mathbf{q}_1.
$$

So $\mathbf{q}_1,\mathbf{q}_2$ is an orthonormal basis of $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$, and $\mathbf{q}_2=p_1(\mathbf{A})\mathbf{q}_1$ for the polynomial $p_1(t)=(t-\alpha_1)/\beta_1$ of degree one. In the colors used throughout this section, a <span style="color:#7c3aed">purple</span> coefficient is computed at the current step, and a <span style="color:#16a34a">green</span> one is the length of the new direction.

**Step 2: Gram–Schmidt on $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1, \mathbf{A}\mathbf{q}_1, \mathbf{A}^2\mathbf{q}_1\}$.** The vector that $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$ adds is $\mathbf{A}^2\mathbf{q}_1$, and textbook Gram–Schmidt would orthogonalize it next. The Lanczos method applies $\mathbf{A}$ to the newest basis vector $\mathbf{q}_2$ instead, which also costs one product. Indeed, by Step 1,

$$
\mathbf{A}\mathbf{q}_2=\frac{1}{\beta_1}\,\textcolor{#2563eb}{\mathbf{A}^2\mathbf{q}_1}-\frac{\alpha_1}{\beta_1}\,\textcolor{#ea580c}{\mathbf{A}\mathbf{q}_1}.
$$

The <span style="color:#ea580c">orange vector</span> already lies in $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\mathbf{q}_2\}$, so removing the components along $\mathbf{q}_1$ and $\mathbf{q}_2$ deletes it, and only the <span style="color:#2563eb">blue power</span> contributes. Hence $\mathbf{A}\mathbf{q}_2$ brings the same new direction as $\mathbf{A}^2\mathbf{q}_1$. Gram–Schmidt needs two coefficients. The one on $\mathbf{q}_1$ comes for free from symmetry and Step 1:

$$
\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_2=(\mathbf{A}\mathbf{q}_1)^\top\mathbf{q}_2=(\alpha_1\mathbf{q}_1+\beta_1\mathbf{q}_2)^\top\mathbf{q}_2=\textcolor{#ea580c}{\beta_1}.
$$

It is the length found in Step 1, now reused, and <span style="color:#ea580c">orange</span> marks such known coefficients. The one on $\mathbf{q}_2$ is the only new number, $\textcolor{#7c3aed}{\alpha_2}=\mathbf{q}_2^\top\mathbf{A}\mathbf{q}_2$. Subtracting both components and normalizing gives

$$
\mathbf{r}_2=\mathbf{A}\mathbf{q}_2-\beta_1\mathbf{q}_1-\alpha_2\mathbf{q}_2,
\qquad
\textcolor{#16a34a}{\beta_2}=\|\mathbf{r}_2\|,
\qquad
\mathbf{q}_3=\frac{\mathbf{r}_2}{\beta_2}=\frac{(\mathbf{A}-\alpha_1\mathbb{I}_d)p_1(\mathbf{A})-\beta_1\mathbb{I}_d}{\beta_2} \mathbf{q}_1,
$$

that is, $\mathbf{A}\mathbf{q}_2=\textcolor{#ea580c}{\beta_1\mathbf{q}_1}+\textcolor{#7c3aed}{\alpha_2\mathbf{q}_2}+\textcolor{#16a34a}{\beta_2\mathbf{q}_3}$. Now $\mathbf{q}_1,\mathbf{q}_2,\mathbf{q}_3$ is an orthonormal basis of $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$, and $\mathbf{q}_3=p_2(\mathbf{A})\mathbf{q}_1$ for the polynomial $p_2(t)=\bigl((t-\alpha_2)\,p_1(t)-\beta_1\bigr)/\beta_2$ of degree two. The coefficients found so far already give the matrix of the Rayleigh quotient on $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$,

$$
\mathbf{T}_2=\mathbf{Q}_2^\top\mathbf{A}\mathbf{Q}_2=\begin{bmatrix}\alpha_1&\beta_1\\\beta_1&\alpha_2\end{bmatrix},
$$

whose two eigenvalues are the smallest and largest values of $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ over the unit vectors of $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$.

**Step 3: Gram–Schmidt on $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1, \mathbf{A}\mathbf{q}_1, \mathbf{A}^2\mathbf{q}_1, \mathbf{A}^3\mathbf{q}_1\}$, where a coefficient vanishes.** The vector that $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$ adds is $\mathbf{A}^3\mathbf{q}_1$, and, as in Step 2, the third product is applied to the newest basis vector instead, giving $\mathbf{A}\mathbf{q}_3$. The argument of Step 2 carries over. Since $\mathbf{q}_3=p_2(\mathbf{A})\mathbf{q}_1$, where $p_2$ has degree two and leading coefficient $1/(\beta_1\beta_2)$,

$$
\mathbf{A}\mathbf{q}_3=\frac{1}{\beta_1\beta_2}\,\textcolor{#2563eb}{\mathbf{A}^3\mathbf{q}_1}+\textcolor{#ea580c}{\mathbf{k}},
\qquad
\textcolor{#ea580c}{\mathbf{k}}\in\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3\},
$$

where the <span style="color:#ea580c">orange vector</span> collects the terms of lower degree. Removing the components along $\mathbf{q}_1,\mathbf{q}_2,\mathbf{q}_3$ deletes it, so $\mathbf{A}\mathbf{q}_3$ brings the same new direction as $\mathbf{A}^3\mathbf{q}_1$. Its coefficient on $\mathbf{q}_1$ is

$$
\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_3=(\mathbf{A}\mathbf{q}_1)^\top\mathbf{q}_3=(\alpha_1\mathbf{q}_1+\beta_1\mathbf{q}_2)^\top\mathbf{q}_3=\textcolor{#7d8783}{0},
$$

because $\mathbf{A}\mathbf{q}_1$ lies in $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$, to which $\mathbf{q}_3$ is orthogonal. Gram–Schmidt has nothing to subtract along $\mathbf{q}_1$, and <span style="color:#7d8783">grey</span> marks such vanishing coefficients. The coefficient on $\mathbf{q}_2$ is $(\mathbf{A}\mathbf{q}_2)^\top\mathbf{q}_3=\textcolor{#ea580c}{\beta_2}$, known from Step 2, and the one on $\mathbf{q}_3$ is the new number $\textcolor{#7c3aed}{\alpha_3}=\mathbf{q}_3^\top\mathbf{A}\mathbf{q}_3$. Two subtractions and a normalization give $\mathbf{q}_4$:
$\mathbf{A}\mathbf{q}_3=\textcolor{#ea580c}{\beta_2\mathbf{q}_2}+\textcolor{#7c3aed}{\alpha_3\mathbf{q}_3}+\textcolor{#16a34a}{\beta_3\mathbf{q}_4}$,
and $\mathbf{q}_1,\dots,\mathbf{q}_4$ is an orthonormal basis of $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$. Solving this relation for $\mathbf{q}_4$, with $\mathbf{q}_2=p_1(\mathbf{A})\mathbf{q}_1$ and $\mathbf{q}_3=p_2(\mathbf{A})\mathbf{q}_1$, gives $\mathbf{q}_4=p_3(\mathbf{A})\mathbf{q}_1$ for the polynomial $p_3(t)=\bigl((t-\alpha_3)\,p_2(t)-\beta_2\,p_1(t)\bigr)/\beta_3$ of degree three. The coefficients found so far give the matrix of the Rayleigh quotient on $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$,

$$
\mathbf{T}_3=\mathbf{Q}_3^\top\mathbf{A}\mathbf{Q}_3=\begin{bmatrix}\alpha_1&\beta_1&\textcolor{#7d8783}{0}\\\beta_1&\alpha_2&\beta_2\\\textcolor{#7d8783}{0}&\beta_2&\alpha_3\end{bmatrix},
$$

where the <span style="color:#7d8783">grey zeros</span> are the vanishing coefficient $\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_3$ and its symmetric copy. Its smallest and largest eigenvalues are the smallest and largest values of $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ over the unit vectors of $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$.

**Step 4: Gram–Schmidt on $\mathcal{K}_5(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1, \mathbf{A}\mathbf{q}_1, \mathbf{A}^2\mathbf{q}_1, \mathbf{A}^3\mathbf{q}_1, \mathbf{A}^4\mathbf{q}_1\}$, and the pattern.** The fourth product gives $\mathbf{A}\mathbf{q}_4$, which, by the argument of Steps 2 and 3, brings the same new direction as $\mathbf{A}^4\mathbf{q}_1$. Its coefficients on $\mathbf{q}_1$ and $\mathbf{q}_2$ vanish, since $\mathbf{A}\mathbf{q}_1$ and $\mathbf{A}\mathbf{q}_2$ lie in $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$, to which $\mathbf{q}_4$ is orthogonal. Its coefficient on $\mathbf{q}_3$ is $\textcolor{#ea580c}{\beta_3}$, known from Step 3, and the new number is $\textcolor{#7c3aed}{\alpha_4}=\mathbf{q}_4^\top\mathbf{A}\mathbf{q}_4$, so that $\mathbf{A}\mathbf{q}_4=\textcolor{#ea580c}{\beta_3\mathbf{q}_3}+\textcolor{#7c3aed}{\alpha_4\mathbf{q}_4}+\textcolor{#16a34a}{\beta_4\mathbf{q}_5}$, and $\mathbf{q}_5=p_4(\mathbf{A})\mathbf{q}_1$ for the polynomial $p_4(t)=\bigl((t-\alpha_4)\,p_3(t)-\beta_3\,p_2(t)\bigr)/\beta_4$ of degree four. The four steps have computed the coordinates of $\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}\mathbf{q}_4$ in the basis:

| | on $\mathbf{q}_1$ | on $\mathbf{q}_2$ | on $\mathbf{q}_3$ | on $\mathbf{q}_4$ | on $\mathbf{q}_5$ |
|---|:---:|:---:|:---:|:---:|:---:|
| $\mathbf{A}\mathbf{q}_1$ (step 1) | $\textcolor{#7c3aed}{\alpha_1}$ | $\textcolor{#16a34a}{\beta_1}$ | | | |
| $\mathbf{A}\mathbf{q}_2$ (step 2) | $\textcolor{#ea580c}{\beta_1}$ | $\textcolor{#7c3aed}{\alpha_2}$ | $\textcolor{#16a34a}{\beta_2}$ | | |
| $\mathbf{A}\mathbf{q}_3$ (step 3) | $\textcolor{#7d8783}{0}$ | $\textcolor{#ea580c}{\beta_2}$ | $\textcolor{#7c3aed}{\alpha_3}$ | $\textcolor{#16a34a}{\beta_3}$ | |
| $\mathbf{A}\mathbf{q}_4$ (step 4) | $\textcolor{#7d8783}{0}$ | $\textcolor{#7d8783}{0}$ | $\textcolor{#ea580c}{\beta_3}$ | $\textcolor{#7c3aed}{\alpha_4}$ | $\textcolor{#16a34a}{\beta_4}$ |

Each <span style="color:#16a34a">green</span> length reappears in <span style="color:#ea580c">orange</span> in the next row, by symmetry. The first four columns are the matrix of the Rayleigh quotient on $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$, which Gram–Schmidt has therefore filled for free:

$$
\mathbf{T}_4=\mathbf{Q}_4^\top\mathbf{A}\mathbf{Q}_4=\begin{bmatrix}
\textcolor{#7c3aed}{\alpha_1}&\textcolor{#16a34a}{\beta_1}&\textcolor{#7d8783}{0}&\textcolor{#7d8783}{0}\\
\textcolor{#ea580c}{\beta_1}&\textcolor{#7c3aed}{\alpha_2}&\textcolor{#16a34a}{\beta_2}&\textcolor{#7d8783}{0}\\
\textcolor{#7d8783}{0}&\textcolor{#ea580c}{\beta_2}&\textcolor{#7c3aed}{\alpha_3}&\textcolor{#16a34a}{\beta_3}\\
\textcolor{#7d8783}{0}&\textcolor{#7d8783}{0}&\textcolor{#ea580c}{\beta_3}&\textcolor{#7c3aed}{\alpha_4}
\end{bmatrix}.
$$

It is tridiagonal: only the diagonal and its two neighbors are nonzero. The last column of the table, $\beta_4$ on $\mathbf{q}_5$, lies outside $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$, and it plays a central role below.

**The general step.** The same reasoning holds at every step $j\in\{1,2,3,\dots\}$, as long as the previous steps have produced $\mathbf{q}_j$, which for $j\ge2$ means that the length $\beta_{j-1}=\|\mathbf{r}_{j-1}\|$ found at step $j-1$ is positive. For $j=1$, there is no earlier basis vector, and the conventions $\mathbf{q}_0=0$ and $\beta_0=0$ of [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown) let the formulas below cover this case too. By symmetry, the coefficient of $\mathbf{A}\mathbf{q}_j$ on an earlier basis vector $\mathbf{q}_i$ is $(\mathbf{A}\mathbf{q}_i)^\top\mathbf{q}_j$, and $\mathbf{A}\mathbf{q}_i$ lies in $\mathcal{K}_{i+1}(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{i+1}\}$, to which $\mathbf{q}_j$ is orthogonal as soon as $i+2\le j$. Only two coefficients then survive, $i=j-1$ and $i=j$:
* The one on $\mathbf{q}_{j-1}$ is the length $\beta_{j-1}=\|\mathbf{r}_{j-1}\|$ already computed at the previous step, as for $\beta_1$ in Step 2:
$$
\mathbf{q}_{j-1}^\top\mathbf{A}\mathbf{q}_j
=(\mathbf{A}\mathbf{q}_{j-1})^\top\mathbf{q}_j
=(\beta_{j-2}\mathbf{q}_{j-2}+\alpha_{j-1}\mathbf{q}_{j-1}+\beta_{j-1}\mathbf{q}_j)^\top\mathbf{q}_j
=\textcolor{#ea580c}{\beta_{j-1}}.
% by symmetry, the relation of step $j-1$, and orthonormality. 
$$ 

* The one on $\mathbf{q}_j$ is the new number $\textcolor{#7c3aed}{\alpha_j}=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$. 

Subtracting both components from $\mathbf{A}\mathbf{q}_j$ leaves the remainder $\mathbf{r}_j$, orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_j$, and when its length $\textcolor{#16a34a}{\beta_j}=\|\mathbf{r}_j\|$ is positive, dividing by it gives the next basis vector $\mathbf{q}_{j+1}$:

$$
\mathbf{r}_j
=\mathbf{A}\mathbf{q}_j\textcolor{#ea580c}{-\beta_{j-1}\mathbf{q}_{j-1}}\textcolor{#7c3aed}{-\alpha_j\mathbf{q}_j}
=\textcolor{#16a34a}{\beta_j\mathbf{q}_{j+1}}.
$$

Step $j$ of [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown) below is therefore the following Gram–Schmidt step, which makes the two subtractions one after the other.
$$
\mathbf{q}_j
\xrightarrow{\ \mathbf{A}\ }
\mathbf{A}\mathbf{q}_j
\xrightarrow{\ \textcolor{#ea580c}{-\beta_{j-1}\mathbf{q}_{j-1}}\ }
\mathbf{z}_j
\xrightarrow{\ \textcolor{#7c3aed}{-\alpha_j\mathbf{q}_j}\ }
\mathbf{r}_j
\xrightarrow{\ \textcolor{#16a34a}{\div\beta_j}\ }
\textcolor{#16a34a}{\mathbf{q}_{j+1}}.
$$

| Quantity | What it is | Why it is needed |
|---|---|---|
| $\mathbf{A}\mathbf{q}_j$ | the only product of step $j$ | temporary candidate that carries the new Krylov direction |
| $\mathbf{z}_j=\mathbf{A}\mathbf{q}_j\textcolor{#ea580c}{-\beta_{j-1}\mathbf{q}_{j-1}}$ | the candidate without its component on $\mathbf{q}_{j-1}$ | that coefficient is $\beta_{j-1}$, already known from step $j-1$ by symmetry |
| $\textcolor{#7c3aed}{\alpha_j}=\mathbf{q}_j^\top\mathbf{z}_j$ | the coefficient on $\mathbf{q}_j$, equal to $\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$ | removes the last old component and fills the diagonal of $\mathbf{T}_m$ |
| $\mathbf{r}_j=\mathbf{z}_j\textcolor{#7c3aed}{-\alpha_j\mathbf{q}_j}$ | the part of $\mathbf{A}\mathbf{q}_j$ orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_j$ | the new direction, not yet normalized |
| $\textcolor{#16a34a}{\beta_j}=\lVert\mathbf{r}_j\rVert$ | the length of the new direction | normalizes it and fills the off-diagonal of $\mathbf{T}_m$. The value $\beta_j=0$ means that no new direction exists (exact breakdown). |
| $\textcolor{#16a34a}{\mathbf{q}_{j+1}}=\mathbf{r}_j/\textcolor{#16a34a}{\beta_j}$ | the new basis vector | extends the orthonormal basis by one vector |

Rearranging the definitions of $\mathbf{z}_j$ and $\mathbf{r}_j$ gives the three-term recurrence
$\mathbf{A}\mathbf{q}_j=\textcolor{#ea580c}{\beta_{j-1}\mathbf{q}_{j-1}}+\textcolor{#7c3aed}{\alpha_j\mathbf{q}_j}+\textcolor{#16a34a}{\beta_j\mathbf{q}_{j+1}}$.
The <span style="color:#ea580c">orange term</span> was found at the previous step, the <span style="color:#7c3aed">purple term</span> is the only coefficient computed at step $j$, and the <span style="color:#16a34a">green term</span> is the new direction. In terms of polynomials, $\mathbf{q}_{j+1}=p_j(\mathbf{A})\mathbf{q}_1$ with $p_0=1$ and 
$\beta_j\,p_j(t)=(t-\alpha_j)\,p_{j-1}(t)-\beta_{j-1}\,p_{j-2}(t)$ for all $j \ge 1$, where $p_{-1}=0$.
<span id="krylov-inner-product"></span>Gram–Schmidt on $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ is thus Gram–Schmidt on the polynomials $1,t,t^2,\dots$ for the inner product $\langle p,r\rangle=\bigl(p(\mathbf{A})\mathbf{q}_1\bigr)^\top r(\mathbf{A})\mathbf{q}_1=\sum_\lambda p(\lambda)\,r(\lambda)\,\|\Pi_\lambda\mathbf{q}_1\|^2$, and the polynomials $p_j$ are <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="lanczos-orthonormal-polynomials" aria-label="Prove that the Lanczos polynomials are orthonormal for this inner product">orthonormal for it</button><span id="lanczos-orthonormal-polynomials" class="explanation-popover" popover="auto" role="note" aria-label="Proof that the Lanczos polynomials are orthonormal" data-label="Derivation">Everything follows from $\mathbf{q}_{j+1}=p_j(\mathbf{A})\mathbf{q}_1$. First, the polynomials are orthonormal because the basis vectors are: for all $i,j\ge0$, <span class="explanation-popover__equation">$\langle p_i,p_j\rangle=\bigl(p_i(\mathbf{A})\mathbf{q}_1\bigr)^\top p_j(\mathbf{A})\mathbf{q}_1=\mathbf{q}_{i+1}^\top\mathbf{q}_{j+1}=\delta_{ij}.$</span> Second, the sum over eigenvalues comes from the expansion $\mathbf{A}^k\mathbf{q}_1=\sum_\lambda\lambda^k\,\Pi_\lambda\mathbf{q}_1$ of the introduction. By linearity, $p(\mathbf{A})\mathbf{q}_1=\sum_\lambda p(\lambda)\,\Pi_\lambda\mathbf{q}_1$ for every polynomial $p$, and since the eigenspaces of distinct eigenvalues are orthogonal, only the terms with $\mu=\lambda$ survive in the product: <span class="explanation-popover__equation">$\bigl(p(\mathbf{A})\mathbf{q}_1\bigr)^\top r(\mathbf{A})\mathbf{q}_1=\sum_{\lambda,\mu}p(\lambda)\,r(\mu)\,(\Pi_\lambda\mathbf{q}_1)^\top\Pi_\mu\mathbf{q}_1=\sum_\lambda p(\lambda)\,r(\lambda)\,\|\Pi_\lambda\mathbf{q}_1\|^2.$</span> Third, this is an inner product on the polynomials that matter. It is symmetric and bilinear, and $\langle p,p\rangle=\sum_\lambda p(\lambda)^2\,\|\Pi_\lambda\mathbf{q}_1\|^2$ vanishes only if $p$ is zero at every eigenvalue $\lambda$ with $\Pi_\lambda\mathbf{q}_1\neq0$. If there are $s$ such eigenvalues, a nonzero polynomial of degree less than $s$ cannot vanish at all of them, and these degrees cover $p_0,\dots,p_{s-1}$, all the polynomials the method produces, since it stops at step $s$ by [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run). Finally, $p_j$ has degree $j$ and the positive leading coefficient $1/(\beta_1\cdots\beta_j)$, so $p_0,\dots,p_j$ are exactly what Gram–Schmidt produces from $1,t,\dots,t^j$ for this inner product.</span></span>.

Compared with textbook Gram–Schmidt on $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$, the method makes two changes, and neither changes the subspaces. 
* First, at step $j$ it multiplies the newest basis vector $\mathbf{q}_j$ by $\mathbf{A}$ instead of forming the next power $\mathbf{A}^j\mathbf{q}_1$. In general, $\mathbf{q}_j$ has a nonzero coefficient on $\mathbf{A}^{j-1}\mathbf{q}_1$ (recall that $\mathbf{q}_j$ exists only when $\beta_{j-1}=\|\mathbf{r}_{j-1}\|>0$), since otherwise it would lie in $\mathcal{K}_{j-1}(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{j-2}\mathbf{q}_1\}=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{j-1}\}$, to which it is orthogonal. By the recurrence for the polynomials $p_j$ above, this coefficient is $1/(\beta_1\beta_2\cdots\beta_{j-1})$, the leading coefficient of $p_{j-1}$. Hence $\mathbf{A}\mathbf{q}_j$ is a nonzero multiple of $\mathbf{A}^j\mathbf{q}_1$ plus a vector of $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, and the two bring the same new direction, the direction $\mathbf{q}_{j+1}$ orthogonal to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, as in Step 2, where $\mathbf{A}\mathbf{q}_2$ and $\mathbf{A}^2\mathbf{q}_1$ both bring $\mathbf{q}_3$. In exact arithmetic, both choices therefore give the same basis. In floating-point arithmetic they do not: the basis computed from the powers $\mathbf{A}^j\mathbf{q}_1$ loses its orthogonality within a few steps, as the example below shows, and the reason is the following. Since $p_j$ has leading coefficient $1/(\beta_1\cdots\beta_j)$, and by the three-term recurrence,

  $$
  \mathbf{A}^j\mathbf{q}_1=\textcolor{#16a34a}{\beta_1\beta_2\cdots\beta_j\,\mathbf{q}_{j+1}}+\underbrace{\mathbf{k}_j}_{\in\,\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)},
  \qquad
  \mathbf{A}\mathbf{q}_j=\textcolor{#16a34a}{\beta_j\,\mathbf{q}_{j+1}}+\underbrace{\beta_{j-1}\mathbf{q}_{j-1}+\alpha_j\mathbf{q}_j}_{\in\,\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)},
  $$

  for some vector $\mathbf{k}_j$. In each of the two vectors, the green term is orthogonal to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_j\}$ and the other terms lie in it, so Gram–Schmidt removes the other terms and keeps only the <span style="color:#16a34a">green term</span>. Let $\mathbf{x}$ be either vector, $\mathbf{A}^j\mathbf{q}_1$ or $\mathbf{A}\mathbf{q}_j$, and $\textcolor{#16a34a}{\mathbf{g}}$ its green term. What decides the accuracy is the ratio $f=\|\textcolor{#16a34a}{\mathbf{g}}\|/\|\mathbf{x}\|$.
  Since $\textcolor{#16a34a}{\mathbf{g}}$ is orthogonal to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$ and $\mathbf{x}-\textcolor{#16a34a}{\mathbf{g}}$ lies in it, $f$ is the sine of the angle between $\mathbf{x}$ and $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$. It lies in $[0,1]$. It equals $1$ when $\mathbf{x}$ is orthogonal to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$ and there is nothing to remove, the best case. It is close to $0$ when $\mathbf{x}$ is close to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, the bad case, because the computed $\mathbf{q}_{j+1}$ then loses orthogonality to the earlier basis vectors, by an amount that grows as $f$ shrinks, as shown next. Gram–Schmidt removes the terms $\mathbf{x}-\textcolor{#16a34a}{\mathbf{g}}$ from $\mathbf{x}$. In double precision, whose unit roundoff is $u\approx1.1\cdot10^{-16}$, it does so with rounding errors <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="gram-schmidt-rounding" aria-label="Explain why the rounding errors of Gram–Schmidt are proportional to the length of x">of the order of $u\|\mathbf{x}\|$</button><span id="gram-schmidt-rounding" class="explanation-popover" popover="auto" role="note" aria-label="Rounding errors of one Gram–Schmidt step" data-label="Derivation">Gram–Schmidt computes the coefficients $c_i=\mathbf{q}_i^\top\mathbf{x}$ of $\mathbf{x}$ on the basis vectors of $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$ and subtracts the nonzero terms $c_i\mathbf{q}_i$ from $\mathbf{x}$ one at a time. There are $n=2$ of them for $\mathbf{A}\mathbf{q}_j$ and $n=j$ for $\mathbf{A}^j\mathbf{q}_1$. Write $\mathbf{y}_k$ for the intermediate vector left after the first $k$ subtractions, so that $\mathbf{y}_0=\mathbf{x}$ and $\mathbf{y}_n=\textcolor{#16a34a}{\mathbf{g}}$. In floating-point arithmetic, every addition, subtraction and multiplication is rounded: its computed result is the exact one times $1+\delta$, with $|\delta|\le u$. Each operation therefore errs by at most $u$ times the size of its result, and the results here have the size of $\mathbf{x}$, not of $\textcolor{#16a34a}{\mathbf{g}}$. Indeed, $|c_i|\le\|\mathbf{q}_i\|\|\mathbf{x}\|=\|\mathbf{x}\|$ by the Cauchy–Schwarz inequality, and the vectors $\mathbf{y}_k$ are no longer than $\mathbf{x}$. The difference $\mathbf{x}-\mathbf{y}_k$, the sum of the first $k$ terms, is the orthogonal projection of $\mathbf{x}$ onto the span of the first $k$ basis vectors used, and $\mathbf{y}_k$ is orthogonal to this span. Pythagoras' theorem therefore gives $\|\mathbf{x}\|^2=\|\mathbf{x}-\mathbf{y}_k\|^2+\|\mathbf{y}_k\|^2$, so $\|\mathbf{y}_k\|\le\|\mathbf{x}\|$ (the triangle inequality would only give $\|\mathbf{y}_k\|\le(k+1)\|\mathbf{x}\|$ because it ignores that the terms $c_i\mathbf{q}_i$ are orthogonal). An inner product of two vectors of $\mathbb{R}^d$ sums $d$ rounded products, so each $c_i$ is computed with an error of at most about $d\,u\,\|\mathbf{x}\|$, and each subtraction of a term adds at most about $2u\|\mathbf{x}\|$, one $u\|\mathbf{x}\|$ for forming the term and one for subtracting it from $\mathbf{y}_{k-1}$. Up to terms of order $u^2$, the computed green term $\widehat{\mathbf{g}}$ therefore satisfies <span class="explanation-popover__equation">$\|\widehat{\mathbf{g}}-\textcolor{#16a34a}{\mathbf{g}}\|\le n\,(d+2)\,u\,\|\mathbf{x}\|.$</span> The factor $n(d+2)$ is a worst case, and actual errors are usually much smaller, because rounding errors of different signs partly cancel. The number of terms makes this bound $j/2$ times larger for $\mathbf{A}^j\mathbf{q}_1$ than for $\mathbf{A}\mathbf{q}_j$, which costs only $\log_{10}(j/2)$ digits, one digit at $j=20$. The large difference between the two vectors comes from the ratio $f=\|\textcolor{#16a34a}{\mathbf{g}}\|/\|\mathbf{x}\|$, not from $n$: at $j=20$, in the example below, $f$ is $8\cdot10^{-12}$ for $\mathbf{A}^{20}\mathbf{q}_1$ and $0.4$ for $\mathbf{A}\mathbf{q}_{20}$. What matters here is that the bound is proportional to $\|\mathbf{x}\|$, not to $\|\textcolor{#16a34a}{\mathbf{g}}\|$: a small $\textcolor{#16a34a}{\mathbf{g}}$ does not come with a small error.</span></span>, and these errors stay in the computed $\textcolor{#16a34a}{\mathbf{g}}$. Relative to $\|\textcolor{#16a34a}{\mathbf{g}}\|=f\|\mathbf{x}\|$, they are therefore of the order of $u/f$, and so are the components that the computed $\mathbf{q}_{j+1}$ keeps on the earlier basis vectors. This loss of accuracy is called *cancellation*. The same thing happens with a number in place of the vector $\mathbf{x}$. Subtracting $1$ from $x=1.000001$ leaves $\textcolor{#16a34a}{g}=10^{-6}$, so $f=\textcolor{#16a34a}{g}/x\approx10^{-6}$, and the error of about $10^{-16}$ with which $x$ is stored stays in $\textcolor{#16a34a}{g}$, where it is a relative error of about $u/f=10^{-10}$.

  The two ratios behave very differently. For $\mathbf{x}=\mathbf{A}\mathbf{q}_j$, the three terms are orthogonal, so $\|\mathbf{x}\|^2=\beta_{j-1}^2+\alpha_j^2+\beta_j^2$ and

  $$
  f=\frac{\beta_j}{\sqrt{\beta_{j-1}^2+\alpha_j^2+\beta_j^2}},
  $$

  which depends only on the coefficients of step $j$, and is close to $0$ only if $\beta_j$ is small compared with $\sqrt{\beta_{j-1}^2+\alpha_j^2}$. For $\mathbf{x}=\mathbf{A}^j\mathbf{q}_1$, the green term has length $\beta_1\beta_2\cdots\beta_j$, so

  $$
  f=\frac{\beta_1\beta_2\cdots\beta_j}{\|\mathbf{A}^j\mathbf{q}_1\|}.
  $$

  Up to scaling, the powers $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$ are the iterates of power iteration started from $\mathbf{q}_1$ ([Section 3](#3-power-iteration)), which, under the hypotheses of [Theorem 1](#32-theorem-1-rate-of-convergence), converge up to sign to the dominant eigenvector, so the angle between $\mathbf{A}^j\mathbf{q}_1$ and the previous power $\mathbf{A}^{j-1}\mathbf{q}_1$, which lies in $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, tends to $0$. As a result, this ratio <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="powers-ratio-bound" aria-label="Prove that the ratio f for the powers shrinks geometrically">shrinks at least geometrically with $j$</button><span id="powers-ratio-bound" class="explanation-popover" popover="auto" role="note" aria-label="Geometric bound on the ratio f for the powers" data-label="Derivation">Write $\mathbf{v}_k=\mathbf{A}^k\mathbf{q}_1/\|\mathbf{A}^k\mathbf{q}_1\|$ for the iterates of power iteration started from $\mathbf{q}_1$, and $s=\operatorname{sgn}(\lambda_\star)$. The green term of $\mathbf{v}_j$ has length $f$ and is orthogonal to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, so $f$ is the distance from $\mathbf{v}_j$ to this subspace, which contains $s\,\mathbf{v}_{j-1}$. Under the hypotheses of [Theorem 1](#32-theorem-1-rate-of-convergence), $s^k\mathbf{v}_k$ lies within $(\rho_2/|\lambda_\star|)^k\tan(\phi_0)$ of $\mathbf{q}_\star$ for every $k$, and since $s^2=1$, <span class="explanation-popover__equation">$f\le\|\mathbf{v}_j-s\,\mathbf{v}_{j-1}\|=\|s^j\mathbf{v}_j-s^{j-1}\mathbf{v}_{j-1}\|\le\|s^j\mathbf{v}_j-\mathbf{q}_\star\|+\|\mathbf{q}_\star-s^{j-1}\mathbf{v}_{j-1}\|\le2\left(\dfrac{\rho_2}{|\lambda_\star|}\right)^{j-1}\tan(\phi_0).$</span> This bound uses only the last power in $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$. Using all of them gives the much faster decrease seen in practice.</span></span>, and in practice much faster, since $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$ contains all the earlier powers, not only the last one. For example, we ran both methods in double precision on the diagonal matrix of size $1000$ with eigenvalues drawn uniformly from $[0,1]$, from a random unit vector $\mathbf{q}_1$, and measured the ratios:

  | step $j$ | $1$ | $5$ | $10$ | $20$ |
  |---|:---:|:---:|:---:|:---:|
  | $f$ for $\mathbf{A}^j\mathbf{q}_1$ | $0.5$ | $4\cdot10^{-3}$ | $6\cdot10^{-6}$ | $8\cdot10^{-12}$ |
  | $f$ for $\mathbf{A}\mathbf{q}_j$ | $0.5$ | $0.4$ | $0.4$ | $0.4$ |

  The same runs also measured how far each computed basis is from orthonormal. Let $\mathbf{Q}_m$ be the matrix whose columns are the first $m$ computed basis vectors. The entries of $\mathbf{Q}_m^\top\mathbf{Q}_m-\mathbb{I}_m$ are the numbers $\mathbf{q}_i^\top\mathbf{q}_k-\delta_{ik}$, so its norm is $0$ for an exactly orthonormal basis, and close to $1$ when two of the basis vectors are nearly parallel. The measured norms are:

  | number $m$ of basis vectors | $10$ | $20$ | $25$ | $30$ |
  |---|:---:|:---:|:---:|:---:|
  | basis from the powers $\mathbf{A}^j\mathbf{q}_1$ | $9\cdot10^{-11}$ | $4\cdot10^{-3}$ | $0.9$ | $0.9$ |
  | basis from the Lanczos recurrence, without reorthogonalization | $3\cdot10^{-15}$ | $5\cdot10^{-15}$ | $5\cdot10^{-15}$ | $6\cdot10^{-15}$ |

  The basis built from the powers thus loses its orthogonality step by step and is far from orthonormal after $25$ vectors, while the Lanczos basis stays orthonormal to within $10^{-14}$.
* Second, it skips the coefficients that vanish. Full Gram–Schmidt on $\mathbf{A}\mathbf{q}_j$ computes the $j$ coefficients $\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$, $i=1,\dots,j$, and subtracts the $j$ terms, which costs $O(jd)$ operations at step $j$, a cost that grows with $j$. By symmetry, the coefficients with $i\le j-2$ are zero, as shown in the general step above. The Lanczos method therefore computes only $\alpha_j$, reuses $\beta_{j-1}$, and subtracts two terms, so that every step costs one product and $O(d)$ further operations, whatever $j$.
  This saving has a price in floating-point arithmetic. The proof that $\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j=0$ for $i\le j-2$ uses that $\mathbf{q}_j$ is exactly orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$, and the computed basis vectors are orthogonal only up to rounding errors. The skipped coefficients are therefore small but not zero, and since the method does not remove them, they stay in the computed $\mathbf{r}_j$. For $i\le j-2$, the new basis vector $\mathbf{q}_{j+1}=\mathbf{r}_j/\beta_j$ then has, up to the rounding errors of step $j$ itself, the component

  $$
  \mathbf{q}_i^\top\mathbf{q}_{j+1}
  =\frac{\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j-\beta_{j-1}\,\mathbf{q}_i^\top\mathbf{q}_{j-1}-\alpha_j\,\mathbf{q}_i^\top\mathbf{q}_j}{\beta_j}
  $$

  on the earlier vector $\mathbf{q}_i$, which is zero in exact arithmetic. Its numerator is made of inner products between earlier basis vectors: $\mathbf{q}_i^\top\mathbf{q}_{j-1}$ and $\mathbf{q}_i^\top\mathbf{q}_j$ directly, and $\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j=(\mathbf{A}\mathbf{q}_i)^\top\mathbf{q}_j$ through the recurrence $\mathbf{A}\mathbf{q}_i=\beta_{i-1}\mathbf{q}_{i-1}+\alpha_i\mathbf{q}_i+\beta_i\mathbf{q}_{i+1}$, up to rounding. The loss of orthogonality of the previous steps thus feeds into every new basis vector, and it accumulates. The same runs as in the first item measured $\lVert\mathbf{Q}_m^\top\mathbf{Q}_m-\mathbb{I}_m\rVert$ for the Lanczos basis, without reorthogonalization and with the reorthogonalization of [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks) described below:

  | number $m$ of basis vectors | $30$ | $100$ | $150$ | $200$ |
  |---|:---:|:---:|:---:|:---:|
  | without reorthogonalization | $6\cdot10^{-15}$ | $3\cdot10^{-12}$ | $6\cdot10^{-10}$ | $1\cdot10^{-6}$ |
  | with reorthogonalization | $9\cdot10^{-16}$ | $1\cdot10^{-15}$ | $1\cdot10^{-15}$ | $1\cdot10^{-15}$ |

  Without reorthogonalization, the loss is much slower than for the powers, where it reached $0.9$ after $25$ vectors, but it does not stop. This is why [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks) below reorthogonalizes: after the three-term step, it removes from $\mathbf{r}_j$ its components on all the stored basis vectors $\mathbf{q}_1,\dots,\mathbf{q}_j$ once more, here in two passes, and the basis then stays orthonormal to about $10^{-15}$ at every $m$ of the table. This gives back the $O(jd)$ cost per step that symmetry had saved. Unlike Gram–Schmidt on the powers, however, this extra pass is accurate. The components it removes have the size of the loss of orthogonality, tiny compared with $\|\mathbf{r}_j\|$, so the ratio $f$ of the first item is close to $1$ for it.

**Eigenvalues and eigenvectors from the coefficients.** After $m$ steps, the coefficients fill the $m\times m$ tridiagonal matrix

$$
\mathbf{T}_m=\begin{bmatrix}
\alpha_1&\beta_1&&&\\
\beta_1&\alpha_2&\beta_2&&\\
&\ddots&\ddots&\ddots&\\
&&\beta_{m-2}&\alpha_{m-1}&\beta_{m-1}\\
&&&\beta_{m-1}&\alpha_m
\end{bmatrix},
$$

and the $m$ recurrences $\mathbf{A}\mathbf{q}_j=\beta_{j-1}\mathbf{q}_{j-1}+\alpha_j\mathbf{q}_j+\beta_j\mathbf{q}_{j+1}$, for $j=1,\dots,m$, combine into <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="lanczos-matrix-identity" aria-label="Explain why the recurrences combine into this matrix identity">one matrix identity</button><span id="lanczos-matrix-identity" class="explanation-popover" popover="auto" role="note" aria-label="Checking the matrix identity column by column" data-label="Derivation">Compare the columns. Column $j$ of $\mathbf{A}\mathbf{Q}_m$ is $\mathbf{A}\mathbf{q}_j$. Column $j$ of $\mathbf{Q}_m\mathbf{T}_m$ is $\mathbf{Q}_m$ times column $j$ of $\mathbf{T}_m$, whose only nonzero entries are $\beta_{j-1}$, $\alpha_j$ and $\beta_j$, in rows $j-1$, $j$ and $j+1$. It is therefore $\beta_{j-1}\mathbf{q}_{j-1}+\alpha_j\mathbf{q}_j+\beta_j\mathbf{q}_{j+1}$, without the first term when $j=1$ since $\beta_0=0$, which is $\mathbf{A}\mathbf{q}_j$ by the recurrence. The last column is the exception: $\mathbf{T}_m$ has no row $m+1$, so column $m$ of $\mathbf{Q}_m\mathbf{T}_m$ stops at $\beta_{m-1}\mathbf{q}_{m-1}+\alpha_m\mathbf{q}_m$ and misses the term $\beta_m\mathbf{q}_{m+1}$ of the recurrence. The <span style="color:#16a34a">green matrix</span> $\beta_m\mathbf{q}_{m+1}\mathbf{e}_m^\top$ adds it back: its columns are zero except the last, which is $\beta_m\mathbf{q}_{m+1}$.</span></span>:

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\textcolor{#16a34a}{\beta_m\mathbf{q}_{m+1}\mathbf{e}_m^\top},
$$

where $\mathbf{e}_m$ is the last standard basis vector of $\mathbb{R}^m$. Inside $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_m\}$, the matrix $\mathbf{A}$ acts on coordinates as the small matrix $\mathbf{T}_m$, except for the <span style="color:#16a34a">green term</span>, through which the last basis vector leaks out of the subspace. Every eigenpair $(\vartheta,\mathbf{y})$ of $\mathbf{T}_m$, with $\|\mathbf{y}\|=1$, therefore gives a candidate eigenpair $(\vartheta,\mathbf{Q}_m\mathbf{y})$ of $\mathbf{A}$. Such a candidate is called a **Ritz pair**: the **Ritz value** $\vartheta$ estimates an eigenvalue of $\mathbf{A}$, and the **Ritz vector** $\mathbf{Q}_m\mathbf{y}$, a unit vector of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, estimates a corresponding eigenvector. [Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals) states their properties. In short,

$$
\underbrace{\mathbf{A}}_{d\times d}
\;\longrightarrow\;
\underbrace{\mathbf{T}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m}_{m\times m}
\;\longrightarrow\;
(\vartheta,\mathbf{y})
\;\longrightarrow\;
(\vartheta,\mathbf{Q}_m\mathbf{y}).
$$

Multiplying the matrix identity $\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\textcolor{#16a34a}{\beta_m\mathbf{q}_{m+1}\mathbf{e}_m^\top}$ above on the right by $\mathbf{y}$ gives

$$
\mathbf{A}\mathbf{Q}_m\mathbf{y}
=\mathbf{Q}_m\mathbf{T}_m\mathbf{y}+\textcolor{#16a34a}{\beta_m\mathbf{q}_{m+1}\,(\mathbf{e}_m^\top\mathbf{y})}
=\vartheta\,\mathbf{Q}_m\mathbf{y}+\textcolor{#16a34a}{\beta_m\,y_m\,\mathbf{q}_{m+1}}.
$$

The residual of the Ritz pair is therefore exactly the <span style="color:#16a34a">leak</span>, and since $\mathbf{q}_{m+1}$ is a unit vector,

$$
\bigl\|\mathbf{A}\mathbf{Q}_m\mathbf{y}-\vartheta\,\mathbf{Q}_m\mathbf{y}\bigr\|=\beta_m|y_m| .
$$

The residual is available without any further product, and [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) turns it into a certificate. The basis vectors are building blocks, not approximate eigenvectors. An approximate eigenvector $\mathbf{Q}_m\mathbf{y}$ combines all of them, so the basis must be stored to form it, whereas the Ritz values need only the numbers $\alpha_j$ and $\beta_j$.

Gram–Schmidt also explains why the Ritz values approach eigenvalues. The argument has three steps.
* First, the green term of $\mathbf{A}^m\mathbf{q}_1$ has the smallest Euclidean norm among the vectors of a certain form. As shown above with $j=m$,
  $\mathbf{A}^m\mathbf{q}_1=\textcolor{#16a34a}{\beta_1\beta_2\cdots\beta_m\,\mathbf{q}_{m+1}}+\mathbf{k}_m$ with $\mathbf{k}_m\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$.
  Every $\mathbf{k}\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ is a combination of $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{m-1}\mathbf{q}_1$, say $\mathbf{k}=\sum_{i=0}^{m-1}a_i\mathbf{A}^i\mathbf{q}_1$, so
  $
  \mathbf{A}^m\mathbf{q}_1-\mathbf{k}
  %=\mathbf{A}^m\mathbf{q}_1-\sum_{i=0}^{m-1}a_i\mathbf{A}^i\mathbf{q}_1
  =\pi(\mathbf{A})\mathbf{q}_1
  $
  with $\pi(t)=t^m-\sum_{i=0}^{m-1}a_i\,t^i$ a <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="monic-polynomial" aria-label="Define a monic polynomial">monic</button><span id="monic-polynomial" class="explanation-popover" popover="auto" role="note" aria-label="Monic polynomial" data-label="Definition">A polynomial of degree $m$ is called monic if its coefficient of $t^m$ is $1$, i.e. if it has the form $t\mapsto t^m+\sum_{i=m-1}^{0}c_i\,t^i$.</span></span> of degree $m$. Conversely, any monic polynomial $t^m+\sum_{i=0}^{m-1}c_i\,t^i$ of degree $m$ arises in this way from $a_i=-c_i$. The vectors $\mathbf{A}^m\mathbf{q}_1-\mathbf{k}$ with $\mathbf{k}\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ are therefore exactly the vectors $\pi(\mathbf{A})\mathbf{q}_1$ with $\pi$ monic of degree $m$. Each of them is the green term $\textcolor{#16a34a}{\beta_1\beta_2\cdots\beta_m\,\mathbf{q}_{m+1}}$ plus $\mathbf{k}_m-\mathbf{k}\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, two orthogonal vectors, so Pythagoras' theorem gives $\|\mathbf{A}^m\mathbf{q}_1-\mathbf{k}\|^2=(\beta_1\cdots\beta_m)^2+\|\mathbf{k}_m-\mathbf{k}\|^2$, which is smallest for $\mathbf{k}=\mathbf{k}_m$, where the vector is the green term itself. Among the vectors $\pi(\mathbf{A})\mathbf{q}_1$ with $\pi$ monic of degree $m$, the green term thus has the smallest Euclidean norm $\|\pi(\mathbf{A})\mathbf{q}_1\|$. Its own polynomial follows from $\mathbf{q}_{m+1}=p_m(\mathbf{A})\mathbf{q}_1$: the green term is $\pi_m(\mathbf{A})\mathbf{q}_1$ for $\pi_m(t)=\beta_1\cdots\beta_m\,p_m(t)$, which is monic because $p_m$ has leading coefficient $1/(\beta_1\cdots\beta_m)$. Hence $\pi_m$ minimizes $\|\pi(\mathbf{A})\mathbf{q}_1\|$ over the monic polynomials of degree $m$. Recall the [inner product on polynomials](#krylov-inner-product) defined with the three-term recurrence, $\langle p,r\rangle=\bigl(p(\mathbf{A})\mathbf{q}_1\bigr)^\top r(\mathbf{A})\mathbf{q}_1=\sum_\lambda p(\lambda)\,r(\lambda)\,\|\Pi_\lambda\mathbf{q}_1\|^2$. It gives $\|\pi(\mathbf{A})\mathbf{q}_1\|^2=\langle\pi,\pi\rangle$, so this Euclidean norm of the vector $\pi(\mathbf{A})\mathbf{q}_1$ is also the norm of the polynomial $\pi$ for that inner product, and with its expression as a sum over the eigenvalues, the minimization reads

  $$
  \sum_{\lambda\in\Lambda(\mathbf{A})}\pi_m(\lambda)^2\,\|\Pi_\lambda\mathbf{q}_1\|^2
  =\min_{\pi\ \text{monic},\ \deg\pi=m}\ \sum_{\lambda\in\Lambda(\mathbf{A})}\pi(\lambda)^2\,\|\Pi_\lambda\mathbf{q}_1\|^2 .
  $$

  In words, $\pi_m$ is the monic polynomial of degree $m$ that is smallest on the eigenvalues of $\mathbf{A}$, each eigenvalue weighted by $\|\Pi_\lambda\mathbf{q}_1\|^2$, the squared length of the component of $\mathbf{q}_1$ in its eigenspace.
* Second, the Ritz values at step $m$ are exactly the roots of $\pi_m$, because <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="ritz-characteristic-polynomial" aria-label="Prove that the characteristic polynomial of T_m is pi_m">$\det(t\,\mathbb{I}_m-\mathbf{T}_m)=\pi_m(t)$</button><span id="ritz-characteristic-polynomial" class="explanation-popover" popover="auto" role="note" aria-label="The characteristic polynomial of the tridiagonal matrix" data-label="Derivation">Let $D_k(t)=\det(t\,\mathbb{I}_k-\mathbf{T}_k)$, with $D_0=1$ and $D_{-1}=0$. The matrix $t\,\mathbb{I}_k-\mathbf{T}_k$ is tridiagonal, and its last row has only two nonzero entries, $-\beta_{k-1}$ and $t-\alpha_k$. Expanding the determinant along this row, and the remaining minor along its last column, gives <span class="explanation-popover__equation">$D_k(t)=(t-\alpha_k)\,D_{k-1}(t)-\beta_{k-1}^2\,D_{k-2}(t).$</span> Multiplying the recurrence $\beta_k\,p_k=(t-\alpha_k)\,p_{k-1}-\beta_{k-1}\,p_{k-2}$ by $\beta_1\cdots\beta_{k-1}$ shows that $P_k=\beta_1\cdots\beta_k\,p_k$ satisfies the same recurrence, <span class="explanation-popover__equation">$P_k(t)=(t-\alpha_k)\,P_{k-1}(t)-\beta_{k-1}^2\,P_{k-2}(t),$</span> with the same starting values $P_0=1$ and $P_{-1}=0$. Hence $D_k=P_k$ for every $k$, and in particular $\det(t\,\mathbb{I}_m-\mathbf{T}_m)=P_m=\pi_m$.</span></span>. If $\vartheta_1,\dots,\vartheta_m$ are the Ritz values, then $\pi_m(t)=\prod_{i=1}^m(t-\vartheta_i)$.
* Third, the two steps combine into one statement. Any $m$ real numbers $z_1,\dots,z_m$ define the monic polynomial $\prod_{i=1}^m(t-z_i)$ of degree $m$, and $\pi_m$ is of this form with $z_i=\vartheta_i$, since $\mathbf{T}_m$ is symmetric and its eigenvalues are real. Since $\pi_m$ minimizes over all monic polynomials of degree $m$, the Ritz values minimize in particular, over all $(z_1,\dots,z_m)\in\mathbb{R}^m$, the sum

  $$
  \sum_{\lambda\in\Lambda(\mathbf{A})}\|\Pi_\lambda\mathbf{q}_1\|^2\prod_{i=1}^m(\lambda-z_i)^2 .
  $$

  Each term is the weight of an eigenvalue $\lambda$ times the product of its squared distances to $z_1,\dots,z_m$. A term with a large weight is reduced by placing one of the $z_i$ close to its eigenvalue, and this is why the Ritz values approach the eigenvalues that $\mathbf{q}_1$ contains. Two statements make this exact. When $m$ equals the number $s$ of eigenvalues with $\Pi_\lambda\mathbf{q}_1\neq0$, choosing the $z_i$ equal to these eigenvalues makes the sum zero, so the minimum is zero, $\beta_1\cdots\beta_s=0$, hence $\beta_s=0$, and the Ritz values are exactly these eigenvalues ([Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run)). Before that, [Lemma 5](#48-lemma-5-variational-characterization-and-monotonicity) shows that the extreme Ritz values move monotonically toward $\lambda_{\min}(\mathbf{A})$ and $\lambda_{\max}(\mathbf{A})$.

### 4.4 A three-dimensional example

The steps of [Section 4.3](#43-how-the-lanczos-method-works) can be followed with numbers on a small matrix. Take

$$
\mathbf{A}=\begin{bmatrix}1&0&0\\0&2&0\\0&0&3\end{bmatrix},
\qquad
\mathbf{q}_1=\frac{1}{\sqrt{3}}\begin{bmatrix}1\\1\\1\end{bmatrix}.
$$

**Step 1.** The candidate $\mathbf{A}\mathbf{q}_1=\frac{1}{\sqrt3}[1,2,3]^\top$ has the coefficient $\alpha_1=\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_1=2$ on $\mathbf{q}_1$, and there is no $\mathbf{q}_0$ to remove. The rest, $\mathbf{r}_1=\frac{1}{\sqrt3}[-1,0,1]^\top$, has length $\beta_1=\sqrt{2/3}$ and gives $\mathbf{q}_2=\frac{1}{\sqrt2}[-1,0,1]^\top$.

**Step 2.** The candidate $\mathbf{A}\mathbf{q}_2=\frac{1}{\sqrt2}[-1,0,3]^\top$ has the coefficient $\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_2=\sqrt{2/3}=\beta_1$ on $\mathbf{q}_1$, as symmetry predicts, and the coefficient $\alpha_2=2$ on $\mathbf{q}_2$. Removing both leaves $\mathbf{r}_2=\frac{1}{3\sqrt2}[1,-2,1]^\top$, of length $\beta_2=1/\sqrt3$, which gives $\mathbf{q}_3=\frac{1}{\sqrt6}[1,-2,1]^\top$.

**Step 3.** The candidate $\mathbf{A}\mathbf{q}_3=\frac{1}{\sqrt6}[1,-4,3]^\top$ minus $\beta_2\mathbf{q}_2$ equals $\frac{1}{\sqrt6}[2,-4,2]^\top=2\mathbf{q}_3$. Hence $\alpha_3=2$ and $\mathbf{r}_3=0$, an exact breakdown: $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$ is already all of $\mathbb{R}^3$, so no new direction exists.

The small matrices are

$$
\mathbf{T}_1=\begin{bmatrix}2\end{bmatrix},
\qquad
\mathbf{T}_2=\begin{bmatrix}2&\sqrt{2/3}\\\sqrt{2/3}&2\end{bmatrix},
\qquad
\mathbf{T}_3=\begin{bmatrix}2&\sqrt{2/3}&0\\\sqrt{2/3}&2&1/\sqrt3\\0&1/\sqrt3&2\end{bmatrix},
$$

and their eigenvalues, the Ritz values, are $\{2\}$, then $\{2-\sqrt{2/3},\,2+\sqrt{2/3}\}\approx\{1.18,\,2.82\}$, then $\{1,2,3\}$. The extreme Ritz values move outward toward the extreme eigenvalues $1$ and $3$, as [Lemma 5](#48-lemma-5-variational-characterization-and-monotonicity) guarantees. At $m=3$, they are exactly the eigenvalues of $\mathbf{A}$, as [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) guarantees for a starting vector with a nonzero component on every eigenvector. At $m=2$, the eigenvectors of $\mathbf{T}_2$ are $\frac{1}{\sqrt2}[1,\pm1]^\top$, so both Ritz pairs have residual $\beta_2|y_2|=\frac{1}{\sqrt3}\cdot\frac{1}{\sqrt2}\approx0.41$ ([Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals)). This bounds their actual distances to the spectrum, $0.18$ in both cases ([Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair)). For comparison, power iteration from the same $\mathbf{q}_1$ estimates $\lambda_{\max}(\mathbf{A})=3$ after the same two products by $\mathbf{v}_1^\top\mathbf{A}\mathbf{v}_1=36/14\approx2.57$, with $\mathbf{v}_1=\mathbf{A}\mathbf{q}_1/\|\mathbf{A}\mathbf{q}_1\|$, which is farther from $3$ than the Ritz value $2.82$. In the polynomial reading of [Section 4.3](#43-how-the-lanczos-method-works), the Ritz values $2\pm\sqrt{2/3}$ are the roots of $\pi_2(t)=(t-2)^2-2/3$, the monic quadratic that makes $\|\pi(\mathbf{A})\mathbf{q}_1\|$ smallest, and at $m=3$ the polynomial $(t-1)(t-2)(t-3)$ makes it zero, which is the breakdown.

### 4.5 Definition 2 [Lanczos recurrence and exact breakdown]

The Lanczos construction starts with

$$
\mathbf{q}_0=0,
\qquad
\beta_0=0,
\qquad
\|\mathbf{q}_1\|=1 .
$$

At iteration $j\ge1$, compute

$$
\begin{aligned}
\mathbf{z}_j&=\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1},\\
\alpha_j&=\mathbf{q}_j^\top\mathbf{z}_j,\\
\mathbf{r}_j&=\mathbf{z}_j-\alpha_j\mathbf{q}_j,\\
\beta_j&=\|\mathbf{r}_j\| .
\end{aligned}
$$

If $\beta_j>0$, define $\mathbf{q}_{j+1}=\mathbf{r}_j/\beta_j$ and continue. If $\beta_j=0$, stop the current run. We refer to this event as an **exact breakdown**. As shown below, it means that the generated subspace is invariant under $\mathbf{A}$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

The vectors $\mathbf{q}_j$ are basis vectors, not generally eigenvectors of $\mathbf{A}$. Each step orthogonalizes $\mathbf{A}\mathbf{q}_j$ against the two most recent basis vectors only, for the reasons given in [Section 4.3](#43-how-the-lanczos-method-works). This <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="three-term-recurrence" aria-label="Explain why a three-term recurrence suffices">three-term recurrence</button><span id="three-term-recurrence" class="explanation-popover" popover="auto" role="note" aria-label="Three-term recurrence" data-label="Linear algebra">Orthogonalizing $\mathbf A\mathbf q_j$ against every previous basis vector, as in Gram–Schmidt, would cost more at every step. For symmetric $\mathbf A$, the coefficients against $\mathbf q_1,\dots,\mathbf q_{j-2}$ vanish ([Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection)), so the two coefficients $\alpha_j$ and $\beta_{j-1}$ suffice.</span></span> is enough because $\mathbf{A}$ is symmetric. [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection) shows that, in exact arithmetic, $\mathbf{A}\mathbf{q}_j$ is automatically orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_{j-2}$, so the projected matrix is tridiagonal and each step costs one product plus $O(d)$ further operations.

### 4.6 Lemma 3 [Orthogonality and tridiagonal projection]

Before and including the first exact breakdown, the Lanczos vectors satisfy

$$
\mathbf{q}_i^\top\mathbf{q}_j=\delta_{ij},
\qquad
\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_m\}=\mathcal{K}_m(\mathbf{A},\mathbf{q}_1).
$$

Let $\mathbf{Q}_m=[\mathbf{q}_1,\dots,\mathbf{q}_m]$, and let $\mathbf{T}_m\in\mathbb{R}^{m\times m}$ be the symmetric tridiagonal matrix

$$
\mathbf{T}_m=
\begin{bmatrix}
\alpha_1&\beta_1&&&\\
\beta_1&\alpha_2&\beta_2&&\\
&\ddots&\ddots&\ddots&\\
&&\beta_{m-2}&\alpha_{m-1}&\beta_{m-1}\\
&&&\beta_{m-1}&\alpha_m
\end{bmatrix},
$$

that is, $(\mathbf{T}_m)_{ii}=\alpha_i$ for $1\le i\le m$, $(\mathbf{T}_m)_{i,i+1}=(\mathbf{T}_m)_{i+1,i}=\beta_i$ for $1\le i<m$, and all other entries equal to zero. Then

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\mathbf{r}_m\mathbf{e}_m^\top .
$$

Moreover,

$$
\mathbf{Q}_m^\top\mathbf{r}_m=0,
\qquad
\mathbf{T}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m .
$$

#### Remark 12 [What the tridiagonal matrix represents]

The identity $\mathbf{T}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m$ says that $\mathbf{T}_m$ represents $\mathbf{A}$ compressed to the Krylov subspace, in the orthonormal basis $\mathbf{q}_1,\dots,\mathbf{q}_m$. The rank-one term $\mathbf{r}_m\mathbf{e}_m^\top$ measures how far that subspace is from being invariant: only the image of the last basis vector leaves it.

#### Proof

**Orthogonality.** We argue by induction. The initial vector is normalized by assumption. Suppose that $\mathbf{q}_1,\dots,\mathbf{q}_j$ are orthonormal and that the previous iterations have been completed. We show that $\mathbf{r}_j$ is orthogonal to each $\mathbf{q}_i$ with $i\le j$.

*The case $i=j$.* Since $\mathbf{q}_j^\top\mathbf{q}_{j-1}=0$, with $\mathbf{q}_0=0$ when $j=1$, we have
$\alpha_j=\mathbf{q}_{j}^\top\mathbf{z}_j=\mathbf{q}_j^\top\bigl(\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}\bigr)=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$.
Therefore,
$\mathbf{q}_j^\top\mathbf{r}_j
=\mathbf{q}_j^\top\mathbf{z}_j-\alpha_j\mathbf{q}_j^\top\mathbf{q}_j
=\alpha_j-\alpha_j
=0$.

*The case $i=j-1$, for $j\ge2$.* The previous recurrence gives
$\mathbf{A}\mathbf{q}_{j-1}=\beta_{j-2}\mathbf{q}_{j-2}+\alpha_{j-1}\mathbf{q}_{j-1}+\beta_{j-1}\mathbf{q}_j$.
Using symmetry,
$\mathbf{q}_{j-1}^\top\mathbf{r}_j
=\mathbf{q}_{j-1}^\top\mathbf{z}_j-\alpha_j\mathbf{q}_{j-1}^\top\mathbf{q}_j
=\mathbf{q}_{j-1}^\top \bigl(\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}\bigr) - 0
=\mathbf{q}_{j-1}^\top\mathbf{A}\mathbf{q}_j-\beta_{j-1}
=\textcolor{#2563eb}{(\mathbf{A}\mathbf{q}_{j-1})^\top\mathbf{q}_j}-\beta_{j-1}
=\beta_{j-1}-\beta_{j-1}
=0$.

*The case $i\le j-2$.* The recurrence at step $i$ shows that
$\mathbf{A}\mathbf{q}_i\in\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{i+1}\}$.
Since $i+1\le j-1$, this vector is orthogonal to $\mathbf{q}_j$. Hence

$$
\begin{aligned}
\mathbf{q}_i^\top\mathbf{r}_j
&=\mathbf{q}_i^\top\mathbf{z}_j-\alpha_j\mathbf{q}_i^\top\mathbf{q}_j
=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j-\beta_{j-1}\underbrace{\mathbf{q}_i^\top\mathbf{q}_{j-1}}_{=0}-\alpha_j\underbrace{\mathbf{q}_i^\top\mathbf{q}_j}_{=0}
=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j\\
&=\textcolor{#2563eb}{(\mathbf{A}\mathbf{q}_i)^\top\mathbf{q}_j}
=\beta_{i-1}\underbrace{\mathbf{q}_{i-1}^\top\mathbf{q}_j}_{=0}+\alpha_i\underbrace{\mathbf{q}_i^\top\mathbf{q}_j}_{=0}+\beta_i\underbrace{\mathbf{q}_{i+1}^\top\mathbf{q}_j}_{=0}
=0 .
\end{aligned}
$$

The first line expands $\mathbf{z}_j=\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}$, and the two inner products vanish by the induction hypothesis, since $i$ differs from $j-1$ and $j$. The second line moves $\mathbf{A}$ onto $\mathbf{q}_i$ by symmetry and expands $\mathbf{A}\mathbf{q}_i=\beta_{i-1}\mathbf{q}_{i-1}+\alpha_i\mathbf{q}_i+\beta_i\mathbf{q}_{i+1}$ by the recurrence at step $i$, with $\mathbf{q}_0=0$ when $i=1$, and its three terms are orthogonal to $\mathbf{q}_j$, as noted above.
The <span style="color:#2563eb">blue terms</span> are where symmetry enters. Moving $\mathbf{A}$ from $\mathbf{q}_j$ onto an older basis vector $\mathbf{q}_i$ turns the inner product into one with $\mathbf{A}\mathbf{q}_i$, and an earlier step of the recurrence has already expanded $\mathbf{A}\mathbf{q}_i$ in the basis. This is why orthogonalizing against the two most recent vectors suffices.

Thus $\mathbf{r}_j$ is orthogonal to every column of $\mathbf{Q}_j$. If $\beta_j>0$, normalizing $\mathbf{r}_j$ extends the orthonormal family.

**Krylov subspaces.** We prove $\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_m\}=\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ by induction on $m$. It holds at $m=1$, since both sides equal $\operatorname{span}\{\mathbf{q}_1\}$. Suppose it holds at $m=j$ and $\beta_j>0$. Then 
$\beta_j \mathbf{q}_{j+1}=\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}-\alpha_j\mathbf{q}_j \in\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$
because $\mathbf{q}_{j-1}$ and $\mathbf{q}_j$ belong to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, whose image under $\mathbf{A}$ lies in $\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$. Consequently,
$\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{j+1}\}\subseteq\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$.
The left-hand side has dimension $j+1$, because its vectors are orthonormal. The right-hand side is generated by $j+1$ vectors, so its dimension is at most $j+1$. The two spaces are therefore equal.

**Matrix identity.** For each completed iteration $j<m$,
$\mathbf{A}\mathbf{q}_j=\beta_{j-1}\mathbf{q}_{j-1}+\alpha_j\mathbf{q}_j+\beta_j\mathbf{q}_{j+1}$.
For the final column,
$\mathbf{A}\mathbf{q}_m=\beta_{m-1}\mathbf{q}_{m-1}+\alpha_m\mathbf{q}_m+\mathbf{r}_m$.
Collecting these identities column by column gives
$\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\mathbf{r}_m\mathbf{e}_m^\top$.
Multiplying on the left by $\mathbf{Q}_m^\top$, and using
$\mathbf{Q}_m^\top\mathbf{Q}_m=\mathbb{I}_m$ and $\mathbf{Q}_m^\top\mathbf{r}_m=0$ gives $\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m=\mathbf{T}_m$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

The residual form of this identity remains valid when $\beta_m=0$. When $\beta_m>0$, it can also be written as 
$\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\beta_m\mathbf{q}_{m+1}\mathbf{e}_m^\top$.
No vector $\mathbf{q}_{m+1}$ needs to be defined at an exact breakdown.

### 4.7 Lemma 4 [Ritz pairs and their residuals]

Let
$$
\mathbf{T}_m\mathbf{y}=\vartheta\,\mathbf{y},
\qquad
\|\mathbf{y}\|=1,
$$
and define
$\mathbf{x}=\mathbf{Q}_m\mathbf{y}.$
Then
$$
\|\mathbf{x}\|=1,
\qquad
\mathbf{x}^\top\mathbf{A}\mathbf{x}=\vartheta,
$$

and

$$
\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}=\mathbf{r}_m\,y_m,
\qquad
\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|=\beta_m|y_m| .
$$

Here $y_m=\mathbf{e}_m^\top\mathbf{y}$ is the last coordinate of $\mathbf{y}$. In particular,

$$
\min_{\lambda\in\Lambda(\mathbf{A})}|\vartheta-\lambda|\le\beta_m|y_m| .
$$

The scalar $\vartheta$ is called a **Ritz value**, and $\mathbf{x}$ a **Ritz vector**.

#### Remark 13 [Residuals at no extra cost]

Ritz pairs are the eigenpairs of the compression of $\mathbf{A}$ to the Krylov subspace, lifted back to $\mathbb{R}^d$. Computing them is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="rayleigh-ritz" aria-label="Explain the Rayleigh-Ritz procedure">Rayleigh–Ritz procedure</button><span id="rayleigh-ritz" class="explanation-popover" popover="auto" role="note" aria-label="Rayleigh-Ritz procedure" data-label="Terminology">Approximating the eigenpairs of $\mathbf A$ by those of its compression $\mathbf Q^\top\mathbf A\mathbf Q$ to a subspace with orthonormal basis $\mathbf Q$ is the Rayleigh–Ritz procedure. Lanczos applies it to Krylov subspaces, where the compression is tridiagonal.</span></span> on a Krylov subspace. Their residuals cost nothing extra: $\beta_m$ and the last coordinate of $\mathbf{y}$ are available from the small problem. A Ritz pair can therefore be certified as accurate even when $\beta_m$ is not small, as long as $|y_m|$ is.

#### Proof

Orthonormality of the columns of $\mathbf{Q}_m$ gives
$\|\mathbf{x}\|^2
=\mathbf{y}^\top\mathbf{Q}_m^\top\mathbf{Q}_m\mathbf{y}
=\mathbf{y}^\top\mathbf{y}
=1$.
Also,
$
\mathbf{x}^\top\mathbf{A}\mathbf{x}
=\mathbf{y}^\top\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m\mathbf{y}
=\mathbf{y}^\top\mathbf{T}_m\mathbf{y}
=\vartheta\,\mathbf{y}^\top\mathbf{y}
=\vartheta$.
Using [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection),
$\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}
=\mathbf{A}\mathbf{Q}_m\mathbf{y}-\vartheta\mathbf{Q}_m\mathbf{y}
=\mathbf{Q}_m\mathbf{T}_m\mathbf{y}+\mathbf{r}_m\mathbf{e}_m^\top\mathbf{y}-\vartheta\mathbf{Q}_m\mathbf{y}
=\mathbf{Q}_m\textcolor{#2563eb}{(\mathbf{T}_m\mathbf{y}-\vartheta\mathbf{y})}+\mathbf{r}_m y_m
=\mathbf{r}_m y_m$.
The <span style="color:#2563eb">blue factor</span> vanishes because $\mathbf{y}$ is an eigenvector of $\mathbf{T}_m$. Taking norms gives
$\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|=\|\mathbf{r}_m\|\,|y_m|=\beta_m|y_m|$.

The eigenvalue bound $\min_{\lambda\in\Lambda(\mathbf{A})}|\vartheta-\lambda|\le\beta_m|y_m|$ is [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) applied to the unit vector $\mathbf{x}$, whose Rayleigh quotient is $\vartheta$.

<details class="article-disclosure">
<summary>The eigenbasis expansion, written out for the Ritz vector</summary>
<div class="article-disclosure__body">

[Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) states that every unit vector $\mathbf{q}$, with Rayleigh quotient $\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ and residual $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$, satisfies $\min_{\lambda\in\Lambda(\mathbf{A})}|\widehat{\lambda}-\lambda|\le\|\mathbf{r}\|$. Take $\mathbf{q}=\mathbf{x}$. It is a unit vector, its Rayleigh quotient is $\widehat{\lambda}=\mathbf{x}^\top\mathbf{A}\mathbf{x}=\vartheta$, and its residual is $\mathbf{r}=\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}$, all by the computations above. Lemma 2 therefore gives
$\min_{\lambda\in\Lambda(\mathbf{A})}|\vartheta-\lambda|
\le\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|
=\beta_m|y_m|$.

More specifically, expand $\mathbf{x}$ in an orthonormal eigenbasis $\mathbf{u}_1,\dots,\mathbf{u}_d$ of $\mathbf{A}$: $
\mathbf{x}=\sum_{i=1}^dc_i\mathbf{u}_i$, $\mathbf{A}\mathbf{u}_i=\lambda_i\mathbf{u}_i$, $\sum_{i=1}^dc_i^2=1$.
Then $\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}=\sum_{i=1}^dc_i(\lambda_i-\vartheta)\mathbf{u}_i$,
and therefore
$$
\begin{aligned}
\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|^2
&=\sum_{i=1}^dc_i^2(\lambda_i-\vartheta)^2\\
&\ge\left(\min_{1\le i\le d}|\lambda_i-\vartheta|^2\right)\sum_{i=1}^dc_i^2\\
&=\min_{1\le i\le d}|\lambda_i-\vartheta|^2 .
\end{aligned}
$$

Taking square roots proves the bound.

</div>
</details>

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

This bound certifies proximity to some eigenvalue of $\mathbf{A}$. By itself, a small residual does not certify that the eigenvalue is a global extremum or that all eigenvalues have been found.

### 4.8 Lemma 5 [Variational characterization and monotonicity]

Let

$$
\vartheta_{\min}^{(m)}=\lambda_{\min}(\mathbf{T}_m),
\qquad
\vartheta_{\max}^{(m)}=\lambda_{\max}(\mathbf{T}_m).
$$

Then

$$
\vartheta_{\min}^{(m)}=\min_{\substack{\mathbf{x}\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\\\|\mathbf{x}\|=1}}\mathbf{x}^\top\mathbf{A}\mathbf{x},
\qquad
\vartheta_{\max}^{(m)}=\max_{\substack{\mathbf{x}\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\\\|\mathbf{x}\|=1}}\mathbf{x}^\top\mathbf{A}\mathbf{x}.
$$

Whenever the next iteration exists,

$$
\lambda_{\min}(\mathbf{A})\le\textcolor{#16a34a}{\vartheta_{\min}^{(m+1)}}\le\textcolor{#16a34a}{\vartheta_{\min}^{(m)}},
\qquad
\textcolor{#16a34a}{\vartheta_{\max}^{(m)}}\le\textcolor{#16a34a}{\vartheta_{\max}^{(m+1)}}\le\lambda_{\max}(\mathbf{A}).
$$

#### Remark 14 [Extreme Ritz values move outward]

The <span style="color:#16a34a">extreme Ritz values</span> are the extreme Rayleigh quotients over a growing family of subspaces. They can therefore only move outward, toward the extreme eigenvalues of $\mathbf{A}$, and never overshoot them, as the <span style="color:#16a34a">green curves</span> of panel (a) in [Figure 6](#figure-lanczos) show.

#### Proof

Every vector in $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ has a unique representation
$\mathbf{x}=\mathbf{Q}_m\mathbf{y}.$
Moreover,
$\|\mathbf{x}\|^2=\mathbf{y}^\top\mathbf{Q}_m^\top\mathbf{Q}_m\mathbf{y}=\|\mathbf{y}\|^2,$
and
$\mathbf{x}^\top\mathbf{A}\mathbf{x}=\mathbf{y}^\top\mathbf{T}_m\mathbf{y}.$
Let $\mathbf{s}_1,\dots,\mathbf{s}_m$ be an orthonormal eigenbasis of $\mathbf{T}_m$, with eigenvalues ordered as $\vartheta_1\le\vartheta_2\le\dots\le\vartheta_m$, so that $\vartheta_{\min}^{(m)}=\vartheta_1$ and $\vartheta_{\max}^{(m)}=\vartheta_m$. For any unit vector
$\mathbf{y}=\sum_{i=1}^mc_i\mathbf{s}_i$, $\sum_{i=1}^mc_i^2=1$,
we have $\mathbf{y}^\top\mathbf{T}_m\mathbf{y}=\sum_{i=1}^m\vartheta_i c_i^2$.
Thus $\vartheta_1\le\mathbf{y}^\top\mathbf{T}_m\mathbf{y}\le\vartheta_m$, with equality for $\mathbf{y}=\mathbf{s}_1$ and $\mathbf{y}=\mathbf{s}_m$ respectively. This proves the two variational identities.

Since $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\subseteq\mathcal{K}_{m+1}(\mathbf{A},\mathbf{q}_1),$
minimizing over the larger set can only decrease the minimum, while maximizing over the larger set can only increase the maximum. Therefore,
$\vartheta_{\min}^{(m+1)}\le\vartheta_{\min}^{(m)}$ and $\vartheta_{\max}^{(m)}\le\vartheta_{\max}^{(m+1)}$.

Finally, every unit vector $\mathbf{x}\in\mathbb{R}^d$ satisfies
$\lambda_{\min}(\mathbf{A})\le\mathbf{x}^\top\mathbf{A}\mathbf{x}\le\lambda_{\max}(\mathbf{A}),$
as follows from the same eigenbasis expansion for $\mathbf{A}$. Apply this to the Ritz vectors $\mathbf{x}=\mathbf{Q}_m\mathbf{s}_1$ and $\mathbf{x}=\mathbf{Q}_m\mathbf{s}_m$. They are unit vectors, since $\|\mathbf{Q}_m\mathbf{s}_i\|=\|\mathbf{s}_i\|=1$, and their Rayleigh quotients are $\mathbf{s}_1^\top\mathbf{T}_m\mathbf{s}_1=\vartheta_1=\vartheta_{\min}^{(m)}$ and $\mathbf{s}_m^\top\mathbf{T}_m\mathbf{s}_m=\vartheta_m=\vartheta_{\max}^{(m)}$. Hence $\lambda_{\min}(\mathbf{A})\le\vartheta_{\min}^{(m)}$ and $\vartheta_{\max}^{(m)}\le\lambda_{\max}(\mathbf{A})$ for every $m$, and with $m+1$ in place of $m$, these are the remaining bounds.
$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

This characterization provides a precise connection with power iteration. Started from $\mathbf{v}_0=\mathbf{q}_1$, and whenever $\mathbf{A}^{m-1}\mathbf{q}_1\neq0$, the normalized power iterate
$\mathbf{v}_{m-1}=\mathbf{A}^{m-1}\mathbf{q}_1/\|\mathbf{A}^{m-1}\mathbf{q}_1\|$
belongs to $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$. Consequently, $\vartheta_{\min}^{(m)}\le\mathbf{v}_{m-1}^\top\mathbf{A}\mathbf{v}_{m-1}\le\vartheta_{\max}^{(m)}$.
Lanczos optimizes the Rayleigh quotient over all unit vectors in the available Krylov subspace. The normalized power vector is one particular member of that subspace.
Both estimates cost $m$ matrix–vector products: $\mathbf{T}_m$ needs $\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}\mathbf{q}_m$, and the Rayleigh quotient of $\mathbf{v}_{m-1}$ needs $\mathbf{A}\mathbf{v}_0,\dots,\mathbf{A}\mathbf{v}_{m-1}$. The eigenvalue of largest magnitude is $\lambda_{\max}(\mathbf{A})$ or $\lambda_{\min}(\mathbf{A})$. In the first case, $\mathbf{v}_{m-1}^\top\mathbf{A}\mathbf{v}_{m-1}\le\vartheta_{\max}^{(m)}\le\lambda_{\max}(\mathbf{A})$, and in the second, $\lambda_{\min}(\mathbf{A})\le\vartheta_{\min}^{(m)}\le\mathbf{v}_{m-1}^\top\mathbf{A}\mathbf{v}_{m-1}$. Either way, at an equal number of products, the extreme Ritz value on the side of the dominant eigenvalue is at least as close to it as the power-iteration estimate (middle panel of the figure in [Section 4.10](#410-corollary-3-random-initialization-and-repeated-eigenvalues)).

This variational statement does not imply that Lanczos has a lower total runtime for every matrix. Constructing and retaining the basis introduces additional work and storage.

### 4.9 Proposition 3 [Exact termination and the eigenvalues recovered by one run]

For each $\lambda\in\Lambda(\mathbf{A})$, let $\Pi_\lambda$ be the orthogonal projection onto the eigenspace $E_\lambda=\ker(\mathbf{A}-\lambda\mathbb{I}_d)$. Define
$\Lambda_1=\{\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{q}_1\neq0\}$ and let $s=|\Lambda_1|$.
Here $\Lambda_1$ contains distinct eigenvalues, irrespective of their multiplicities.
Then, a single Lanczos run terminates at exactly iteration $s$:
$$
\beta_j>0\quad(1\le j<s),
\qquad
\beta_s=0.
$$
Moreover, at termination, $\mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{T}_s$ and $\Lambda(\mathbf{T}_s)=\Lambda_1$. Since $\mathbf{T}_s$ has size $s=|\Lambda_1|$, each of these eigenvalues is simple for $\mathbf{T}_s$.

#### Remark 15 [One direction per distinct eigenvalue]

A run can only discover the eigenvalues present in $\mathbf{q}_1$, and it discovers each of them once, however large its multiplicity. Inside the Krylov subspace, the whole eigenspace $E_\lambda$ is represented by the single vector $\Pi_\lambda\mathbf{q}_1$. The number of steps before an exact breakdown is therefore the number of distinct eigenvalues present in $\mathbf{q}_1$, not $d$.

#### Proof

Write $\mathbf{p}_\lambda=\Pi_\lambda\mathbf{q}_1$ for $\lambda\in\Lambda_1$,
and define $W=\operatorname{span}\{\mathbf{p}_\lambda:\lambda\in\Lambda_1\}$.
The vectors $\mathbf{p}_\lambda$ are nonzero and mutually orthogonal, so $\dim W=s$.

**The Krylov subspaces lie in $W$.** Every power of $\mathbf{A}$ applied to $\mathbf{q}_1$ belongs to $W$, because $\mathbf{q}_1=\sum_{\lambda\in\Lambda_1}\mathbf{p}_\lambda$ and, for $k\ge1$,
$\mathbf{A}^k\mathbf{q}_1=\sum_{\lambda\in\Lambda_1}\lambda^k\mathbf{p}_\lambda$.
Therefore,
$\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\subseteq W$ for every $m$.

**The Krylov subspace of order $s$ equals $W$.** For each $\lambda\in\Lambda_1$, define the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="lagrange-polynomial" aria-label="Explain the Lagrange interpolation polynomial">Lagrange polynomial</button><span id="lagrange-polynomial" class="explanation-popover" popover="auto" role="note" aria-label="Lagrange interpolation polynomial" data-label="Definition">$\ell_\lambda$ is the unique polynomial of degree at most $s-1$ that equals $1$ at $\lambda$ and $0$ at the other $s-1$ points of $\Lambda_1$. Applied to $\mathbf A$, it keeps the component of $\mathbf q_1$ along $E_\lambda$ and removes every other component present in $\mathbf q_1$.</span></span>
$$
\ell_\lambda(t)=\prod_{\substack{\mu\in\Lambda_1\\\mu\neq\lambda}}\frac{t-\mu}{\lambda-\mu}.
$$
When $s=1$, the empty product is defined as $1$. This polynomial has degree at most $s-1$ and satisfies
$\ell_\lambda(\mu)=\delta_{\lambda,\mu}$ for all $\mu\in\Lambda_1$, with the Kronecker delta of the [notation](#notation).
Consequently,
$\ell_\lambda(\mathbf{A})\mathbf{q}_1
=\sum_{\mu\in\Lambda_1}\ell_\lambda(\mu)\mathbf{p}_\mu
=\mathbf{p}_\lambda$.
Since $\ell_\lambda$ has degree at most $s-1$, $\mathbf{p}_\lambda\in\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)$.
Thus $W\subseteq\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)$, and hence $\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)=W$.

**No breakdown before iteration $s$.** Suppose a breakdown occurred at some $m<s$. [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection) would give
$\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m$,
so
$\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\operatorname{range}(\mathbf{Q}_m)$
would be invariant under $\mathbf{A}$. Since it contains $\mathbf{q}_1$, it would then contain every power $\mathbf{A}^k\mathbf{q}_1$ with $k\in\{0,1,2,\dots\}$, not only the powers with $k\le m-1$ that span it by definition. In particular, it would contain $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{s-1}\mathbf{q}_1$, and therefore
$\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)\subseteq\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$.
This is impossible, because 
$\dim\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)=s>m=\dim\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$.
Thus no breakdown occurs before iteration $s$.

**Breakdown at iteration $s$.** At iteration $s$, all columns of $\mathbf{Q}_s$ belong to $W$. Since $W$ is spanned by eigenvectors, it is invariant under $\mathbf{A}$. Therefore,
$\mathbf{r}_s=\mathbf{A}\mathbf{q}_s-\beta_{s-1}\mathbf{q}_{s-1}-\alpha_s\mathbf{q}_s\in W$.
But [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection) also gives
$\mathbf{r}_s\perp\operatorname{range}(\mathbf{Q}_s)=W$.
Hence $\mathbf{r}_s=0$, and $\beta_s=0$. The identity $\mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{T}_s+\mathbf{r}_s\mathbf{e}_s^\top$ of [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection) then becomes $\mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{T}_s$.

**The eigenvalues of $\mathbf{T}_s$.** The normalized vectors
$\mathbf{u}_\lambda=\mathbf{p}_\lambda/\|\mathbf{p}_\lambda\|$ for 
$\lambda\in\Lambda_1$ form an orthonormal eigenbasis of $W$. 
Set $\mathbf{y}_\lambda=\mathbf{Q}_s^\top\mathbf{u}_\lambda$.
Transposing $\mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{T}_s$, with $\mathbf{A}$ and $\mathbf{T}_s$ symmetric, gives $\mathbf{Q}_s^\top\mathbf{A}=\mathbf{T}_s\mathbf{Q}_s^\top$, so
$
\mathbf{T}_s\mathbf{y}_\lambda
=\mathbf{T}_s\mathbf{Q}_s^\top\mathbf{u}_\lambda
=\mathbf{Q}_s^\top\mathbf{A}\mathbf{u}_\lambda
=\lambda\,\mathbf{Q}_s^\top\mathbf{u}_\lambda
=\lambda\,\mathbf{y}_\lambda$.
Moreover, $\mathbf{u}_\mu\in W=\operatorname{range}(\mathbf{Q}_s)$, so $\mathbf{Q}_s\mathbf{Q}_s^\top\mathbf{u}_\mu=\mathbf{u}_\mu$, and for $\lambda,\mu\in\Lambda_1$,
$\mathbf{y}_\lambda^\top\mathbf{y}_\mu
=\mathbf{u}_\lambda^\top\mathbf{Q}_s\mathbf{Q}_s^\top\mathbf{u}_\mu
=\mathbf{u}_\lambda^\top\mathbf{u}_\mu
=\delta_{\lambda,\mu}$.
Thus the $s$ vectors $\mathbf{y}_\lambda$, $\lambda\in\Lambda_1$, are orthonormal, hence an orthonormal basis of $\mathbb{R}^s$, and each of them is an eigenvector of $\mathbf{T}_s$. Every $\lambda\in\Lambda_1$ is therefore an eigenvalue of $\mathbf{T}_s$, with the nonzero eigenvector $\mathbf{y}_\lambda$. Conversely, let $\mathbf{T}_s\mathbf{z}=\xi\mathbf{z}$ with $\mathbf{z}\neq0$, and expand $\mathbf{z}=\sum_{\lambda\in\Lambda_1}c_\lambda\mathbf{y}_\lambda$ in this basis. Then
$0=\mathbf{T}_s\mathbf{z}-\xi\mathbf{z}=\sum_{\lambda\in\Lambda_1}c_\lambda(\lambda-\xi)\,\mathbf{y}_\lambda$,
so $c_\lambda(\lambda-\xi)=0 \ \forall \lambda\in\Lambda_1$, by linear independence. Some $c_\lambda$ is nonzero, since $\mathbf{z}\neq0$, so $\xi$ equals that $\lambda$. The elements of $\Lambda_1$ being distinct, $c_\mu=0 \ \forall \mu\neq\xi$, and $\mathbf{z}$ is a multiple of $\mathbf{y}_\xi$. Hence $\Lambda(\mathbf{T}_s)=\Lambda_1$, and each eigenvalue of $\mathbf{T}_s$ is simple, with eigenvectors the multiples of a single $\mathbf{y}_\lambda$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

### 4.10 Corollary 3 [Random initialization and repeated eigenvalues]

If $\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$ and $\mathbf{q}_1=\mathbf{g}/\|\mathbf{g}\|$, then, with probability one, a terminating Lanczos run recovers every distinct eigenvalue of $\mathbf{A}$. If $\mathbf{A}$ has $d$ distinct eigenvalues, the run reaches $s=d$ and yields a complete eigenbasis. If eigenvalues are repeated, a single run yields only one eigenvector from each corresponding eigenspace. With probability one, the eigenvector found in $E_\lambda$ at termination is the normalized projection of $\mathbf{q}_1$ onto $E_\lambda$.

#### Proof

For each $\lambda\in\Lambda(\mathbf{A})$, the event $\Pi_\lambda\mathbf{g}=0$ means that $\mathbf{g}$ lies in the proper linear subspace $E_\lambda^\perp$, which has Gaussian probability zero. Since there are finitely many distinct eigenvalues, the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="union-bound" aria-label="State the union bound">union bound</button><span id="union-bound" class="explanation-popover" popover="auto" role="note" aria-label="Union bound" data-label="Fact">For finitely many events, the probability that at least one of them occurs is at most the sum of their probabilities. A finite union of probability-zero events therefore has probability zero.</span></span> gives
$\mathbb{P}\bigl(\exists\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{g}=0\bigr)
\le\sum_{\lambda\in\Lambda(\mathbf{A})}\mathbb{P}(\Pi_\lambda\mathbf{g}=0)
=0$.
Thus $\Lambda_1=\Lambda(\mathbf{A})$ with probability one, and [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) applies.

For any polynomial $p$, $\Pi_\lambda p(\mathbf{A})\mathbf{q}_1=p(\lambda)\Pi_\lambda\mathbf{q}_1$. Therefore, the projection of the entire Krylov subspace onto $E_\lambda$ is contained in $\operatorname{span}\{\Pi_\lambda\mathbf{q}_1\}$, which is one-dimensional when $\Pi_\lambda\mathbf{q}_1\neq0$. A single run cannot recover additional independent directions in that eigenspace. At termination, the eigenvectors $\mathbf{y}_\lambda$ of $\mathbf{T}_s$ from the proof of [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) lift to the Ritz vectors $\mathbf{Q}_s\mathbf{y}_\lambda=\mathbf{u}_\lambda$, the normalized projections of $\mathbf{q}_1$.

When the spectrum is simple and every eigenvalue is present in the initialization, $s=d$. In this case, $\mathbf{Q}_d$ is an orthogonal matrix, and
$\mathbf{A}=\mathbf{Q}_d\mathbf{T}_d\mathbf{Q}_d^\top$.
An orthonormal eigenbasis of $\mathbf{T}_d$ therefore yields a complete orthonormal eigenbasis of $\mathbf{A}$ after multiplication by $\mathbf{Q}_d$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

An early exact breakdown does not necessarily mean that a complete eigenbasis of $\mathbf{A}$ has been found. It means that the current Krylov subspace has become invariant. For example, when $\mathbf{A}=c\,\mathbb{I}_d$, every initialization produces a breakdown after one iteration, even though the eigenspace has dimension $d$.
To recover all eigendirections, the construction can continue with new starting vectors orthogonal to the previously generated basis.

<figure class="article-figure article-figure-wide" id="figure-lanczos">
  <img src="/images/blog/matrix-vector-products/lanczos.svg" width="1400" height="450" alt="Three panels. Left: Ritz values at each Lanczos step spreading outward until, at step 12, they coincide with the 12 distinct eigenvalues of a 60 by 60 matrix. Middle: error of the largest Ritz value against the power-iteration estimate at equal numbers of products, Lanczos reaching rounding level at step 12. Right: the coefficient beta dropping by about eleven orders of magnitude at step 12." loading="lazy" />
  <figcaption><strong>Figure 6.</strong> The Lanczos method on a symmetric matrix $\mathbf A=\mathbf U\operatorname{diag}(\lambda_1,\dots,\lambda_{60})\mathbf U^\top$ of size $d=60$ with only $s=12$ distinct eigenvalues, of multiplicities 1 to 9 (labels on the right of panel a), including the eigenvalue 0 with multiplicity 8. The numbers $\lambda_1,\dots,\lambda_{60}$ are these eigenvalues, each repeated according to its multiplicity, and the eigenvectors, the columns of $\mathbf U$, are random: $\mathbf U$ is uniformly distributed over the orthogonal matrices, and is sampled as the orthogonal factor of the QR decomposition of a $60\times60$ matrix with independent standard Gaussian entries (with the signs chosen so that the triangular factor has a positive diagonal). (a) One run: Ritz values at each step $m$ (<span style="color:#27322f">black dots</span>) against the distinct eigenvalues (<span style="color:#7d8783">grey lines</span>). The extreme Ritz values (<span style="color:#16a34a">green</span>) move outward monotonically (<a href="#48-lemma-5-variational-characterization-and-monotonicity">Lemma 5</a>), the power-iteration estimate after the same number of products (<span style="color:#ea580c">orange</span>) stays between them, and at $m=s=12$ the Ritz values are exactly the 12 distinct eigenvalues, each appearing once (<a href="#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run">Proposition 3</a>). (b) Distance from $\lambda_{\max}(\mathbf A)=2$ of the largest Ritz value (<span style="color:#16a34a">green</span>) and of the power-iteration estimate (<span style="color:#ea580c">orange</span>), at equal numbers of matrix–vector products. (c) The coefficient $\beta_m$ falls from about $0.3$ at $m=11$ to about $10^{-12}$ at $m=12$: the exact breakdown of <a href="#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run">Proposition 3</a>, up to floating-point error. For every starting vector, the 12 Ritz values at $m=12$ match the 12 distinct eigenvalues to within $3\cdot10^{-15}$. Panels (b) and (c) show medians over 200 Gaussian starting vectors, with bands containing the middle half of the runs (the interquartile range), and all runs use full reorthogonalization.</figcaption>
</figure>

### 4.11 Proposition 4 [Complete decomposition by continuation with new starting vectors]

Let $\mathbf{A}\in\mathbb{R}^{d\times d}$ be symmetric. Initialize $\mathbf{q}_0=0$, $\beta_0=0$, and any unit vector $\mathbf{q}_1$. Run the Lanczos recurrence for $j=1,\dots,d$, with the following rule whenever $j<d$:

$$
\mathbf{q}_{j+1}=
\begin{cases}
\mathbf{r}_j/\beta_j,&\beta_j>0,\\
\text{any unit vector orthogonal to }\mathbf{q}_1,\dots,\mathbf{q}_j,&\beta_j=0 .
\end{cases}
$$

At a breakdown, retain $\beta_j=0$ and all previously generated basis vectors.

Then the following hold.

1. **Orthogonal basis.** The matrix $\mathbf{Q}_d=[\mathbf{q}_1,\dots,\mathbf{q}_d]$ is orthogonal, $\mathbf{Q}_d^\top\mathbf{Q}_d=\mathbb{I}_d$.
2. **Tridiagonal form.** $\beta_d=0$ and $\mathbf{A}\mathbf{Q}_d=\mathbf{Q}_d\mathbf{T}_d$, hence $\mathbf{A}=\mathbf{Q}_d\mathbf{T}_d\mathbf{Q}_d^\top$. The matrix $\mathbf{T}_d$ is symmetric and tridiagonal, with a zero off-diagonal entry at every boundary between runs.
3. **Complete eigenbasis.** If $\mathbf{T}_d=\mathbf{Z}\mathbf{D}\mathbf{Z}^\top$ with $\mathbf{Z}^\top\mathbf{Z}=\mathbb{I}_d$ and $\mathbf{D}\in\mathbb{R}^{d\times d}$ the diagonal matrix of the eigenvalues of $\mathbf{T}_d$, then $\mathbf{V}=\mathbf{Q}_d\mathbf{Z}$ is a complete orthogonal eigenvector matrix of $\mathbf{A}$:
$\mathbf{V}^\top\mathbf{V}=\mathbb{I}_d$,
and $\mathbf{A}\mathbf{V}=\mathbf{V}\mathbf{D}$.

#### Remark 16 [Why continuation completes the basis]

Each run discovers one direction in every eigenspace that its starting vector reaches, then stops at an invariant subspace ([Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run)). Continuing with a new starting vector in the orthogonal complement, which is invariant as well, lets the next run find the directions that the previous runs could not see.

#### Proof

At an exact breakdown after iteration $j$, $\mathbf{A}\mathbf{Q}_j=\mathbf{Q}_j\mathbf{T}_j$.
Thus $\operatorname{range}(\mathbf{Q}_j)$ is invariant under $\mathbf{A}$.
Its orthogonal complement is also invariant. Indeed, if $\mathbf{Q}_j^\top\mathbf{x}=0$, symmetry gives
$\mathbf{Q}_j^\top\mathbf{A}\mathbf{x}
=(\mathbf{A}\mathbf{Q}_j)^\top\mathbf{x}
=(\mathbf{Q}_j\mathbf{T}_j)^\top\mathbf{x}
=\mathbf{T}_j\mathbf{Q}_j^\top\mathbf{x}
=0$.
A new unit vector in this complement therefore starts another Lanczos run entirely within the complement. Since $\beta_j=0$, the first recurrence of that run contains no contribution from the preceding block.

Within each run, [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection) ensures orthogonality. Vectors from different runs are orthogonal because they lie in orthogonal invariant subspaces. Each run adds at least one vector, and the construction can continue whenever fewer than $d$ vectors have been obtained. Thus it produces $d$ orthonormal vectors.

At the final iteration, $\mathbf{r}_d$ is orthogonal to every column of $\mathbf{Q}_d$. These columns form a basis of $\mathbb{R}^d$, so
$\mathbf{r}_d=0$ and $\beta_d=0$.
Collecting all recurrences, including the zero couplings at block boundaries, gives
$\mathbf{A}\mathbf{Q}_d=\mathbf{Q}_d\mathbf{T}_d$.
Multiplying on the right by $\mathbf{Q}_d^\top$ gives $\mathbf{A}=\mathbf{Q}_d\mathbf{T}_d\mathbf{Q}_d^\top$.

Finally,
$\mathbf{A}(\mathbf{Q}_d\mathbf{Z})
=\mathbf{Q}_d\mathbf{T}_d\mathbf{Z}
=\mathbf{Q}_d\mathbf{Z}\mathbf{D}$
and
$(\mathbf{Q}_d\mathbf{Z})^\top(\mathbf{Q}_d\mathbf{Z})
=\mathbf{Z}^\top\mathbf{Q}_d^\top\mathbf{Q}_d\mathbf{Z}
=\mathbf{Z}^\top\mathbf{Z}
=\mathbb{I}_d$.
Therefore, $\mathbf{V}=\mathbf{Q}_d\mathbf{Z}$ is a complete orthonormal eigenbasis.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

A procedural choice of the new starting vector is

$$
\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d),
\qquad
\mathbf{u}=\mathbf{g}-\mathbf{Q}_j(\mathbf{Q}_j^\top\mathbf{g}),
\qquad
\mathbf{q}_{j+1}=\frac{\mathbf{u}}{\|\mathbf{u}\|}.
$$

When $j<d$, a fresh Gaussian draw gives $\mathbf{u}\neq0$ with probability one.
This continuation retains the entire existing basis. It is different from restarting schemes that discard or compress basis vectors to limit memory use.

### 4.12 Algorithm 2 [Lanczos with reorthogonalization and residual checks]

[Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) and [Proposition 4](#411-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors) recover eigenpairs exactly, but only at exact breakdowns and in exact arithmetic. The algorithm below is their finite version. It keeps the recurrence of [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown) and the Ritz pairs of [Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals), with three changes.

* **The basis is reorthogonalized.** In floating-point arithmetic, the short recurrence gradually loses orthogonality, as measured in [Section 4.3](#43-how-the-lanczos-method-works). Each new remainder is therefore projected once more against all the stored basis vectors, $\mathbf{r}_j\leftarrow\mathbf{r}_j-\mathbf{Q}_j(\mathbf{Q}_j^\top\mathbf{r}_j)$,
  before $\beta_j$ is computed. Such <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="full-reorthogonalization" aria-label="Show the source on full reorthogonalization">full reorthogonalization</button><span id="full-reorthogonalization" class="explanation-popover" popover="auto" role="note" aria-label="Full reorthogonalization" data-label="Source">See the discussion of full reorthogonalization in Bai et al., <a href="https://netlib.org/utk/people/JackDongarra/etemplates/node109.html"><em>Templates for the Solution of Algebraic Eigenvalue Problems</em></a>. The projection is carried out with classical or modified Gram–Schmidt, and it is repeated when the norm of the new vector drops below about $1/\sqrt2$ of its previous value, which rarely happens.</span></span> is a standard part of practical Lanczos implementations ([Bai et al., 2000](#ref-bai2000)).
* **Ritz pairs are accepted on a certificate.** Instead of waiting for an exact breakdown, the algorithm tests the Ritz pairs as the basis grows, and accepts a pair as soon as its residual with $\mathbf{A}$ is below a tolerance, which by [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) places its Rayleigh quotient near an eigenvalue of $\mathbf{A}$.
* **Breakdowns are detected with a tolerance.** A computed $\beta_j$ is almost never exactly zero. A step is treated as a breakdown when $\beta_j$ falls below a small threshold, and for a complete decomposition the basis is then continued from a new starting vector, as in [Proposition 4](#411-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors).

The algorithm uses three tolerances.
* The tolerances $\varepsilon_{\mathrm{abs}}>0$ and $\varepsilon_{\mathrm{rel}}\ge0$ play the same roles as in [Algorithm 1](#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation). Step 5 accepts a candidate $(\widehat\lambda,\mathbf{x})$ when $\|\mathbf{A}\mathbf{x}-\widehat\lambda\mathbf{x}\|\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{x}\|$, and by [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) the accepted $\widehat\lambda$ then lies within this threshold of an eigenvalue of $\mathbf{A}$. With $\varepsilon_{\mathrm{rel}}=0$, every eigenvalue is found to within the same distance $\varepsilon_{\mathrm{abs}}$, and a positive $\varepsilon_{\mathrm{rel}}$ asks instead for about the same number of correct digits on every eigenvalue, as explained for [Algorithm 1](#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation). One difference makes the relative rule safer here. Algorithm 2 does not deflate, so the residual of an accepted pair does not enter the test of any other pair, and the [stall of Algorithm 1](#tolerance-stall), where residuals inherited from earlier stages exceed the threshold, cannot occur. The floor remains: rounding errors keep computed residuals of the order of the unit roundoff times $\|\mathbf{A}\|$, so $\varepsilon_{\mathrm{abs}}$ must exceed that level. [Figure 7](#figure-lanczos-vs-deflation) uses the constant threshold $\varepsilon_{\mathrm{abs}}=10^{-10}\|\mathbf{A}\|$ and $\varepsilon_{\mathrm{rel}}=0$, the same as for Algorithm 1, so that the two algorithms are compared on equal terms.
* The breakdown tolerance $\tau>0$ decides when $\beta_j$ counts as zero in step 3. Even when the Krylov subspace is invariant, rounding errors leave a computed $\beta_j$ of the order of the unit roundoff times $\|\mathbf{A}\|$, so $\tau$ must exceed that level. The runs of [Figure 7](#figure-lanczos-vs-deflation) use $\tau=10^{-10}$ times the largest $|\alpha_i|$ or $\beta_i$ computed so far, an estimate of $\|\mathbf{A}\|$ that costs nothing.

Choose a maximum number of steps, and run the following steps.

1. **Fix the operator.** Provide a routine computing $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$, and keep everything that defines $\mathbf{A}$ fixed throughout the run. For a Hessian $\nabla_\theta^2f(\theta)$, these are $\theta$ and the function $f$, including the data it is evaluated on.
2. **Initialize the basis.** Choose a unit vector $\mathbf{q}_1$, for instance $\mathbf{q}_1=\mathbf{g}/\|\mathbf{g}\|$ with $\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$, which by [Corollary 3](#410-corollary-3-random-initialization-and-repeated-eigenvalues) has a component in every eigenspace of $\mathbf{A}$ with probability one. Set $\mathbf{q}_0=0$, $\beta_0=0$, $\mathbf{Q}_1=[\mathbf{q}_1]$ and $j=1$.
3. **Extend the basis (one product).** Compute $\mathbf{z}_j=\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}$, $\alpha_j=\mathbf{q}_j^\top\mathbf{z}_j$ and $\mathbf{r}_j=\mathbf{z}_j-\alpha_j\mathbf{q}_j$, as in [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown). Reorthogonalize, $\mathbf{r}_j\leftarrow\mathbf{r}_j-\mathbf{Q}_j(\mathbf{Q}_j^\top\mathbf{r}_j)$, repeating the projection once when needed, and set $\beta_j=\|\mathbf{r}_j\|$. If $\beta_j>\tau$, set $\mathbf{q}_{j+1}=\mathbf{r}_j/\beta_j$ and $\mathbf{Q}_{j+1}=[\mathbf{Q}_j,\mathbf{q}_{j+1}]$. Otherwise, the run has broken down, and $\beta_j$ is set to $0$.
4. **Solve the small eigenvalue problem.** At selected steps, assemble $\mathbf{T}_m$ with $m=j$, and compute its desired eigenpairs $\mathbf{T}_m\mathbf{y}_i=\vartheta_i\mathbf{y}_i$ with $\|\mathbf{y}_i\|=1$, for instance the $k$ of largest absolute value. Screen them with the inexpensive estimate $\beta_m|(\mathbf{y}_i)_m|$ of their residual ([Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals)), which costs no product: a pair whose estimate exceeds $\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}|\vartheta_i|$ is not yet a candidate.
5. **Check the candidates with the original operator (one product each).** For each remaining pair, form $\mathbf{x}_i=\mathbf{Q}_m\mathbf{y}_i/\|\mathbf{Q}_m\mathbf{y}_i\|$, compute $\widehat{\lambda}_i=\mathbf{x}_i^\top\mathbf{A}\mathbf{x}_i$ and $\varepsilon_i=\|\mathbf{A}\mathbf{x}_i-\widehat{\lambda}_i\mathbf{x}_i\|$, and accept $(\widehat\lambda_i,\mathbf{x}_i)$ when $\varepsilon_i\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{x}_i\|$.
6. **Stop or continue.** Stop when the requested pairs have been accepted, and report nonconvergence when the maximum number of steps is reached first. Otherwise, if the run has not broken down, increase $j$ by one and return to step 3. If it has broken down and a complete decomposition is requested, continue as in [Proposition 4](#411-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors): draw $\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$, set $\mathbf{q}_{j+1}=\mathbf{P}\mathbf{g}/\|\mathbf{P}\mathbf{g}\|$ with $\mathbf{P}\mathbf{g}=\mathbf{g}-\mathbf{Q}_j(\mathbf{Q}_j^\top\mathbf{g})$, keep $\beta_j=0$, set $\mathbf{Q}_{j+1}=[\mathbf{Q}_j,\mathbf{q}_{j+1}]$, increase $j$ by one, and return to step 3.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

#### How the steps implement the theory

The table lists what each step computes and the result it implements. The paragraphs after it justify the three points where the finite computation departs from the exact theory.

| Step | Computes | Implements |
| --- | --- | --- |
| 3. Extend | $\alpha_j$, $\beta_j$ and $\mathbf{q}_{j+1}$ | one step of the recurrence of [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown) |
| 3. Reorthogonalize | $\mathbf{r}_j\leftarrow\mathbf{r}_j-\mathbf{Q}_j(\mathbf{Q}_j^\top\mathbf{r}_j)$ | the orthogonality $\mathbf{Q}_j^\top\mathbf{r}_j=0$ of [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection), which rounding errors would otherwise erode |
| 3. Breakdown | $\beta_j\le\tau$ | the exact breakdown of [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown) and [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) |
| 4. Small problem | $\mathbf{T}_m\mathbf{y}_i=\vartheta_i\mathbf{y}_i$ and $\beta_m\lvert(\mathbf{y}_i)_m\rvert$ | the Ritz pairs of [Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals) and the norm of their residual |
| 5. Check | $\lVert\mathbf{A}\mathbf{x}_i-\widehat\lambda_i\mathbf{x}_i\rVert\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\lVert\mathbf{A}\mathbf{x}_i\rVert$ | the certificate of [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair), with the Rayleigh quotient of [Remark 11](#remark-11-any-scalar-and-the-best-one) |
| 6. Continue | a new unit vector orthogonal to $\mathbf{Q}_j$, with $\beta_j=0$ | the continuation of [Proposition 4](#411-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors) |

**Reorthogonalization restores what the theory assumes.** [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection), and through it [Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals) and Propositions [3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) and [4](#411-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors), assume that the basis vectors are exactly orthonormal. The short recurrence keeps them orthogonal only up to errors that accumulate from step to step, and the projection of step 3 removes these errors at every step. In the example of [Section 4.3](#43-how-the-lanczos-method-works), it keeps $\|\mathbf{Q}_m^\top\mathbf{Q}_m-\mathbb{I}_m\|$ at about $10^{-15}$ up to $m=200$, against $10^{-6}$ without it.

**The inexpensive estimate screens, the explicit residual certifies.** In exact arithmetic, $\beta_m|(\mathbf{y}_i)_m|$ equals the residual norm $\|\mathbf{A}\mathbf{x}_i-\vartheta_i\mathbf{x}_i\|$ ([Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals)). In floating-point arithmetic, the remaining loss of orthogonality and the corrections of the reorthogonalization perturb the identity $\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\mathbf{r}_m\mathbf{e}_m^\top$ on which this equality rests. Step 4 therefore uses the estimate only to decide which pairs are worth a product, and step 5 computes the residual with $\mathbf{A}$ itself, which [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) turns into a guarantee however $\mathbf{x}_i$ was obtained. The normalization by $\|\mathbf{Q}_m\mathbf{y}_i\|$ in step 5 matters for the same reason: when the computed basis is not exactly orthonormal, $\mathbf{Q}_m\mathbf{y}_i$ need not have unit norm, and the inexpensive estimate can then be too optimistic (see the implementation discussion of the [Lanczos algorithm](https://netlib.org/utk/people/JackDongarra/etemplates/node104.html) in [Bai et al., 2000](#ref-bai2000)).

**A small $\beta_j$ is an approximate breakdown.** By [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection), $\|\mathbf{A}\mathbf{Q}_j-\mathbf{Q}_j\mathbf{T}_j\|=\|\mathbf{r}_j\mathbf{e}_j^\top\|=\beta_j$, so $\beta_j\le\tau$ means that $\mathbf{A}$ maps $\operatorname{range}(\mathbf{Q}_j)$ into itself up to an error $\tau$. It is not an exact proof of invariance or spectral completeness. Likewise, satisfying a residual tolerance certifies an approximate eigenpair, not that no more extreme eigenvalue has been missed.

Each basis-extension step requires one matrix–vector product. If $C_{\mathbf{A}}$ denotes its cost, $m$ steps require $m$ such products, with total cost $mC_{\mathbf{A}}$, in addition to any products used for explicit residual checks. Retaining $\mathbf{Q}_m$ requires $O(dm)$ storage. Full reorthogonalization costs $O(dj)$ operations at iteration $j$, and therefore
$\sum_{j=1}^mO(dj)=O(dm^2)$
operations over the run, in addition to solving the projected eigenvalue problems.

When $\mathbf{A}$ is available only through products, as in the examples of [Section 2](#2-matrices-known-only-through-their-products), the run never forms $\mathbf{A}$. Nevertheless, a complete decomposition with $m=d$ still requires $O(d^2)$ storage for the basis, and full reorthogonalization then costs $O(d^3)$ operations. The practical benefit is greatest when the required eigenpairs reach the desired accuracy with $m\ll d$.

[Figure 7](#figure-lanczos-vs-deflation) below compares the two algorithms on the network Hessian of [Figure 5](#figure-hessian-deflation) and on the bipartite graph of [Figure 2](#figure-bipartite-graph).

<figure class="article-figure" id="figure-lanczos-vs-deflation">
  <img src="/images/blog/matrix-vector-products/lanczos-vs-deflation.svg" width="1040" height="430" alt="Two panels comparing Algorithm 2 (Lanczos) with Algorithm 1. Left: Hessian-vector products needed to certify the j largest eigenpairs of a network Hessian, from about 13 to 35 for Lanczos and from about 220 to 1100 for Algorithm 1. Right: eigenpairs certified against products for a 50-node bipartite graph; Lanczos with continuation reaches all 50 after 50 products, Algorithm 1 after about fifteen thousand." loading="lazy" />
  <figcaption><strong>Figure 7.</strong> <a href="#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks">Algorithm 2</a> (<span style="color:#16a34a">green</span>) against <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> (<span style="color:#2563eb">blue</span>), with the same constant threshold $10^{-10}\|\mathbf A\|$ on the residuals for $\mathbf A$. (a) The network Hessian of <a href="#figure-hessian-deflation">Figure 5</a>: Hessian–vector products used until the $j$ eigenpairs of largest absolute value are certified, medians over 20 starting vectors with bands containing the middle half of the runs. For <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a>, the count runs over stages 1 to $j$. For <a href="#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks">Algorithm 2</a>, it is the number of Lanczos steps until the $j$ Ritz pairs of largest absolute value pass the inexpensive test $\beta_m|(\mathbf y_i)_m|\le10^{-10}\|\mathbf A\|$ of <a href="#47-lemma-4-ritz-pairs-and-their-residuals">Lemma 4</a>, plus one product per pair for the explicit check of step 5, whose residuals were all below $5.6\cdot10^{-10}$. The eight pairs cost 35 products with <a href="#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks">Algorithm 2</a> and about 1,100 with <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a>. (b) The complete eigenbasis of the bipartite graph of <a href="#figure-bipartite-graph">Figure 2</a>. The first Lanczos run breaks down at step 41, the number of distinct eigenvalues (<a href="#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run">Proposition 3</a>), where its 41 Ritz pairs all pass the test. Each of the nine continuation runs of <a href="#411-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors">Proposition 4</a> starts in the eigenspace of $0$, orthogonally to the basis built so far, and breaks down after one step. The 50 products, plus 50 explicit checks (<span style="color:#16a34a">green square</span>), give all 50 eigenpairs with residuals below $1.2\cdot10^{-11}$, against 15,296 products for <a href="#38-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> with every stage taken to its limit, as in <a href="#figure-complete-eigenbasis">Figure 3</a>.</figcaption>
</figure>

## 5. The Arnoldi method

All the matrices of [Section 2](#2-matrices-known-only-through-their-products) are symmetric, and every proof so far uses the spectral theorem. Many matrices known through their products are not symmetric, and for them three facts of [Section 1](#1-notation-and-preliminaries) fail. First, the eigenvalues of a real matrix can be complex, and then they come in conjugate pairs $\lambda,\bar\lambda$. Second, eigenvectors of distinct eigenvalues need not be orthogonal. Third, the eigenvectors need not span the whole space: the matrix $\left[\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right]$ has the double eigenvalue $0$, but its only eigenvectors are the multiples of $[1,0]^\top$.

The Krylov subspace of [Definition 1](#41-definition-1-krylov-subspace) and the [Gram–Schmidt process](#42-the-gramschmidt-process) need no symmetry. The **Arnoldi method** ([Arnoldi, 1951](#ref-arnoldi1951)) runs Gram–Schmidt on $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ for any square matrix, as the Lanczos method does for a symmetric one, and [Saad (1980)](#ref-saad1980) developed it into a method for approximating a few eigenvalues of a large nonsymmetric matrix. This section follows the plan of [Section 4](#4-the-lanczos-method): examples first, then the method worked by hand, its exact properties, the certificate that replaces [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair), and an implementable procedure.

From here on, $\mathbf{A}\in\mathbb{R}^{d\times d}$ is any real square matrix, and $\Lambda(\mathbf{A})\subset\mathbb{C}$ is the set of its distinct eigenvalues. For a complex vector $\mathbf{y}$, $\mathbf{y}^*$ is its conjugate transpose and $\|\mathbf{y}\|=(\mathbf{y}^*\mathbf{y})^{1/2}$, which is the Euclidean norm when $\mathbf{y}$ is real.

### 5.1 Nonsymmetric matrices known through their products

* **Random walks on directed graphs.** On a directed graph with $d$ nodes, let $W_{ij}\ge0$ be the weight of the edge from node $i$ to node $j$, with $W_{ij}=0$ when there is no such edge. An edge from $i$ to $j$ need not come with an edge from $j$ to $i$, so $\mathbf{W}$ is not symmetric. A random walk steps from node $i$ to node $j$ with probability $P_{ij}=W_{ij}/\sum_kW_{ik}$, assuming that every node has an outgoing edge. The rows of the transition matrix $\mathbf{P}=\operatorname{diag}(\mathbf{W}\mathbf{1})^{-1}\mathbf{W}$ sum to one, $\mathbf{P}\mathbf{1}=\mathbf{1}$. If $(\mathbf{p}_k)_i$ is the probability that the walker is at node $i$ after $k$ steps, then $(\mathbf{p}_{k+1})_j=\sum_i(\mathbf{p}_k)_iP_{ij}$, that is, $\mathbf{p}_{k+1}=\mathbf{P}^\top\mathbf{p}_k$. A stationary distribution, which one step of the walk leaves unchanged, is therefore an eigenvector of $\mathbf{P}^\top$ with eigenvalue $1$, $\mathbf{P}^\top\boldsymbol{\pi}=\boldsymbol{\pi}$. It gives the long-run fraction of time that the walker spends at each node, and on the graph of hyperlinks between web pages, with random jumps added, it is the PageRank of [Brin and Page (1998)](#ref-brin1998). The other eigenvalues of $\mathbf{P}^\top$ say how fast $\mathbf{p}_k$ approaches $\boldsymbol{\pi}$, and complex ones make it oscillate on the way, as [Figure 8](#figure-directed-graph) below shows. A product with $\mathbf{P}^\top$ costs one pass over the edges, $(\mathbf{P}^\top\mathbf{x})_j=\sum_iP_{ij}x_i$ over the edges $i\to j$.
* **Convection–diffusion.** The operator $u\mapsto-u''+c\,u'$ on $[0,1]$, with $u(0)=u(1)=0$, describes a quantity that diffuses while a flow of speed $c$ carries it, such as heat or a pollutant in a moving fluid. When $c=0$, it is the operator $u\mapsto-u''$ discretized in [Section 2.2](#22-other-matrices-known-through-their-products), and we discretize it in the same way. The grid points are $t_i=ih$ for $i=0,\dots,d+1$, with step $h=1/(d+1)$. The unknown $x_i$ approximates $u(t_i)$ at each of the $d$ interior points, and the boundary conditions set $x_0=x_{d+1}=0$. Replacing the derivatives at $t_i$ by the centered differences
  $$
  u''(t_i)\approx\frac{x_{i+1}-2x_i+x_{i-1}}{h^2},
  \qquad
  u'(t_i)\approx\frac{x_{i+1}-x_{i-1}}{2h}
  $$
  gives the matrix defined by
  $$
  (\mathbf{A}\mathbf{x})_i=\frac{2x_i-(1+\gamma)\,x_{i-1}-(1-\gamma)\,x_{i+1}}{h^2},
  \qquad
  \gamma=\frac{ch}{2},
  $$
  for $i\in\{1,\dots,d\}$. When $c=0$, this is the symmetric matrix $(\mathbf{A}\mathbf{x})_i=(2x_i-x_{i-1}-x_{i+1})/h^2$ of that example. When $c\neq0$, it is not symmetric, because its entry $A_{i,i-1}=-(1+\gamma)/h^2$ below the diagonal differs from the entry $A_{i-1,i}=-(1-\gamma)/h^2$ above it. For $0<\gamma<1$, it is similar to a symmetric matrix. With $s=\sqrt{(1+\gamma)/(1-\gamma)}$ and $\mathbf{S}=\operatorname{diag}(1,s,s^2,\dots,s^{d-1})$, the matrix $\widetilde{\mathbf{A}}=\mathbf{S}^{-1}\mathbf{A}\mathbf{S}$ is <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="convection-symmetrization" aria-label="Show why the rescaled convection-diffusion matrix is symmetric">symmetric and tridiagonal</button><span id="convection-symmetrization" class="explanation-popover" popover="auto" role="note" aria-label="Symmetrizing the convection-diffusion matrix" data-label="Derivation">Since $\mathbf{S}$ is diagonal with $S_{ii}=s^{i-1}$, the entries of $\widetilde{\mathbf{A}}=\mathbf{S}^{-1}\mathbf{A}\mathbf{S}$ are $\widetilde A_{ik}=s^{k-i}A_{ik}$. The diagonal is unchanged, $\widetilde A_{ii}=A_{ii}=2/h^2$, and the entries beyond the neighboring diagonals stay zero. On the two neighboring diagonals, <span class="explanation-popover__equation">$\widetilde A_{i,i-1}=\frac{A_{i,i-1}}{s}=-\frac{1+\gamma}{h^2}\sqrt{\frac{1-\gamma}{1+\gamma}}=-\frac{\sqrt{1-\gamma^2}}{h^2},\qquad \widetilde A_{i-1,i}=s\,A_{i-1,i}=-\frac{1-\gamma}{h^2}\sqrt{\frac{1+\gamma}{1-\gamma}}=-\frac{\sqrt{1-\gamma^2}}{h^2}.$</span> The two are equal, so $\widetilde{\mathbf{A}}$ is symmetric. The rescaling makes the weight $1+\gamma$ of the upstream neighbor and the weight $1-\gamma$ of the downstream one meet at their geometric mean.</span></span>, with $2/h^2$ on its diagonal and $-\sqrt{1-\gamma^2}/h^2$ on both neighboring diagonals. The eigenvalues of $\mathbf{A}$ are therefore real, equal to $\bigl(2-2\sqrt{1-\gamma^2}\cos(k\pi/(d+1))\bigr)/h^2$ for $k=1,\dots,d$. The eigenvectors of $\mathbf{A}$, however, are those of $\widetilde{\mathbf{A}}$ with entry $i$ multiplied by $s^{i-1}$. They are all concentrated near the end of the grid toward which the flow goes, and they are far from orthogonal. A product costs $O(d)$ operations, as for $-u''$.
* **Jacobians of dynamical systems.** A system that evolves by $\dot{\mathbf{x}}=F(\mathbf{x})$, for a map $F:\mathbb{R}^d\to\mathbb{R}^d$, is at rest at a point $\mathbf{x}^\star$ where $F(\mathbf{x}^\star)=0$. A small perturbation $\mathbf{y}=\mathbf{x}-\mathbf{x}^\star$ then evolves approximately by $\dot{\mathbf{y}}=\mathbf{J}\mathbf{y}$, where $\mathbf{J}$ is the Jacobian of $F$ at $\mathbf{x}^\star$, with entries $J_{ij}=\partial F_i(\mathbf{x}^\star)/\partial x_j$, because $F(\mathbf{x}^\star+\mathbf{y})\approx\mathbf{J}\mathbf{y}$ to first order. If $\mathbf{J}$ has a basis of eigenvectors, $\mathbf{J}\mathbf{v}=\lambda\mathbf{v}$, the solution is a combination of the terms $e^{\lambda t}\mathbf{v}$, which decay when the real part of $\lambda$ is negative and grow when it is positive. The equilibrium is therefore stable when every eigenvalue has a negative real part, and the eigenvalues with the largest real part decide. The matrix $\mathbf{J}$ is not symmetric in general. A product with it is a directional derivative, $\mathbf{J}\mathbf{x}=\frac{d}{d\varepsilon}F(\mathbf{x}^\star+\varepsilon\mathbf{x})\big|_{\varepsilon=0}$, which forward-mode automatic differentiation computes at the cost of a few evaluations of $F$. In machine learning, simultaneous gradient steps on the two players of a game, such as the generator and the discriminator of a generative adversarial network (GAN), follow such a vector field, and eigenvalues of its Jacobian with large imaginary parts make the iterates circle around an equilibrium rather than approach it ([Mescheder et al., 2017](#ref-mescheder2017)). The simplest case is $\min_\theta\max_\phi\theta\phi$, where $\dot\theta=-\phi$ and $\dot\phi=\theta$, so that $\mathbf{J}=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$, whose eigenvalues $\pm i$ describe a pure rotation.

<figure class="article-figure article-figure-wide" id="figure-directed-graph">
  <img src="/images/blog/matrix-vector-products/directed-graph.svg" width="1200" height="860" alt="Four panels on a random walk. Top left: a directed graph with three groups of twenty nodes on the corners of a triangle, with most edges going from each group to the next, counterclockwise. Top right: its transition matrix, with three dense blocks above the block diagonal and in the bottom-left corner. Bottom left: the probability of each group over thirty steps, oscillating with period three and settling at one third. Bottom right: the eigenvalues in the complex plane, with the eigenvalue 1, a complex pair of modulus 0.81 at angles plus and minus 119 degrees, and the others within radius 0.45." loading="lazy" />
  <figcaption><strong>Figure 8.</strong> The random walk on a directed graph used in this section (<a href="#51-nonsymmetric-matrices-known-through-their-products">Section 5.1</a>). (a) The graph: 60 nodes in three groups of 20, <span style="color:#0d9488">teal</span>, <span style="color:#ca8a04">gold</span> and <span style="color:#db2777">pink</span>, arranged in the cycle 1 → 2 → 3 → 1 that the large arrows show. Every node has five outgoing edges: four to distinct nodes of the next group, and one to a node anywhere else, other than itself and these four, all drawn uniformly at random. The 259 edges that end in the next group are drawn in the color of the group they leave, and the other 41 in <span style="color:#27322f">black</span> (<span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>). (b) The transition matrix $\mathbf P$, with rows and columns ordered by group and $P_{ij}=1/5$ for every edge $i\to j$: the edges into the next group fill three colored blocks, and the other edges are the scattered <span style="color:#27322f">black</span> entries. (c) The probability of each group under the distribution $\mathbf p_k=(\mathbf P^\top)^k\mathbf p_0$ of the walker, started uniformly on group 1. It circulates through the groups with period three and settles at the stationary values 0.335, 0.331 and 0.334, close to $1/3$ (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>). Between steps 10 and 30, the largest gap to these values shrinks by a factor of 0.80 per step. (d) The eigenvalues of $\mathbf P$ in the complex plane, all inside the unit circle (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>). The eigenvalue 1 has the eigenvector $\mathbf 1$ for $\mathbf P$, and the stationary distribution for $\mathbf P^\top$. The pair $\lambda_2,\bar\lambda_2=0.81\,e^{\pm i\,119^\circ}$ produces the circulation of panel (c), a third of a turn per step damped by the factor 0.81. The other 57 eigenvalues have modulus below 0.45.</figcaption>
</figure>

The figures of this section use a random walk on a graph with three groups of 20 nodes arranged in a cycle, shown in [Figure 8](#figure-directed-graph) below. Every node has five outgoing edges, four to distinct nodes of the next group and one to another node anywhere, all drawn uniformly at random, and the walk follows one of them, chosen uniformly. The walk therefore tends to move from group 1 to group 2, then to group 3, and back to group 1. Started in group 1, its distribution circulates through the groups while it settles (panel c), and the eigenvalues explain how. If $\mathbf{P}^\top$ has a basis of eigenvectors, the distribution expands as

$$
\mathbf{p}_k=\boldsymbol{\pi}+c\,\lambda_2^k\,\mathbf{w}+\bar c\,\bar\lambda_2^k\,\bar{\mathbf{w}}+\cdots=\boldsymbol{\pi}+2\operatorname{Re}\bigl(c\,\lambda_2^k\,\mathbf{w}\bigr)+\cdots,
$$

where $\mathbf{P}^\top\mathbf{w}=\lambda_2\mathbf{w}$, the number $c$ depends on $\mathbf{p}_0$, and the dots stand for the terms of the other eigenvalues. The conjugate terms come from $\bar\lambda_2$, which is also an eigenvalue, with eigenvector $\bar{\mathbf{w}}$, because $\mathbf{P}$ is real. Here $\lambda_2=0.81\,e^{i\,119^\circ}$ (panel d), so $\lambda_2^k=0.81^k\,e^{i\,119^\circ k}$: the middle term turns by $119^\circ$, about a third of a turn, at each step, which moves the mass from one group to the next, and it shrinks by the factor $0.81$. The other 57 eigenvalues have modulus below $0.45$, and their terms die out faster.


**Power iteration without symmetry.** The iteration $\mathbf{v}_{k+1}=\mathbf{A}\mathbf{v}_k/\|\mathbf{A}\mathbf{v}_k\|$ of [Section 3](#3-power-iteration) needs no symmetry. For $\mathbf{A}=\mathbf{P}^\top$ started from a distribution, it is the walk itself, rescaled: $\mathbf{v}_k=\mathbf{p}_k/\|\mathbf{p}_k\|$. When a single real eigenvalue $\lambda_1$ has the largest modulus, as $1$ does here, the expansion above shows that the iterates approach its eigenvector, with an error that shrinks like $(|\lambda_2|/|\lambda_1|)^k$. Two features of [Section 3](#3-power-iteration) are lost. The Rayleigh quotient $\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k$ converges at that same rate, not at its square as in [Theorem 1](#32-theorem-1-rate-of-convergence), because the error of the iterate is no longer orthogonal to the eigenvector. And the next eigenvalues can form a conjugate pair of equal modulus, which defeats power iteration in the way the opposite eigenvalues of [Remark 5](#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude) do. Once the eigenvector of $1$ is deflated, the pair dominates, and the iterates keep turning in the real plane that carries $\lambda_2$ and $\bar\lambda_2$, without settling on a direction. Panel (b) of [Figure 9](#figure-arnoldi) below shows both effects, and the Arnoldi method removes them.

### 5.2 How the Arnoldi method works

> <span class="article-kicker article-kicker--theorem">The idea</span> The Arnoldi method runs the [Gram–Schmidt process](#42-the-gramschmidt-process) on the Krylov subspace 
> $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{m-1}\mathbf{q}_1\}=\bigl\{p(\mathbf{A})\mathbf{q}_1:p\text{ is a polynomial of degree at most }m-1\bigr\}$, one matrix–vector product per step, exactly as the Lanczos method does, but for a matrix that need not be symmetric. 
> Without symmetry, no coefficient vanishes for free: each step removes the components of the new vector along all the previous basis vectors. The coefficients fill a small upper Hessenberg matrix, and its eigenvalues, real or complex, approximate eigenvalues of $\mathbf{A}$.

After $m-1$ products, the Krylov subspace contains every vector $p(\mathbf{A})\mathbf{q}_1$ with $p$ of degree at most $m-1$, whatever $\mathbf{A}$ is. To search it, we again need an orthonormal basis $\mathbf{q}_1,\dots,\mathbf{q}_m$, the columns of $\mathbf{Q}_m$, and the coordinates of $\mathbf{A}\mathbf{q}_j$ in that basis, which form the matrix $\mathbf{H}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m$ with entries $(\mathbf{H}_m)_{ij}=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$. The search itself changes. Without symmetry, the eigenvalues are no longer the extreme values of the Rayleigh quotient, so we look instead for a vector $\mathbf{x}=\mathbf{Q}_m\mathbf{y}$ of the subspace and a number $\vartheta$ whose residual $\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}$ is orthogonal to the subspace. An exact eigenpair satisfies this condition, since its residual is zero. Multiplying the residual by $\mathbf{Q}_m^\top$ turns this condition into $\mathbf{H}_m\mathbf{y}=\vartheta\mathbf{y}$, an $m\times m$ eigenvalue problem. Gram–Schmidt on $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$ computes the basis and the matrix $\mathbf{H}_m$ together, and its first steps, compared with those of [Section 4.3](#43-how-the-lanczos-method-works), show what symmetry was providing. The colors are those of [Section 4.3](#43-how-the-lanczos-method-works): a <span style="color:#7c3aed">purple</span> coefficient is computed by an inner product at the current step, a <span style="color:#16a34a">green</span> one is the length of the new direction, and a <span style="color:#7d8783">grey</span> one vanishes.

**Step 0: Gram–Schmidt on $\mathcal{K}_1(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1\}$.** There is nothing to do, as in Step 0 of [Section 4.3](#43-how-the-lanczos-method-works): the unit vector $\mathbf{q}_1$ is an orthonormal basis of $\mathcal{K}_1(\mathbf{A},\mathbf{q}_1)$.

**Step 1: Gram–Schmidt on $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$.** The first product gives $\mathbf{A}\mathbf{q}_1$, and Gram–Schmidt removes its component $\textcolor{#7c3aed}{h_{11}}=\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_1$ along $\mathbf{q}_1$ and divides the remainder $\mathbf{r}_1=\mathbf{A}\mathbf{q}_1-h_{11}\mathbf{q}_1$ by its length $\textcolor{#16a34a}{h_{21}}=\|\mathbf{r}_1\|$. This is Step 1 of [Section 4.3](#43-how-the-lanczos-method-works), where these numbers were called $\alpha_1$ and $\beta_1$:

$$
\mathbf{A}\mathbf{q}_1=\textcolor{#7c3aed}{h_{11}\mathbf{q}_1}+\textcolor{#16a34a}{h_{21}\mathbf{q}_2},
\qquad
\mathbf{q}_2=\frac{\mathbf{r}_1}{h_{21}}=\frac{\mathbf{A}-h_{11}\mathbb{I}_d}{h_{21}}\,\mathbf{q}_1 .
$$

**Step 2: Gram–Schmidt on $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$.** The second product gives $\mathbf{A}\mathbf{q}_2$, which brings the same new direction as $\mathbf{A}^2\mathbf{q}_1$, by the argument of Step 2 of [Section 4.3](#43-how-the-lanczos-method-works), which used no symmetry. Its component along $\mathbf{q}_1$ is where the two methods part. For a symmetric matrix, this component was $(\mathbf{A}\mathbf{q}_1)^\top\mathbf{q}_2=\beta_1$, the length found in Step 1. In general,

$$
\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_2=(\mathbf{A}^\top\mathbf{q}_1)^\top\mathbf{q}_2,
$$

and nothing relates $\mathbf{A}^\top\mathbf{q}_1$ to the Krylov subspace, so this component is a new number $\textcolor{#7c3aed}{h_{12}}$, computed by an inner product. Removing it and the component $\textcolor{#7c3aed}{h_{22}}=\mathbf{q}_2^\top\mathbf{A}\mathbf{q}_2$ along $\mathbf{q}_2$, then normalizing, gives

$$
\mathbf{A}\mathbf{q}_2=\textcolor{#7c3aed}{h_{12}\mathbf{q}_1}+\textcolor{#7c3aed}{h_{22}\mathbf{q}_2}+\textcolor{#16a34a}{h_{32}\mathbf{q}_3}.
$$

The coefficients found so far give $\mathbf{H}_2=\mathbf{Q}_2^\top\mathbf{A}\mathbf{Q}_2=\left[\begin{smallmatrix}h_{11}&h_{12}\\h_{21}&h_{22}\end{smallmatrix}\right]$, which has the shape of $\mathbf{T}_2$ in [Section 4.3](#43-how-the-lanczos-method-works), except that its entry $h_{12}$ above the diagonal is a new number, not the length $h_{21}$ found in Step 1.

**Step 3: Gram–Schmidt on $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$.** The third product gives $\mathbf{A}\mathbf{q}_3$. In [Section 4.3](#43-how-the-lanczos-method-works), its component along $\mathbf{q}_1$ vanished because $\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_3=(\mathbf{A}\mathbf{q}_1)^\top\mathbf{q}_3=0$. Here this component is $(\mathbf{A}^\top\mathbf{q}_1)^\top\mathbf{q}_3$, nonzero in general, and all three components have to be computed and removed:

$$
\mathbf{A}\mathbf{q}_3=\textcolor{#7c3aed}{h_{13}\mathbf{q}_1}+\textcolor{#7c3aed}{h_{23}\mathbf{q}_2}+\textcolor{#7c3aed}{h_{33}\mathbf{q}_3}+\textcolor{#16a34a}{h_{43}\mathbf{q}_4}.
$$

Some coefficients still vanish. The vector $\mathbf{A}\mathbf{q}_1$ lies in $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\mathbf{q}_2\}$, so it has no component along $\mathbf{q}_3$, and $\mathbf{A}\mathbf{q}_2$, which lies in $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$, has none along $\mathbf{q}_4$. These zeros use only the nesting of the Krylov subspaces, not symmetry. Writing the coefficients of $\mathbf{A}\mathbf{q}_j$ in column $j$, as the identity $(\mathbf{H}_3)_{ij}=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$ requires, and doing the same for the Lanczos coefficients, gives

$$
\mathbf{H}_3=\begin{bmatrix}
\textcolor{#7c3aed}{h_{11}}&\textcolor{#7c3aed}{h_{12}}&\textcolor{#7c3aed}{h_{13}}\\
\textcolor{#16a34a}{h_{21}}&\textcolor{#7c3aed}{h_{22}}&\textcolor{#7c3aed}{h_{23}}\\
\textcolor{#7d8783}{0}&\textcolor{#16a34a}{h_{32}}&\textcolor{#7c3aed}{h_{33}}
\end{bmatrix},
\qquad
\mathbf{T}_3=\begin{bmatrix}
\textcolor{#7c3aed}{\alpha_1}&\textcolor{#ea580c}{\beta_1}&\textcolor{#7d8783}{0}\\
\textcolor{#16a34a}{\beta_1}&\textcolor{#7c3aed}{\alpha_2}&\textcolor{#ea580c}{\beta_2}\\
\textcolor{#7d8783}{0}&\textcolor{#16a34a}{\beta_2}&\textcolor{#7c3aed}{\alpha_3}
\end{bmatrix}.
$$

In $\mathbf{T}_3$, symmetry made each entry above the diagonal either a length already found (<span style="color:#ea580c">orange</span>) or zero (<span style="color:#7d8783">grey</span>). In $\mathbf{H}_3$, these entries are new numbers (<span style="color:#7c3aed">purple</span>). The zeros below the first subdiagonal remain in both matrices, because they come from the nesting of Krylov subspaces. A matrix whose entries below the first subdiagonal are zero is called **upper Hessenberg**.

**The general step.** The same reasoning holds at every step $j\in\{1,2,3,\dots\}$, as long as the previous steps have produced $\mathbf{q}_j$, which for $j\ge2$ means that $h_{j,j-1}>0$. The product $\mathbf{A}\mathbf{q}_j$ lies in $\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{j+1}\}$, because $\mathbf{q}_j\in\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, so it has no component beyond $\mathbf{q}_{j+1}$. Its components $\textcolor{#7c3aed}{h_{ij}}=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$ along all of $\mathbf{q}_1,\dots,\mathbf{q}_j$, on the other hand, have to be computed and removed. What remains is orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_j$, and when its length $\textcolor{#16a34a}{h_{j+1,j}}$ is positive, dividing by it gives the next basis vector:

$$
\mathbf{r}_j=\mathbf{A}\mathbf{q}_j\textcolor{#7c3aed}{-\sum_{i=1}^jh_{ij}\mathbf{q}_i}=\textcolor{#16a34a}{h_{j+1,j}\mathbf{q}_{j+1}},
\qquad
\textcolor{#16a34a}{h_{j+1,j}}=\|\mathbf{r}_j\| .
$$

Step $j$ of [Definition 3](#53-definition-3-arnoldi-recurrence-and-exact-breakdown) below is therefore the following Gram–Schmidt step.

$$
\mathbf{q}_j
\xrightarrow{\ \mathbf{A}\ }
\mathbf{A}\mathbf{q}_j
\xrightarrow{\ \textcolor{#7c3aed}{-h_{1j}\mathbf{q}_1-\cdots-h_{jj}\mathbf{q}_j}\ }
\mathbf{r}_j
\xrightarrow{\ \textcolor{#16a34a}{\div h_{j+1,j}}\ }
\textcolor{#16a34a}{\mathbf{q}_{j+1}}.
$$

Step $j$ costs one product, $j$ inner products and $j$ vector updates, that is, $O(dj)$ operations besides the product, and every basis vector has to be kept. After $m$ steps, the method has used $m$ products, $O(dm^2)$ further operations and $O(dm)$ storage. This is the price of nonsymmetry. It equals the cost of the Lanczos method with full reorthogonalization ([Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks)), which subtracts all the components anyway, to protect orthogonality in floating-point arithmetic.

**Eigenvalues and eigenvectors from the coefficients.** Writing the relations $\mathbf{A}\mathbf{q}_j=\sum_{i=1}^{j+1}h_{ij}\mathbf{q}_i$ for $j=1,\dots,m$ as the columns of <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="arnoldi-matrix-identity" aria-label="Explain why the relations combine into this matrix identity">one matrix identity</button><span id="arnoldi-matrix-identity" class="explanation-popover" popover="auto" role="note" aria-label="Checking the Arnoldi matrix identity column by column" data-label="Derivation">Compare the columns. Column $j$ of $\mathbf{A}\mathbf{Q}_m$ is $\mathbf{A}\mathbf{q}_j$. Column $j$ of $\mathbf{Q}_m\mathbf{H}_m$ is $\mathbf{Q}_m$ times column $j$ of $\mathbf{H}_m$, whose nonzero entries are $h_{1j},\dots,h_{j+1,j}$, in rows $1$ to $j+1$. For $j<m$, it is therefore $\sum_{i=1}^{j+1}h_{ij}\mathbf{q}_i=\mathbf{A}\mathbf{q}_j$. The last column is the exception: $\mathbf{H}_m$ has no row $m+1$, so column $m$ of $\mathbf{Q}_m\mathbf{H}_m$ stops at $\sum_{i=1}^{m}h_{im}\mathbf{q}_i$ and misses the term $h_{m+1,m}\mathbf{q}_{m+1}$. The <span style="color:#16a34a">green matrix</span> $h_{m+1,m}\mathbf{q}_{m+1}\mathbf{e}_m^\top$ adds it back: its columns are zero except the last, which is $h_{m+1,m}\mathbf{q}_{m+1}$.</span></span> gives

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{H}_m+\textcolor{#16a34a}{h_{m+1,m}\mathbf{q}_{m+1}\mathbf{e}_m^\top},
$$

the analogue of the Lanczos identity, with the same leak through the last basis vector. Every eigenpair $(\vartheta,\mathbf{y})$ of $\mathbf{H}_m$, with $\mathbf{y}\in\mathbb{C}^m$ and $\|\mathbf{y}\|=1$, gives a Ritz pair $(\vartheta,\mathbf{Q}_m\mathbf{y})$, as in [Section 4.3](#43-how-the-lanczos-method-works). Multiplying the identity by $\mathbf{y}$ gives

$$
\mathbf{A}\mathbf{Q}_m\mathbf{y}
=\mathbf{Q}_m\mathbf{H}_m\mathbf{y}+h_{m+1,m}\mathbf{q}_{m+1}(\mathbf{e}_m^\top\mathbf{y})
=\vartheta\,\mathbf{Q}_m\mathbf{y}+\textcolor{#16a34a}{h_{m+1,m}\,y_m\,\mathbf{q}_{m+1}},
$$

so the residual of the Ritz pair is again the leak, of norm $\textcolor{#16a34a}{h_{m+1,m}|y_m|}$. The matrix $\mathbf{H}_m$ is real but, in general, not symmetric, so its eigenvalues can be complex, in conjugate pairs, which is how the method finds the complex eigenvalues of $\mathbf{A}$.

The polynomial reading of [Section 4.3](#43-how-the-lanczos-method-works) carries over, with one change. Its first two steps used no symmetry: the green term of $\mathbf{A}^m\mathbf{q}_1$ is now $h_{21}h_{32}\cdots h_{m+1,m}\,\mathbf{q}_{m+1}$, it has the smallest norm among the vectors $\pi(\mathbf{A})\mathbf{q}_1$ with $\pi$ monic of degree $m$, and the Ritz values at step $m$ are the roots of the minimizing polynomial $\pi_m$ ([Proposition 5](#55-proposition-5-ritz-pairs-the-minimizing-polynomial-and-exact-termination)). What changes is how $\|\pi(\mathbf{A})\mathbf{q}_1\|$ depends on the eigenvalues. Suppose that $\mathbf{A}=\mathbf{V}\mathbf{D}\mathbf{V}^{-1}$ as in [Lemma 7](#56-lemma-7-residual-certificate-without-symmetry) below, with the eigenvalues $\lambda_1,\dots,\lambda_d$ on the diagonal of $\mathbf{D}$, and write $\mathbf{q}_1=\mathbf{V}\mathbf{c}$. Then $\pi(\mathbf{A})\mathbf{q}_1=\mathbf{V}\pi(\mathbf{D})\mathbf{c}$, and since $\|\mathbf{z}\|=\|\mathbf{V}^{-1}\mathbf{V}\mathbf{z}\|\le\|\mathbf{V}^{-1}\|\,\|\mathbf{V}\mathbf{z}\|$ for every $\mathbf{z}$,

$$
\frac{\|\pi(\mathbf{D})\mathbf{c}\|}{\|\mathbf{V}^{-1}\|}
\le\|\pi(\mathbf{A})\mathbf{q}_1\|
\le\|\mathbf{V}\|\,\|\pi(\mathbf{D})\mathbf{c}\|,
\qquad
\|\pi(\mathbf{D})\mathbf{c}\|^2=\sum_{i=1}^d|\pi(\lambda_i)|^2\,|c_i|^2 .
$$

For a symmetric matrix, $\mathbf{V}$ can be chosen orthogonal, both factors equal $1$, and $\|\pi(\mathbf{A})\mathbf{q}_1\|$ is exactly the weighted sum over the eigenvalues of [Section 4.3](#43-how-the-lanczos-method-works). Otherwise, the two bounds differ by the factor $\kappa(\mathbf{V})=\|\mathbf{V}\|\,\|\mathbf{V}^{-1}\|$, which is large when the eigenvectors are nearly parallel. The terms $\pi(\lambda_i)c_i\mathbf{v}_i$ of $\pi(\mathbf{A})\mathbf{q}_1=\sum_{i=1}^d\pi(\lambda_i)c_i\mathbf{v}_i$, where $\mathbf{v}_i$ is column $i$ of $\mathbf{V}$, can then cancel, and a polynomial can make $\|\pi(\mathbf{A})\mathbf{q}_1\|$ small without being small at the eigenvalues that $\mathbf{q}_1$ contains. The Ritz values can then stay far from every eigenvalue until $m$ is large, and in panel (c) of [Figure 9](#figure-arnoldi) below they leave the real axis although every eigenvalue is real.

### 5.3 Definition 3 [Arnoldi recurrence and exact breakdown]

Start with a unit vector $\mathbf{q}_1$. At iteration $j\ge1$, compute

$$
\begin{aligned}
h_{ij}&=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j\qquad(1\le i\le j),\\
\mathbf{r}_j&=\mathbf{A}\mathbf{q}_j-\sum_{i=1}^jh_{ij}\mathbf{q}_i,\\
h_{j+1,j}&=\|\mathbf{r}_j\| .
\end{aligned}
$$

If $h_{j+1,j}>0$, define $\mathbf{q}_{j+1}=\mathbf{r}_j/h_{j+1,j}$ and continue. If $h_{j+1,j}=0$, stop the run, an **exact breakdown** as in [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown). For $m\ge1$, $\mathbf{H}_m\in\mathbb{R}^{m\times m}$ is the matrix with entries $(\mathbf{H}_m)_{ij}=h_{ij}$ for $i\le j+1$ and $(\mathbf{H}_m)_{ij}=0$ for $i>j+1$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

The coefficients can equally be computed one at a time: starting from $\mathbf{w}=\mathbf{A}\mathbf{q}_j$, compute $h_{ij}=\mathbf{q}_i^\top\mathbf{w}$ and update $\mathbf{w}\leftarrow\mathbf{w}-h_{ij}\mathbf{q}_i$ for $i=1,\dots,j$. This is the modified Gram–Schmidt process. Each update leaves the inner products of $\mathbf{w}$ with the other basis vectors unchanged, since the basis vectors are orthogonal, so in exact arithmetic both orders give the same numbers. In floating-point arithmetic, the one-at-a-time order loses less orthogonality, and implementations repeat the subtraction when needed, as in [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks).

### 5.4 Lemma 6 [Orthogonality and Hessenberg projection]

Before and including the first exact breakdown, the Arnoldi vectors satisfy

$$
\mathbf{q}_i^\top\mathbf{q}_j=\delta_{ij},
\qquad
\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_m\}=\mathcal{K}_m(\mathbf{A},\mathbf{q}_1).
$$

With $\mathbf{Q}_m=[\mathbf{q}_1,\dots,\mathbf{q}_m]$ and the upper Hessenberg matrix $\mathbf{H}_m$ of [Definition 3](#53-definition-3-arnoldi-recurrence-and-exact-breakdown),

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{H}_m+\mathbf{r}_m\mathbf{e}_m^\top,
\qquad
\mathbf{Q}_m^\top\mathbf{r}_m=0,
\qquad
\mathbf{H}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m .
$$

If $\mathbf{A}$ is symmetric, then $\mathbf{H}_m$ is the symmetric tridiagonal matrix $\mathbf{T}_m$ of [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection), and the Arnoldi vectors are the Lanczos vectors.

#### Remark 17 [Where the zeros of the projected matrix come from]

The zeros of $\mathbf{T}_m$ have two sources. Those below the first subdiagonal come from the nesting of Krylov subspaces: $\mathbf{A}\mathbf{q}_j$ lies in $\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$, so it has no component along $\mathbf{q}_{j+2},\mathbf{q}_{j+3},\dots$. Those above the first superdiagonal come from symmetry, which turns the component $\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$ into $(\mathbf{A}\mathbf{q}_i)^\top\mathbf{q}_j$. The Arnoldi method keeps the first source and loses the second. This is why $\mathbf{H}_m$ is upper Hessenberg, and why each step removes every previous component.

#### Proof

**Orthogonality.** We argue by induction, as for [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection). The initial vector is normalized by assumption. Suppose that $\mathbf{q}_1,\dots,\mathbf{q}_j$ are orthonormal. For each $i\le j$, by the definition of $h_{ij}$ and since $\mathbf{q}_i^\top\mathbf{q}_l=\delta_{il}$,

$$
\mathbf{q}_i^\top\mathbf{r}_j=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j-\sum_{l=1}^jh_{lj}\,\mathbf{q}_i^\top\mathbf{q}_l=h_{ij}-h_{ij}=0 .
$$

Thus $\mathbf{r}_j$ is orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_j$, and if $h_{j+1,j}>0$, normalizing it extends the orthonormal family. Unlike the proof of [Lemma 3](#46-lemma-3-orthogonality-and-tridiagonal-projection), this one needs no symmetry, because every component is removed explicitly.

**Krylov subspaces.** We prove $\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_m\}=\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ by induction on $m$. It holds at $m=1$, since both sides equal $\operatorname{span}\{\mathbf{q}_1\}$. Suppose that it holds at $m=j$ and that $h_{j+1,j}>0$. The vector $\mathbf{q}_{j+1}$ is a combination of $\mathbf{A}\mathbf{q}_j$ and $\mathbf{q}_1,\dots,\mathbf{q}_j$, all of which lie in $\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$, since $\mathbf{q}_j\in\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$ and $\mathbf{A}$ maps $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$ into $\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$. Hence $\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{j+1}\}\subseteq\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$. The left-hand side has dimension $j+1$, because its spanning vectors are orthonormal, and the right-hand side is spanned by $j+1$ vectors, so the two spaces are equal.

**Matrix identity.** The definition of $\mathbf{r}_j$ gives $\mathbf{A}\mathbf{q}_j=\sum_{i=1}^jh_{ij}\mathbf{q}_i+\mathbf{r}_j$, where $\mathbf{r}_j=h_{j+1,j}\mathbf{q}_{j+1}$ for $j<m$. These identities, for $j=1,\dots,m$, are the columns of $\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{H}_m+\mathbf{r}_m\mathbf{e}_m^\top$. Multiplying on the left by $\mathbf{Q}_m^\top$, and using $\mathbf{Q}_m^\top\mathbf{Q}_m=\mathbb{I}_m$ and $\mathbf{Q}_m^\top\mathbf{r}_m=0$ from the orthogonality part, gives $\mathbf{H}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m$.

**The symmetric case.** If $\mathbf{A}$ is symmetric, then so is $\mathbf{H}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m$. Its entries below the first subdiagonal vanish, so by symmetry its entries above the first superdiagonal vanish too, and $h_{j-1,j}=h_{j,j-1}$. Step $j$ therefore subtracts only $h_{j,j-1}\mathbf{q}_{j-1}$ and $h_{jj}\mathbf{q}_j$ from $\mathbf{A}\mathbf{q}_j$, which is the Lanczos step of [Definition 2](#45-definition-2-lanczos-recurrence-and-exact-breakdown) with $\alpha_j=h_{jj}$ and $\beta_{j-1}=h_{j,j-1}$. By induction on $j$, both methods produce the same vectors and the same coefficients, and $\mathbf{H}_m=\mathbf{T}_m$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

### 5.5 Proposition 5 [Ritz pairs, the minimizing polynomial, and exact termination]

Suppose that the run has reached iteration $m$, that is, $h_{j+1,j}>0$ for $1\le j<m$.

1. **Ritz pairs.** Let $\mathbf{H}_m\mathbf{y}=\vartheta\mathbf{y}$ with $\mathbf{y}\in\mathbb{C}^m$ and $\|\mathbf{y}\|=1$, and let $\mathbf{x}=\mathbf{Q}_m\mathbf{y}$. Then
   $$
   \|\mathbf{x}\|=1,
   \qquad
   \mathbf{x}^*\mathbf{A}\mathbf{x}=\vartheta,
   \qquad
   \mathbf{A}\mathbf{x}-\vartheta\mathbf{x}=\mathbf{r}_m\,y_m,
   \qquad
   \|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|=h_{m+1,m}|y_m| .
   $$
2. **The minimizing polynomial.** The characteristic polynomial $\pi_m(t)=\det(t\,\mathbb{I}_m-\mathbf{H}_m)$ is the unique monic polynomial of degree $m$ that minimizes $\|\pi(\mathbf{A})\mathbf{q}_1\|$, and
   $$
   \pi_m(\mathbf{A})\mathbf{q}_1=h_{21}h_{32}\cdots h_{m,m-1}\,\mathbf{r}_m .
   $$
   The Ritz values at step $m$ are the roots of $\pi_m$.
3. **Exact termination.** Let $\mu$ be the monic polynomial of smallest degree $s$ such that $\mu(\mathbf{A})\mathbf{q}_1=0$, called the minimal polynomial of $\mathbf{q}_1$. A run reaches iteration $s$ and breaks down there: $h_{j+1,j}>0$ for $j<s$, and $h_{s+1,s}=0$. Moreover, at termination,
   $$
   \mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{H}_s,
   \qquad
   \det(t\,\mathbb{I}_s-\mathbf{H}_s)=\mu(t),
   $$
   so the Ritz values are exactly the roots of $\mu$, and every one of them is an eigenvalue of $\mathbf{A}$. If $\mathbf{A}$ has a basis of eigenvectors and $\mathbf{q}_1=\sum_{\lambda\in\Lambda(\mathbf{A})}\mathbf{q}_\lambda$ with $\mathbf{A}\mathbf{q}_\lambda=\lambda\mathbf{q}_\lambda$, then
   $$
   \mu(t)=\prod_{\lambda\in\Lambda_1}(t-\lambda),
   \qquad
   \Lambda_1=\{\lambda\in\Lambda(\mathbf{A}):\mathbf{q}_\lambda\neq0\},
   \qquad
   s=|\Lambda_1|,
   $$
   as in [Proposition 3](#49-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run).

#### Remark 18 [Ritz values lie in the field of values]

Item 1 says that every Ritz value is the value of $\mathbf{x}^*\mathbf{A}\mathbf{x}$ at a unit vector. It therefore lies in the **field of values**

$$
W(\mathbf{A})=\bigl\{\mathbf{x}^*\mathbf{A}\mathbf{x}:\mathbf{x}\in\mathbb{C}^d,\ \|\mathbf{x}\|=1\bigr\},
$$

which also contains every eigenvalue, since $\mathbf{x}^*\mathbf{A}\mathbf{x}=\lambda$ for a unit eigenvector $\mathbf{x}$. For a symmetric matrix, $W(\mathbf{A})$ is the interval $[\lambda_{\min}(\mathbf{A}),\lambda_{\max}(\mathbf{A})]$, and the Ritz values stay between the extreme eigenvalues ([Lemma 5](#48-lemma-5-variational-characterization-and-monotonicity)). For a nonsymmetric matrix, $W(\mathbf{A})$ is a convex region of the complex plane that can be much larger than the smallest convex set containing the eigenvalues, and the Ritz values can lie anywhere in it. In panel (c) of [Figure 9](#figure-arnoldi) below, every eigenvalue is real, but the field of values is an oval around them, and the Ritz values spread over it.

#### Proof

**Ritz pairs.** As in the proof of [Lemma 4](#47-lemma-4-ritz-pairs-and-their-residuals), the orthonormality of the columns of $\mathbf{Q}_m$ gives $\|\mathbf{x}\|^2=\mathbf{y}^*\mathbf{Q}_m^\top\mathbf{Q}_m\mathbf{y}=\|\mathbf{y}\|^2=1$, and [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection) gives $\mathbf{x}^*\mathbf{A}\mathbf{x}=\mathbf{y}^*\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m\mathbf{y}=\mathbf{y}^*\mathbf{H}_m\mathbf{y}=\vartheta$ and

$$
\begin{aligned}
\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}
&=\mathbf{Q}_m\mathbf{H}_m\mathbf{y}+\mathbf{r}_m\mathbf{e}_m^\top\mathbf{y}-\vartheta\mathbf{Q}_m\mathbf{y}\\
&=\mathbf{Q}_m\textcolor{#2563eb}{(\mathbf{H}_m\mathbf{y}-\vartheta\mathbf{y})}+\mathbf{r}_my_m\\
&=\mathbf{r}_my_m ,
\end{aligned}
$$

where the <span style="color:#2563eb">blue factor</span> vanishes because $\mathbf{y}$ is an eigenvector of $\mathbf{H}_m$. Taking norms gives $\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|=\|\mathbf{r}_m\|\,|y_m|=h_{m+1,m}|y_m|$.

**The minimizing polynomial.** Let $\mathbf{e}_1=[1,0,\dots,0]^\top$ be the first standard basis vector of $\mathbb{R}^m$, so that $\mathbf{q}_1=\mathbf{Q}_m\mathbf{e}_1$. We first show that

$$
\mathbf{A}^k\mathbf{q}_1=\mathbf{Q}_m\mathbf{H}_m^k\mathbf{e}_1\quad\text{for }0\le k\le m-1,
\qquad
\mathbf{A}^m\mathbf{q}_1=\mathbf{Q}_m\mathbf{H}_m^m\mathbf{e}_1+h_{21}h_{32}\cdots h_{m,m-1}\,\mathbf{r}_m .
$$

Because $\mathbf{H}_m$ is upper Hessenberg, $h_{il}=0$ for $i>l+1$, so $(\mathbf{H}_m\mathbf{v})_i=\sum_{l\ge i-1}h_{il}v_l$ for every vector $\mathbf{v}\in\mathbb{C}^m$. If the last nonzero entry of $\mathbf{v}$ is in position $p$, this sum is zero for $i\ge p+2$, and for $i=p+1$ it reduces to $(\mathbf{H}_m\mathbf{v})_{p+1}=h_{p+1,p}\,v_p$: multiplying by $\mathbf{H}_m$ moves the last nonzero entry down by one position and multiplies it by a subdiagonal entry. Starting from $\mathbf{e}_1$, by induction on $k$, the vector $\mathbf{H}_m^k\mathbf{e}_1$ has nonzero entries only in its first $k+1$ positions, and its entry in position $k+1$ is the product $h_{21}h_{32}\cdots h_{k+1,k}$ of the subdiagonal entries met on the way. If $\mathbf{A}^k\mathbf{q}_1=\mathbf{Q}_m\mathbf{H}_m^k\mathbf{e}_1$ for some $k\le m-1$, then the identity of [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection) gives

$$
\mathbf{A}^{k+1}\mathbf{q}_1=\mathbf{A}\mathbf{Q}_m\mathbf{H}_m^k\mathbf{e}_1=\mathbf{Q}_m\mathbf{H}_m^{k+1}\mathbf{e}_1+\mathbf{r}_m\,\bigl(\mathbf{e}_m^\top\mathbf{H}_m^k\mathbf{e}_1\bigr).
$$

The last factor is the entry of $\mathbf{H}_m^k\mathbf{e}_1$ in position $m$. It vanishes for $k\le m-2$ and equals $h_{21}h_{32}\cdots h_{m,m-1}$ for $k=m-1$, which proves both identities by induction from $k=0$.

Now let $\pi(t)=t^m+\sum_{k=0}^{m-1}c_kt^k$ be a monic polynomial of degree $m$. Applying the two identities to each power gives

$$
\begin{aligned}
\pi(\mathbf{A})\mathbf{q}_1
&=\mathbf{A}^m\mathbf{q}_1+\sum_{k=0}^{m-1}c_k\mathbf{A}^k\mathbf{q}_1
=\mathbf{Q}_m\mathbf{H}_m^m\mathbf{e}_1+h_{21}h_{32}\cdots h_{m,m-1}\,\mathbf{r}_m+\sum_{k=0}^{m-1}c_k\mathbf{Q}_m\mathbf{H}_m^k\mathbf{e}_1\\
&=\mathbf{Q}_m\,\pi(\mathbf{H}_m)\,\mathbf{e}_1+h_{21}h_{32}\cdots h_{m,m-1}\,\mathbf{r}_m .
\end{aligned}
$$

For $\pi=\pi_m$, the Cayley–Hamilton theorem gives $\pi_m(\mathbf{H}_m)=0$, hence $\pi_m(\mathbf{A})\mathbf{q}_1=h_{21}\cdots h_{m,m-1}\,\mathbf{r}_m$, which is orthogonal to $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ by [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection). For any monic $\pi$ of degree $m$, the difference $\pi-\pi_m$ has degree at most $m-1$, so $(\pi-\pi_m)(\mathbf{A})\mathbf{q}_1$ lies in $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, and Pythagoras' theorem gives

$$
\|\pi(\mathbf{A})\mathbf{q}_1\|^2=\|\pi_m(\mathbf{A})\mathbf{q}_1\|^2+\|(\pi-\pi_m)(\mathbf{A})\mathbf{q}_1\|^2\ge\|\pi_m(\mathbf{A})\mathbf{q}_1\|^2 .
$$

Equality forces $(\pi-\pi_m)(\mathbf{A})\mathbf{q}_1=0$. The vectors $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}^{m-1}\mathbf{q}_1$ span the space $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, of dimension $m$ by [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection), so they are linearly independent, and the coefficients of $\pi-\pi_m$ all vanish: $\pi=\pi_m$. Finally, the Ritz values are the eigenvalues of $\mathbf{H}_m$, which are the roots of its characteristic polynomial.

**Exact termination.** For $j<s$, no monic polynomial of degree $j$ annihilates $\mathbf{q}_1$, by the minimality of $s$. If the run has reached iteration $j<s$, item 2 at step $j$ therefore gives

$$
h_{21}h_{32}\cdots h_{j,j-1}\,h_{j+1,j}=\|\pi_j(\mathbf{A})\mathbf{q}_1\|>0,
$$

hence $h_{j+1,j}>0$, and the run reaches iteration $j+1$. Starting from $j=1$, this shows that the run reaches iteration $s$. There, the monic polynomial $\mu$ of degree $s$ gives $\|\mu(\mathbf{A})\mathbf{q}_1\|=0$, so the minimum in item 2 is zero. By uniqueness, $\pi_s=\mu$, and $h_{21}\cdots h_{s,s-1}\,h_{s+1,s}=\|\mu(\mathbf{A})\mathbf{q}_1\|=0$ forces $h_{s+1,s}=0$. With $\mathbf{r}_s=0$, the identity of [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection) becomes $\mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{H}_s$. If $\mathbf{H}_s\mathbf{y}=\vartheta\mathbf{y}$ with $\mathbf{y}\neq0$, then $\mathbf{A}(\mathbf{Q}_s\mathbf{y})=\vartheta\,\mathbf{Q}_s\mathbf{y}$ with $\mathbf{Q}_s\mathbf{y}\neq0$, so every Ritz value at termination is an eigenvalue of $\mathbf{A}$.

**The case of a basis of eigenvectors.** If $\mathbf{q}_1=\sum_\lambda\mathbf{q}_\lambda$ with $\mathbf{A}\mathbf{q}_\lambda=\lambda\mathbf{q}_\lambda$, then $p(\mathbf{A})\mathbf{q}_1=\sum_{\lambda\in\Lambda_1}p(\lambda)\,\mathbf{q}_\lambda$ for every polynomial $p$. Nonzero eigenvectors of distinct eigenvalues are linearly independent, so $p(\mathbf{A})\mathbf{q}_1=0$ exactly when $p(\lambda)=0$ for every $\lambda\in\Lambda_1$. The monic polynomial of smallest degree with this property is $\prod_{\lambda\in\Lambda_1}(t-\lambda)$, of degree $|\Lambda_1|$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

Item 2 also proves, for the Lanczos method, the statement of [Section 4.3](#43-how-the-lanczos-method-works) that the Ritz values are the roots of the minimizing polynomial, since $\mathbf{H}_m=\mathbf{T}_m$ for a symmetric matrix ([Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection)). Without a basis of eigenvectors, $\mu$ can have repeated roots, and $\mathbf{H}_s$ then has them as repeated eigenvalues.

### 5.6 Lemma 7 [Residual certificate without symmetry]

Suppose that $\mathbf{A}=\mathbf{V}\mathbf{D}\mathbf{V}^{-1}$ for an invertible matrix $\mathbf{V}\in\mathbb{C}^{d\times d}$, whose columns are eigenvectors, and a diagonal matrix $\mathbf{D}$, which holds the eigenvalues. Let

$$
\kappa(\mathbf{V})=\|\mathbf{V}\|\,\|\mathbf{V}^{-1}\|,
\qquad\text{where }\|\mathbf{M}\|=\max_{\|\mathbf{z}\|=1}\|\mathbf{M}\mathbf{z}\| .
$$

Then, for every unit vector $\mathbf{x}\in\mathbb{C}^d$ and every $z\in\mathbb{C}$,

$$
\min_{\lambda\in\Lambda(\mathbf{A})}|z-\lambda|\le\kappa(\mathbf{V})\,\|\mathbf{A}\mathbf{x}-z\mathbf{x}\| .
$$

This is the theorem of [Bauer and Fike (1960)](#ref-bauer1960), written for a residual.

#### Remark 19 [A small residual far from the spectrum]

When $\mathbf{A}$ is symmetric, $\mathbf{V}$ can be chosen orthogonal, so $\kappa(\mathbf{V})=1$ and the lemma is [Remark 11](#remark-11-any-scalar-and-the-best-one). Otherwise $\kappa(\mathbf{V})$ can be large, and it cannot be dropped. Take

$$
\mathbf{A}=\begin{bmatrix}0&1\\\delta&0\end{bmatrix},
\qquad
\mathbf{x}=\begin{bmatrix}1\\0\end{bmatrix},
$$

with a small $\delta>0$. The eigenvalues are $\pm\sqrt\delta$, the Rayleigh quotient is $\mathbf{x}^\top\mathbf{A}\mathbf{x}=0$, and the residual $\mathbf{A}\mathbf{x}-0\cdot\mathbf{x}=[0,\delta]^\top$ has norm $\delta$, whereas the distance from $0$ to the spectrum is $\sqrt\delta$. For $\delta=10^{-6}$, the residual is $10^{-6}$ and the distance $10^{-3}$, a thousand times larger. The eigenvectors $[1,\pm\sqrt\delta]^\top$ are nearly parallel, $\kappa(\mathbf{V})=1/\sqrt\delta=1000$, and the lemma holds with equality. What a small residual $\mathbf{r}=\mathbf{A}\mathbf{x}-z\mathbf{x}$ always certifies is that $(z,\mathbf{x})$ is an exact eigenpair of a nearby matrix: $(\mathbf{A}-\mathbf{r}\mathbf{x}^*)\mathbf{x}=\mathbf{A}\mathbf{x}-\mathbf{r}=z\mathbf{x}$, and the perturbation has norm $\|\mathbf{r}\mathbf{x}^*\|=\|\mathbf{r}\|$. In the example, $\mathbf{A}-\mathbf{r}\mathbf{x}^\top=\left[\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right]$, whose eigenvalue $0$ is double. A perturbation of size $\delta$ has moved the eigenvalues by $\sqrt\delta$, and the eigenvalues of a nonsymmetric matrix can in general move much farther than the size of a perturbation. The set of numbers $z$ with $\|\mathbf{A}\mathbf{x}-z\mathbf{x}\|<\varepsilon$ for some unit vector $\mathbf{x}$ is called the $\varepsilon$-pseudospectrum of $\mathbf{A}$, and for a nonsymmetric matrix it can extend far beyond the points within $\varepsilon$ of an eigenvalue ([Trefethen and Embree, 2005](#ref-trefethen2005)).

#### Proof

If $z$ is an eigenvalue, the left-hand side is zero. Otherwise, the diagonal matrix $\mathbf{D}-z\mathbb{I}_d$ is invertible, and $\mathbf{A}-z\mathbb{I}_d=\mathbf{V}(\mathbf{D}-z\mathbb{I}_d)\mathbf{V}^{-1}$ gives

$$
\mathbf{x}=\mathbf{V}(\mathbf{D}-z\mathbb{I}_d)^{-1}\mathbf{V}^{-1}(\mathbf{A}\mathbf{x}-z\mathbf{x}) .
$$

The largest diagonal entry of $(\mathbf{D}-z\mathbb{I}_d)^{-1}$ in absolute value is $1/\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda-z|$, and it is the norm of this diagonal matrix. Taking norms therefore gives

$$
1=\|\mathbf{x}\|\le\frac{\|\mathbf{V}\|\,\|\mathbf{V}^{-1}\|}{\min_{\lambda\in\Lambda(\mathbf{A})}|\lambda-z|}\,\|\mathbf{A}\mathbf{x}-z\mathbf{x}\|,
$$

which is the claim.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

For the convection–diffusion matrix of [Section 5.1](#51-nonsymmetric-matrices-known-through-their-products), $\mathbf{A}=\mathbf{S}\widetilde{\mathbf{A}}\mathbf{S}^{-1}$ and $\widetilde{\mathbf{A}}=\mathbf{U}\mathbf{D}\mathbf{U}^\top$ with $\mathbf{U}$ orthogonal, so $\mathbf{V}=\mathbf{S}\mathbf{U}$ diagonalizes $\mathbf{A}$, with $\kappa(\mathbf{V})=\|\mathbf{S}\|\,\|\mathbf{S}^{-1}\|=s^{d-1}$. Panel (d) of [Figure 9](#figure-arnoldi) tests [Lemma 2](#37-lemma-2-residual-certificate-for-an-approximate-eigenpair) and [Lemma 7](#56-lemma-7-residual-certificate-without-symmetry) on its Ritz pairs.

<figure class="article-figure article-figure-wide" id="figure-arnoldi">
  <img src="/images/blog/matrix-vector-products/arnoldi.svg" width="1200" height="860" alt="Four panels on the Arnoldi method. Top left: Ritz values of a random walk at steps 5, 10 and 20 in the complex plane, converging to the eigenvalue 1 and to a complex pair, inside the field of values. Top right: distance to the eigenvalue against matrix-vector products, where power iteration decreases slowly and the Arnoldi Ritz values reach rounding level by 40 products. Bottom left: Ritz values of a convection-diffusion matrix with real eigenvalues, spread off the real axis inside an oval field of values. Bottom right: a log-log scatter of distance against residual, blue points of the symmetric matrix all below the diagonal, orange points of the nonsymmetric one partly above it." loading="lazy" />
  <figcaption><strong>Figure 9.</strong> The Arnoldi method on the two nonsymmetric matrices of <a href="#51-nonsymmetric-matrices-known-through-their-products">Section 5.1</a>. (a) The random walk of <a href="#figure-directed-graph">Figure 8</a>, through products with $\mathbf P^\top$: its eigenvalues (<span style="color:#7d8783">grey</span> dots), its field of values (<span style="color:#7d8783">grey</span> line, <a href="#remark-18-ritz-values-lie-in-the-field-of-values">Remark 18</a>), the unit circle (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>), and the Ritz values of one run, from a Gaussian starting vector, at $m=5$, 10 and 20, colored from <span style="background:linear-gradient(90deg,#b0dd2f,#481d6f);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span>. At $m=5$, the Ritz values $0.75$ and $-0.23\pm0.51i$ already point to the eigenvalue 1 and to the pair $\lambda_2,\bar\lambda_2$. They are within $6\cdot10^{-4}$ and $3\cdot10^{-4}$ of them at $m=10$, and within $1.1\cdot10^{-8}$ and $5.7\cdot10^{-7}$ at $m=20$, while the other Ritz values lie among the remaining eigenvalues. (b) Distance to an eigenvalue against the number of products, as medians over 200 Gaussian starting vectors, with bands containing the middle half of the runs. The Rayleigh quotient of power iteration on $\mathbf P^\top$ (<span style="color:#ea580c">orange</span>) approaches 1 at the rate $|\lambda_2|=0.81$ per product (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dotted;text-decoration-thickness:0.12em;text-underline-offset:0.2em">dotted</span>), with a median error of $1.2\cdot10^{-3}$ after 20 products and $1.2\cdot10^{-5}$ after 40, and power iteration never approaches $\lambda_2$. After 20 products, the Ritz value nearest to 1 (<span style="color:#16a34a">green</span>) is within $4.4\cdot10^{-9}$ of it, and the one nearest to $\lambda_2$ (<span style="color:#16a34a">green</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>) within $4.4\cdot10^{-7}$. Both reach rounding level, about $10^{-15}$, by 40 products. (c) The convection–diffusion matrix with $d=40$ and $\gamma=0.2$, multiplied by $h^2$. Its eigenvalues are real, between 0.046 and 3.954 (<span style="color:#27322f">black</span> ticks), and its field of values (<span style="color:#7d8783">grey</span>) is an oval that reaches $\pm0.40\,i$. The Ritz values of one run, from a Gaussian starting vector, at $m=10$, 20 and 30 (<span style="background:linear-gradient(90deg,#b0dd2f,#481d6f);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span>) are all real at $m=10$, but 14 of them at $m=20$ and 24 at $m=30$ are not, with imaginary parts up to $0.32$. (d) Distance from each Ritz value to the spectrum against its residual $\|\mathbf A\mathbf x-\vartheta\mathbf x\|$, for all the Ritz pairs of steps $m=1,\dots,30$ from 5 Gaussian starting vectors, 2,325 pairs per matrix. For the symmetric matrix of $-u''$ ($\gamma=0$, <span style="color:#2563eb">blue</span>), every point lies below the diagonal, as <a href="#37-lemma-2-residual-certificate-for-an-approximate-eigenpair">Lemma 2</a> requires, and the largest ratio of distance to residual is 0.49. For convection–diffusion (<span style="color:#ea580c">orange</span>), 311 points lie above the diagonal, with distances up to 5.2 times the residual. <a href="#56-lemma-7-residual-certificate-without-symmetry">Lemma 7</a> holds, with $\kappa(\mathbf V)=s^{d-1}=2.7\cdot10^3$, but it overestimates every distance by a factor of at least 520. The inexpensive residual $h_{m+1,m}|y_m|$ of <a href="#55-proposition-5-ritz-pairs-the-minimizing-polynomial-and-exact-termination">Proposition 5</a> agrees with the explicit one to a relative $2\cdot10^{-14}$.</figcaption>
</figure>

### 5.7 Algorithm 3 [Arnoldi with reorthogonalization and residual checks]

[Proposition 5](#55-proposition-5-ritz-pairs-the-minimizing-polynomial-and-exact-termination) recovers eigenvalues exactly, but only at an exact breakdown and in exact arithmetic. The algorithm below is its finite version. It follows [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks), with three differences.

* **Every component is removed.** Step 3 subtracts the components along all the stored basis vectors, as [Definition 3](#53-definition-3-arnoldi-recurrence-and-exact-breakdown) requires, and repeating this subtraction once is the reorthogonalization. A step therefore costs the same as a step of [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks).
* **The small eigenvalue problem is not symmetric.** The matrix $\mathbf{H}_m$ needs a dense eigensolver for nonsymmetric matrices, and its eigenpairs can be complex.
* **The residual certifies less.** By [Remark 19](#remark-19-a-small-residual-far-from-the-spectrum), an accepted pair is an exact eigenpair of a matrix close to $\mathbf{A}$, but by [Lemma 7](#56-lemma-7-residual-certificate-without-symmetry), its distance to the spectrum of $\mathbf{A}$ can exceed the residual by the factor $\kappa(\mathbf{V})$, which the run does not compute.

The tolerances $\varepsilon_{\mathrm{abs}}>0$, $\varepsilon_{\mathrm{rel}}\ge0$ and $\tau>0$ play the same roles as in [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks). Step 5 accepts a candidate $(\widehat\lambda,\mathbf{x})$ when its residual $\mathbf{r}=\mathbf{A}\mathbf{x}-\widehat\lambda\mathbf{x}$ satisfies $\|\mathbf{r}\|\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{x}\|$, and step 3 treats $h_{j+1,j}\le\tau$ as a breakdown. Only the meaning of an accepted pair changes: $(\widehat\lambda,\mathbf{x})$ is then an exact eigenpair of $\mathbf{A}-\mathbf{r}\mathbf{x}^*$, a matrix within $\|\mathbf{r}\|$ of $\mathbf{A}$, and when $\mathbf{A}=\mathbf{V}\mathbf{D}\mathbf{V}^{-1}$, its $\widehat\lambda$ lies within $\kappa(\mathbf{V})\,\|\mathbf{r}\|$ of an eigenvalue of $\mathbf{A}$.

Choose a maximum number of steps, and run the following steps.

1. **Fix the operator.** Provide a routine computing $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$, and keep everything that defines $\mathbf{A}$ fixed throughout the run.
2. **Initialize the basis.** Choose a unit vector $\mathbf{q}_1$, for instance $\mathbf{q}_1=\mathbf{g}/\|\mathbf{g}\|$ with $\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$. Set $\mathbf{Q}_1=[\mathbf{q}_1]$ and $j=1$.
3. **Extend the basis (one product).** Compute $\mathbf{w}=\mathbf{A}\mathbf{q}_j$, the coefficients $\mathbf{h}=\mathbf{Q}_j^\top\mathbf{w}$, whose entries are $h_{1j},\dots,h_{jj}$, and $\mathbf{w}\leftarrow\mathbf{w}-\mathbf{Q}_j\mathbf{h}$. Repeat the projection once, with $\mathbf{c}=\mathbf{Q}_j^\top\mathbf{w}$, $\mathbf{w}\leftarrow\mathbf{w}-\mathbf{Q}_j\mathbf{c}$ and $\mathbf{h}\leftarrow\mathbf{h}+\mathbf{c}$, and store $\mathbf{h}$ in column $j$ of $\mathbf{H}$. Set $h_{j+1,j}=\|\mathbf{w}\|$. If $h_{j+1,j}>\tau$, set $\mathbf{q}_{j+1}=\mathbf{w}/h_{j+1,j}$ and $\mathbf{Q}_{j+1}=[\mathbf{Q}_j,\mathbf{q}_{j+1}]$. Otherwise, the run has broken down, and $h_{j+1,j}$ is set to $0$.
4. **Solve the small eigenvalue problem.** At selected steps, with $m=j$, compute the eigenpairs $\mathbf{H}_m\mathbf{y}_i=\vartheta_i\mathbf{y}_i$, with $\mathbf{y}_i\in\mathbb{C}^m$ and $\|\mathbf{y}_i\|=1$, by a dense eigensolver for nonsymmetric matrices, and select the wanted ones, for instance those of largest modulus or of largest real part. Nonreal Ritz values come in conjugate pairs, and one member of each pair suffices: since $\mathbf{H}_m$ is real, $\mathbf{H}_m\bar{\mathbf{y}}_i=\bar\vartheta_i\bar{\mathbf{y}}_i$, and the conjugate pair has the same residual norm. Screen the wanted pairs with the inexpensive estimate $h_{m+1,m}|(\mathbf{y}_i)_m|$ of their residual ([Proposition 5](#55-proposition-5-ritz-pairs-the-minimizing-polynomial-and-exact-termination)), which costs no product: a pair whose estimate exceeds $\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}|\vartheta_i|$ is not yet a candidate.
5. **Check the candidates with the original operator.** For each remaining pair, form $\mathbf{x}_i=\mathbf{Q}_m\mathbf{y}_i/\|\mathbf{Q}_m\mathbf{y}_i\|$, compute $\widehat\lambda_i=\mathbf{x}_i^*\mathbf{A}\mathbf{x}_i$ and $\varepsilon_i=\|\mathbf{A}\mathbf{x}_i-\widehat\lambda_i\mathbf{x}_i\|$, and accept $(\widehat\lambda_i,\mathbf{x}_i)$ when $\varepsilon_i\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{x}_i\|$. This costs one product for a real $\mathbf{x}_i$, and two for a complex one, since $\mathbf{A}\mathbf{x}_i=\mathbf{A}\operatorname{Re}(\mathbf{x}_i)+i\,\mathbf{A}\operatorname{Im}(\mathbf{x}_i)$.
6. **Stop, continue or restart.** Stop when the wanted pairs have been accepted, and report nonconvergence when the maximum number of steps is reached first. Otherwise, if the run has not broken down, increase $j$ by one and return to step 3. If it has broken down, $\operatorname{range}(\mathbf{Q}_m)$ is invariant up to $\tau$, and the other eigenvalues require a new run from a new starting vector. When storing $\mathbf{Q}_m$ or orthogonalizing against it becomes too expensive, restart as described below.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

#### How the steps implement the theory

| Step | Computes | Implements |
| --- | --- | --- |
| 3. Extend | $h_{1j},\dots,h_{j+1,j}$ and $\mathbf{q}_{j+1}$ | one step of the recurrence of [Definition 3](#53-definition-3-arnoldi-recurrence-and-exact-breakdown) |
| 3. Repeat the projection | the correction $\mathbf{c}$, added to column $j$ of $\mathbf{H}$ | the orthogonality $\mathbf{Q}_j^\top\mathbf{r}_j=0$ of [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection), which rounding errors would otherwise erode |
| 3. Breakdown | $h_{j+1,j}\le\tau$ | the exact breakdown of [Definition 3](#53-definition-3-arnoldi-recurrence-and-exact-breakdown) and the exact termination of [Proposition 5](#55-proposition-5-ritz-pairs-the-minimizing-polynomial-and-exact-termination) |
| 4. Small problem | $\mathbf{H}_m\mathbf{y}_i=\vartheta_i\mathbf{y}_i$ and $h_{m+1,m}\lvert(\mathbf{y}_i)_m\rvert$ | the Ritz pairs of [Proposition 5](#55-proposition-5-ritz-pairs-the-minimizing-polynomial-and-exact-termination) and the norm of their residual |
| 5. Check | $\lVert\mathbf{A}\mathbf{x}_i-\widehat\lambda_i\mathbf{x}_i\rVert\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\lVert\mathbf{A}\mathbf{x}_i\rVert$ | the certificate of [Remark 19](#remark-19-a-small-residual-far-from-the-spectrum) and [Lemma 7](#56-lemma-7-residual-certificate-without-symmetry) |
| 6. Restart | a new starting vector built from the wanted Ritz vectors | a polynomial filter, in the sense of [Section 4.1](#41-definition-1-krylov-subspace), applied to the starting vector |

**The repeated projection.** In exact arithmetic, the second projection of step 3 finds $\mathbf{c}=0$, since $\mathbf{w}$ is already orthogonal to $\mathbf{Q}_j$, and $\mathbf{h}$ is unchanged. In floating-point arithmetic, the first projection leaves in $\mathbf{w}$ components along the stored vectors of the order of the unit roundoff $u$ times $\|\mathbf{A}\mathbf{q}_j\|$. Relative to $\|\mathbf{w}\|=h_{j+1,j}$, this is of the order of $u/f$ with $f=h_{j+1,j}/\|\mathbf{A}\mathbf{q}_j\|$, by the cancellation explained in [Section 4.3](#43-how-the-lanczos-method-works), and the second projection removes these components. Adding $\mathbf{c}$ to $\mathbf{h}$ keeps the identity $\mathbf{A}\mathbf{q}_j=\mathbf{Q}_j\mathbf{h}+h_{j+1,j}\mathbf{q}_{j+1}$ exact up to rounding, so that $\mathbf{H}_m$ is still the matrix of [Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection).

**The residual certifies a nearby matrix.** An accepted pair $(\widehat\lambda_i,\mathbf{x}_i)$ is an exact eigenpair of a matrix within $\varepsilon_i$ of $\mathbf{A}$ ([Remark 19](#remark-19-a-small-residual-far-from-the-spectrum)). Whether $\widehat\lambda_i$ is within $\varepsilon_i$ of an eigenvalue of $\mathbf{A}$ depends on the sensitivity of the eigenvalues, such as the factor $\kappa(\mathbf{V})$ of [Lemma 7](#56-lemma-7-residual-certificate-without-symmetry), which the run does not compute. When the eigenvalue itself must be accurate, for instance when the sign of a real part decides the stability of an equilibrium, the residual should be read together with an estimate of that sensitivity, as discussed by [Saad (2011)](#ref-saad2011) and by [Trefethen and Embree (2005)](#ref-trefethen2005).

**Restarts bound the cost.** Since step $j$ costs $O(dj)$ operations and every basis vector is stored, $m$ cannot grow indefinitely. A restart runs the method again from a new starting vector that combines the wanted Ritz vectors, so that the next Krylov subspace concentrates on them. The implicitly restarted Arnoldi method of [Sorensen (1992)](#ref-sorensen1992) restarts without any new product. It applies to $\mathbf{H}_m$ a few shifted QR steps, with the unwanted Ritz values as shifts, which amounts to multiplying the starting vector by the polynomial $\prod(t-\vartheta)$ over those values, a polynomial filter in the sense of [Section 4.1](#41-definition-1-krylov-subspace) that damps the unwanted eigencomponents. It then keeps a basis of $k<m$ vectors and extends it again. This is the method of the ARPACK software ([Lehoucq, Sorensen and Yang, 1998](#ref-lehoucq1998)), on which the sparse eigenvalue routines of several scientific computing environments are built. For a symmetric matrix, the procedure computes, in exact arithmetic, the same vectors and coefficients as the Lanczos method ([Lemma 6](#54-lemma-6-orthogonality-and-hessenberg-projection)), at the cost of [Algorithm 2](#412-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks).

## References

- <span id="ref-arnoldi1951"></span>W. E. Arnoldi, [“The Principle of Minimized Iterations in the Solution of the Matrix Eigenvalue Problem”](https://doi.org/10.1090/qam/42792), *Quarterly of Applied Mathematics* 9(1):17–29, 1951.
- <span id="ref-bai2000"></span>Zhaojun Bai, James Demmel, Jack Dongarra, Axel Ruhe, and Henk van der Vorst (editors), [*Templates for the Solution of Algebraic Eigenvalue Problems: A Practical Guide*](https://netlib.org/utk/people/JackDongarra/etemplates/book.html), SIAM, 2000. Sections used here: the Hermitian [Lanczos algorithm](https://netlib.org/utk/people/JackDongarra/etemplates/node104.html) and [full reorthogonalization](https://netlib.org/utk/people/JackDongarra/etemplates/node109.html).
- <span id="ref-bauer1960"></span>Friedrich L. Bauer and Charles T. Fike, [“Norms and Exclusion Theorems”](https://doi.org/10.1007/BF01386217), *Numerische Mathematik* 2(1):137–141, 1960.
- <span id="ref-brin1998"></span>Sergey Brin and Lawrence Page, [“The Anatomy of a Large-Scale Hypertextual Web Search Engine”](https://doi.org/10.1016/S0169-7552%2898%2900110-X), *Computer Networks and ISDN Systems* 30(1–7):107–117, 1998.
- <span id="ref-cooley1965"></span>James W. Cooley and John W. Tukey, [“An Algorithm for the Machine Calculation of Complex Fourier Series”](https://doi.org/10.1090/S0025-5718-1965-0178586-1), *Mathematics of Computation* 19(90):297–301, 1965.
- <span id="ref-dembo1983"></span>Ron S. Dembo and Trond Steihaug, [“Truncated-Newton Algorithms for Large-Scale Unconstrained Optimization”](https://doi.org/10.1007/BF02592055), *Mathematical Programming* 26(2):190–212, 1983.
- <span id="ref-golub2000"></span>Gene H. Golub and Henk A. van der Vorst, [“Eigenvalue Computation in the 20th Century”](https://doi.org/10.1016/S0377-0427%2800%2900413-1), *Journal of Computational and Applied Mathematics* 123(1–2):35–65, 2000.
- <span id="ref-hestenes1952"></span>Magnus R. Hestenes and Eduard Stiefel, [“Methods of Conjugate Gradients for Solving Linear Systems”](https://doi.org/10.6028/jres.049.044), *Journal of Research of the National Bureau of Standards* 49(6):409–436, 1952.
- <span id="ref-householder1964"></span>Alston S. Householder, *The Theory of Matrices in Numerical Analysis*, Blaisdell, New York, 1964. Reprinted by Dover, 1975.
- <span id="ref-lanczos1950"></span>Cornelius Lanczos, [“An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators”](https://doi.org/10.6028/jres.045.026), *Journal of Research of the National Bureau of Standards* 45(4):255–282, 1950.
- <span id="ref-lehoucq1998"></span>Richard B. Lehoucq, Danny C. Sorensen, and Chao Yang, [*ARPACK Users’ Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods*](https://doi.org/10.1137/1.9780898719628), SIAM, 1998.
- <span id="ref-vonluxburg2007"></span>Ulrike von Luxburg, [“A Tutorial on Spectral Clustering”](https://doi.org/10.1007/s11222-007-9033-z), *Statistics and Computing* 17(4):395–416, 2007.
- <span id="ref-mescheder2017"></span>Lars Mescheder, Sebastian Nowozin, and Andreas Geiger, [“The Numerics of GANs”](https://arxiv.org/abs/1705.10461), *Advances in Neural Information Processing Systems* 30, 2017.
- <span id="ref-muntz1913"></span>Ch. Müntz, “Solution directe de l’équation séculaire et de quelques problèmes analogues transcendants”, *Comptes Rendus de l’Académie des Sciences*, Paris, 1913.
- <span id="ref-osborne1958"></span>Elmer E. Osborne, [“On Acceleration and Matrix Deflation Processes Used with the Power Method”](https://www.jstor.org/stable/2098697), *Journal of the Society for Industrial and Applied Mathematics* 6(3):279–287, 1958.
- <span id="ref-pearlmutter1994"></span>Barak A. Pearlmutter, [“Fast Exact Multiplication by the Hessian”](https://doi.org/10.1162/neco.1994.6.1.147), *Neural Computation* 6(1):147–160, 1994.
- <span id="ref-saad1980"></span>Yousef Saad, [“Variations on Arnoldi’s Method for Computing Eigenelements of Large Unsymmetric Matrices”](https://doi.org/10.1016/0024-3795%2880%2990169-X), *Linear Algebra and its Applications* 34:269–295, 1980.
- <span id="ref-saad2011"></span>Yousef Saad, [*Numerical Methods for Large Eigenvalue Problems*](https://doi.org/10.1137/1.9781611970739), revised edition, SIAM, 2011.
- <span id="ref-sorensen1992"></span>Danny C. Sorensen, [“Implicit Application of Polynomial Filters in a *k*-Step Arnoldi Method”](https://doi.org/10.1137/0613025), *SIAM Journal on Matrix Analysis and Applications* 13(1):357–385, 1992.
- <span id="ref-trefethen2005"></span>Lloyd N. Trefethen and Mark Embree, [*Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators*](https://doi.org/10.1515/9780691213101), Princeton University Press, 2005.
