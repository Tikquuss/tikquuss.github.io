---
title: "Eigenpairs from Matrix–Vector Products: Power Iteration, Deflation, and Lanczos"
date: "2026-10-12"
category: "Mathematics"
tags:
  - Linear Algebra
  - Eigenvalues
  - Power Iteration
  - Lanczos Method
  - Matrix-Free Methods
excerpt: "What power iteration, deflation, and the Lanczos method recover about a symmetric matrix from matrix–vector products alone: convergence, failure modes, exact termination, and residual certificates, with complete proofs."
---

This document studies what can be learned about the eigenvalues and eigenvectors of a real symmetric matrix $\mathbf{A}\in\mathbb{R}^{d\times d}$ that is known only through the map $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$. Storing $\mathbf{A}$ takes $d^2$ numbers, and forming it can cost much more. In many applications, however, the product $\mathbf{A}\mathbf{x}$ is cheap, because it follows directly from how $\mathbf{A}$ is defined, whether $\mathbf{A}$ is built from a data matrix, a sparse graph or a discretized differential operator, or is the Hessian of a function whose gradient can be computed ([Section 2](#2-matrices-known-only-through-their-products) gives more examples). Two families of methods need nothing more than such products: **[power iteration](#3-power-iteration)** ([Section 3](#3-power-iteration)), which extracts one eigenpair at a time and reaches further eigenpairs through [shifts](#32-corollary-1-recovering-algebraic-extrema-by-shifting) and [deflation](#33-lemma-1-orthogonal-deflation), and the **[Lanczos method](#4-the-lanczos-method)** ([Section 4](#4-the-lanczos-method)), which builds an orthonormal basis of a [Krylov subspace](#41-definition-1-krylov-subspace) and solves a small tridiagonal eigenvalue problem inside it.

The results assume neither positive definiteness nor distinct eigenvalues. Each statement says precisely which eigenpairs are recovered, and the remarks say which are not. All the delicate cases are treated explicitly: a dominant eigenvalue of either sign, opposite eigenvalues of equal magnitude, repeated eigenvalues, and zero eigenvalues. Sections [3](#3-power-iteration) and [4](#4-the-lanczos-method) each end with an implementable procedure, [Algorithm 1](#37-algorithm-1-deflated-squared-power-iteration-with-sign-separation) for power iteration and [Algorithm 2](#411-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks) for the Lanczos method. Both stop when a residual computed with the original matrix falls below a tolerance. The figures run these procedures on matrices with prescribed spectra and on three matrices of [Section 2](#2-matrices-known-only-through-their-products): the adjacency matrix of a graph, a covariance matrix and the Hessian of a neural network. The code that produces them, with implementations of both algorithms, is [on GitHub](https://github.com/Tikquuss/eigenpairs_from_matrix-vector_products).

The central object throughout is the decomposition of a vector along the eigenspaces of $\mathbf{A}$. Let $\Lambda(\mathbf{A})$ be the set of distinct eigenvalues of $\mathbf{A}$, and let $\Pi_\lambda$ be the orthogonal projection onto the eigenspace of $\lambda\in\Lambda(\mathbf{A})$. Then, for every vector $\mathbf{v}\in\mathbb{R}^d$ and every integer $k\ge0$,

$$
\mathbf{v}=\sum_{\lambda\in\Lambda(\mathbf{A})}\Pi_\lambda\mathbf{v},
\qquad
\mathbf{A}^k\mathbf{v}=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^k\,\Pi_\lambda\mathbf{v}.
$$

Started from $\mathbf{v}$, power iteration keeps the last vector of the sequence $\mathbf{v},\mathbf{A}\mathbf{v},\mathbf{A}^2\mathbf{v},\dots$ and lets the factors $\lambda^k$ select the eigenvalue of largest magnitude. The Lanczos method keeps the whole sequence and optimizes over its span. Every proof below follows the components $\Pi_\lambda\mathbf{v}$ through these operations.

## 1. Notation and preliminaries

### Notation

* $\|\cdot\|$ is the Euclidean norm, $\mathbb{I}_d$ the $d\times d$ identity matrix, $\delta_{i\ell}$ the Kronecker delta, and $\mathbf{e}_m=[0,\ldots,0,1]^\top\in\mathbb{R}^m$ the last standard basis vector of $\mathbb{R}^m$.
* For a subspace $S\subseteq\mathbb{R}^d$, $S^\perp=\{\mathbf{x}\in\mathbb{R}^d:\mathbf{x}^\top\mathbf{s}=0 \ \forall \mathbf{s}\in S\}$ is its orthogonal complement.
* For a matrix $\mathbf{M}$ with $n$ columns, $\ker(\mathbf{M})=\{\mathbf{x}\in\mathbb{R}^n:\mathbf{M}\mathbf{x}=0\}$ is its null space and $\operatorname{range}(\mathbf{M})=\{\mathbf{M}\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}$ is its column space. 

* Throughout, $\mathbf{A}\in\mathbb{R}^{d\times d}$ is symmetric. We let $\Lambda(\mathbf{A})$ be the set of its **distinct** eigenvalues, so that a repeated eigenvalue appears once in $\Lambda(\mathbf{A})$, and we write $\lambda_{\min}(\mathbf{A})=\min\Lambda(\mathbf{A})$ and $\lambda_{\max}(\mathbf{A})=\max\Lambda(\mathbf{A})$.
For each $\lambda\in\Lambda(\mathbf{A})$, $E_\lambda=\ker(\mathbf{A}-\lambda\mathbb{I}_d)$ is the eigenspace of $\lambda$, and $\Pi_\lambda$ is the orthogonal projection onto $E_\lambda$. If the columns of $\mathbf{U}_\lambda\in\mathbb{R}^{d\times m_\lambda}$ form an orthonormal basis of $E_\lambda$, where $m_\lambda$ is the multiplicity of $\lambda$, then $\Pi_\lambda=\mathbf{U}_\lambda\mathbf{U}_\lambda^\top$. An expression in terms of $\mathbf{A}$ alone is given below.

* For vectors $\mathbf{u}_1,\dots,\mathbf{u}_m\in\mathbb{R}^d$, $[\mathbf{u}_1,\dots,\mathbf{u}_m]\in\mathbb{R}^{d\times m}$ is the matrix with these columns, and $\operatorname{span}\{\mathbf{u}_1,\dots,\mathbf{u}_m\}$ is the set of their linear combinations
* $\operatorname{diag}(c_1,\dots,c_m)$ is the diagonal matrix with diagonal entries $c_1,\dots,c_m \in \mathbb{R}$. 


* $\mathcal{N}(0,\mathbb{I}_d)$ is the standard Gaussian distribution on $\mathbb{R}^d$, and $\operatorname{sgn}(t)\in\{-1,1\}$ is the sign of a real number $t\neq0$.

### Spectral decomposition

The <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="spectral-theorem" aria-label="State the spectral theorem used here">spectral theorem</button><span id="spectral-theorem" class="explanation-popover" popover="auto" role="note" aria-label="Spectral theorem" data-label="Theorem">Every real symmetric matrix has an orthonormal basis of eigenvectors. Collecting the basis vectors that share the eigenvalue $\lambda$ as the columns of a matrix $\mathbf U_\lambda$ gives $\Pi_\lambda=\mathbf U_\lambda\mathbf U_\lambda^\top$, and the three identities follow from the orthonormality of the basis.</span></span> gives

$$
\mathbf{A}=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda\,\Pi_\lambda,
\qquad
\mathbb{I}_d=\sum_{\lambda\in\Lambda(\mathbf{A})}\Pi_\lambda,
\qquad
\Pi_\lambda\Pi_\mu=0
\quad\text{when }\lambda\neq\mu .
$$

Since each projection also satisfies $\Pi_\lambda^2=\Pi_\lambda$, the powers of $\mathbf{A}$ act on each eigenspace separately, and so does every polynomial $p$:

$$
\mathbf{A}^k=\sum_{\lambda\in\Lambda(\mathbf{A})}\lambda^k\,\Pi_\lambda,
\qquad
p(\mathbf{A})=\sum_{\lambda\in\Lambda(\mathbf{A})}p(\lambda)\,\Pi_\lambda .
$$

Taking for $p$ the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="lagrange-projection" aria-label="Explain the Lagrange polynomial and why it gives the projection">Lagrange polynomial</button><span id="lagrange-projection" class="explanation-popover" popover="auto" role="note" aria-label="Lagrange polynomial" data-label="Definition">Given finitely many distinct numbers, the Lagrange polynomial attached to one of them equals $1$ at that number and $0$ at all the others. For $\lambda\in\Lambda(\mathbf A)$ it is <span class="explanation-popover__equation">$\displaystyle\ell_\lambda(t)=\prod_{\mu\in\Lambda(\mathbf A),\,\mu\neq\lambda}\frac{t-\mu}{\lambda-\mu}.$</span> Each factor vanishes at its own $\mu$ and equals $1$ at $t=\lambda$, so $\ell_\lambda(\lambda)=1$ and $\ell_\lambda(\mu)=0$ for every other $\mu\in\Lambda(\mathbf A)$. Its degree is $|\Lambda(\mathbf A)|-1$. Substituting it for $p$ in $p(\mathbf A)=\sum_{\nu}p(\nu)\,\Pi_\nu$ keeps only the term $\nu=\lambda$, so $\ell_\lambda(\mathbf A)=\Pi_\lambda$. For example, if $\Lambda(\mathbf A)=\{1,3\}$, then $\ell_1(t)=(3-t)/2$ and $\Pi_1=(3\,\mathbb I_d-\mathbf A)/2$.</span></span> that equals $1$ at $\lambda$ and $0$ at the other distinct eigenvalues expresses each projection through $\mathbf{A}$ alone:

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
* **Newton-type optimization.** To minimize $f(\theta)$, Newton's method takes the step $\Delta\theta$ that solves the linear system $\nabla^2f(\theta)\,\Delta\theta=-\nabla f(\theta)$. Truncated Newton methods ([Dembo and Steihaug, 1983](#ref-dembo1983)) solve this system by the conjugate gradient method ([Hestenes and Stiefel, 1952](#ref-hestenes1952)), which accesses the Hessian only through products.
* **Molecular vibrations.** The potential energy $f(\theta)$ of a molecule depends on the positions $\theta\in\mathbb{R}^{3N}$ of its $N$ atoms. Near an equilibrium $\theta^\star$, where $\nabla f(\theta^\star)=0$, small displacements obey Newton's law of motion $\mathbf{M}\ddot\theta=-\nabla^2f(\theta^\star)(\theta-\theta^\star)$, where $\mathbf{M}$ is the diagonal matrix of atomic masses, each repeated for the three coordinates of its atom. The eigenvectors of the mass-weighted Hessian $\mathbf{M}^{-1/2}\nabla^2f(\theta^\star)\mathbf{M}^{-1/2}$ are the vibration patterns of the molecule (its normal modes), and its eigenvalues are the squared vibration frequencies $\omega^2$.

Such a Hessian is symmetric by <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="schwarz-theorem" aria-label="Explain why the Hessian is symmetric">Schwarz's theorem</button><span id="schwarz-theorem" class="explanation-popover" popover="auto" role="note" aria-label="Schwarz's theorem" data-label="Theorem">When the second partial derivatives of $f$ are continuous near $\theta$, they do not depend on the order of differentiation: $\partial^2f/\partial\theta_i\partial\theta_j=\partial^2f/\partial\theta_j\partial\theta_i$. The Hessian is therefore a real symmetric matrix.</span></span>. In that case, each product can be evaluated as
$\mathbf{A}\mathbf{x}=\nabla_\theta\bigl(\nabla_\theta f(\theta)^\top\mathbf{x}\bigr),$
with $\mathbf{x}$ held constant during differentiation. The scalar $\nabla_\theta f(\theta)^\top\mathbf{x}$ is the directional derivative of $f$ along $\mathbf{x}$. Its gradient with respect to $\theta$ is $\nabla_\theta^2f(\theta)^\top\mathbf{x}$, which equals $\mathbf{A}\mathbf{x}$ because the Hessian is symmetric. Automatic differentiation therefore returns the product at a <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="hvp-cost" aria-label="Explain the cost of a Hessian-vector product">small multiple of the cost of a gradient</button><span id="hvp-cost" class="explanation-popover" popover="auto" role="note" aria-label="Cost of a Hessian-vector product" data-label="Fact">The product is obtained by differentiating the gradient computation once more, in forward or reverse mode. Each pass costs a small constant multiple of the original computation, so the product costs about as much as a few gradient evaluations.</span></span> ([Pearlmutter, 1994](#ref-pearlmutter1994)), and the $d\times d$ matrix is never formed.

The point $\theta$ and everything that defines $f$ must remain fixed during an eigensolver run. For a training loss, evaluating it on a different random subset of the data (a minibatch) between two products changes the matrix being analyzed, and the statements below then no longer describe the computation.


Every symmetric matrix $\mathbf{A}$ is in fact a Hessian: the quadratic $\mathbf{x}\mapsto\frac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ has the Hessian $\mathbf{A}$ at every point. This does not help to compute products, since the gradient of that quadratic is $\mathbf{A}\mathbf{x}$ itself. The Hessian route pays off when $f$ is given by a program whose gradient is cheap, so that curvature is obtained from first-order code. For the matrices of the following section, the products come instead from the structure that defines $\mathbf{A}$.

### 2.2 Other matrices known through their products

Products are just as cheap in many cases where $\mathbf{A}$ is not presented as a Hessian.

* **Gram and covariance matrices.** For a data matrix $\mathbf{X}\in\mathbb{R}^{n\times d}$, whose $n$ rows are observations of $d$ variables, the product $\mathbf{X}^\top\mathbf{X}\mathbf{x}=\mathbf{X}^\top(\mathbf{X}\mathbf{x})$ costs $O(nd)$ operations, whereas forming $\mathbf{X}^\top\mathbf{X}$ costs $O(nd^2)$ and storing it costs $O(d^2)$. When each column of $\mathbf{X}$ has zero mean, the sample covariance matrix of the $d$ variables is the average of the outer products of the observations, $\frac1n\mathbf{X}^\top\mathbf{X}=\frac1n\sum_{i=1}^n\mathbf{X}_i\mathbf{X}_i^\top$, where $\mathbf{X}_i\in\mathbb{R}^d$ is the $i$-th row of $\mathbf{X}$. Its leading eigenvector, the first principal component, is the direction of largest variance in the data: it maximizes the variance $\frac1n\|\mathbf{X}\mathbf{u}\|^2$ of the data projected on $\mathbf{u}$ over unit vectors $\mathbf{u}$.
* **Gauss–Newton matrices.** Many objectives are compositions $f(\theta)=\psi\bigl(F(\theta)\bigr)$ of a map $F:\mathbb{R}^d\to\mathbb{R}^p$ and a function $\psi:\mathbb{R}^p\to\mathbb{R}$, both twice differentiable. For the training loss of [Section 2.1](#21-hessians), $F(\theta)$ stacks the outputs $\mathbf{z}_1(\theta),\dots,\mathbf{z}_n(\theta)$ into one vector of $\mathbb{R}^p$, and $\psi(\mathbf{z})=\frac1n\sum_{i=1}^n\ell(\mathbf{z}_i,\mathbf{y}_i)$ for $\mathbf{z}=[\mathbf{z}_1;\dots;\mathbf{z}_n]$. Let $\mathbf{J}\in\mathbb{R}^{p\times d}$ be the Jacobian of $F$ at $\theta$, with entries $J_{kj}=\partial F_k(\theta)/\partial\theta_j$, and let $\mathbf{H}=\nabla^2\psi\bigl(F(\theta)\bigr)\in\mathbb{R}^{p\times p}$. The chain rule gives $\nabla^2f(\theta)=\mathbf{J}^\top\mathbf{H}\mathbf{J}+\sum_{k=1}^p\partial_k\psi\bigl(F(\theta)\bigr)\,\nabla^2F_k(\theta)$, where $\partial_k\psi$ is the partial derivative of $\psi$ with respect to its $k$-th argument. The generalized Gauss–Newton matrix $\mathbf{G}=\mathbf{J}^\top\mathbf{H}\mathbf{J}$ keeps the first term. It is symmetric, and positive semidefinite when $\psi$ is convex. It is applied as $\mathbf{G}\mathbf{x}=\mathbf{J}^\top\bigl(\mathbf{H}(\mathbf{J}\mathbf{x})\bigr)$, where automatic differentiation computes $\mathbf{J}\mathbf{x}$ in forward mode and $\mathbf{J}^\top\mathbf{u}$ in reverse mode without forming $\mathbf{J}$. For least squares, $\psi(\mathbf{z})=\frac12\|\mathbf{z}-\mathbf{y}\|^2$ for a target $\mathbf{y}\in\mathbb{R}^p$, so $\mathbf{H}=\mathbb{I}_p$ and $\mathbf{G}=\mathbf{J}^\top\mathbf{J}$. When $F$ is linear, $F(\theta)=\mathbf{J}\theta$, every $\nabla^2F_k$ vanishes and $\mathbf{G}$ is the exact Hessian. When $F$ is nonlinear, $\mathbf{G}=\mathbf{J}^\top\mathbf{J}$ is the classical Gauss–Newton approximation of the Hessian. For the squared loss and the softmax cross-entropy of a neural network, $\mathbf{G}$ is also the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="fisher-gauss-newton" aria-label="Explain why the Gauss-Newton matrix equals the Fisher information">Fisher information matrix</button><span id="fisher-gauss-newton" class="explanation-popover" popover="auto" role="note" aria-label="Fisher information and Gauss-Newton" data-label="Fact">Suppose the model gives each target a probability density $p(\mathbf y\,|\,\mathbf z)$ that depends on the output $\mathbf z=F(\theta)$, and that $\ell$ is the corresponding negative log-likelihood. The Fisher information matrix is <span class="explanation-popover__equation">$\mathbb E_{\mathbf y\sim p(\cdot\,|\,F(\theta))}\bigl[\nabla_\theta\log p(\mathbf y\,|\,F(\theta))\,\nabla_\theta\log p(\mathbf y\,|\,F(\theta))^\top\bigr].$</span> By the chain rule it equals $\mathbf J^\top\mathbf I_{\mathbf z}\mathbf J$ at $\mathbf z=F(\theta)$, where <span class="explanation-popover__equation">$\mathbf I_{\mathbf z}=\mathbb E_{\mathbf y\sim p(\cdot\,|\,\mathbf z)}\bigl[\nabla_{\mathbf z}\log p(\mathbf y\,|\,\mathbf z)\,\nabla_{\mathbf z}\log p(\mathbf y\,|\,\mathbf z)^\top\bigr]=\mathbb E_{\mathbf y\sim p(\cdot\,|\,\mathbf z)}\bigl[-\nabla_{\mathbf z}^2\log p(\mathbf y\,|\,\mathbf z)\bigr].$</span> When $-\nabla_{\mathbf z}^2\log p(\mathbf y\,|\,\mathbf z)$ does not depend on $\mathbf y$, the last expectation is $\mathbf H$, so the Fisher matrix is $\mathbf G$. This holds for a Gaussian with mean $\mathbf z$ (the squared loss, $\mathbf H=\mathbb I$) and for a categorical distribution with probabilities $s_k=e^{z_k}/\sum_je^{z_j}$ (the softmax cross-entropy, $\mathbf H=\operatorname{diag}(\mathbf s)-\mathbf s\mathbf s^\top$).</span></span> of the model.
* **Graphs.** For an undirected graph on $d$ nodes, the adjacency matrix $\mathbf{W}$ has the entries $W_{ij}=W_{ji}\ge0$, the weight of the edge between nodes $i$ and $j$, with $W_{ij}=0$ when there is no edge. It is symmetric, and so is the Laplacian $\mathbf{L}=\operatorname{diag}(\mathbf{W}\mathbf{1})-\mathbf{W}$, where $\mathbf{1}$ is the all-ones vector. A product with either matrix costs one pass over the edges. Since $\mathbf{x}^\top\mathbf{L}\mathbf{x}=\sum_{i<j}W_{ij}(x_i-x_j)^2$, the eigenvectors of $\mathbf{L}$ with the smallest eigenvalues vary little across heavily weighted edges. They are used to split the graph into well-connected groups, in spectral clustering and graph partitioning ([von Luxburg, 2007](#ref-vonluxburg2007)).
* **Discretized differential operators.** Finite-difference and finite-element discretizations of self-adjoint differential operators, such as those of heat conduction, elasticity or quantum mechanics, are applied point by point or element by element, in time proportional to the number of unknowns and without assembling the matrix. For instance, the operator $u\mapsto-u''$ on $[0,1]$ with $u(0)=u(1)=0$, discretized on the grid $t_i=ih$ with $h=1/(d+1)$, becomes the matrix defined by $(\mathbf{A}\mathbf{x})_i=(2x_i-x_{i-1}-x_{i+1})/h^2$ for $i\in\{1,\dots,d\}$, with $x_0=x_{d+1}=0$. [Lanczos (1950)](#ref-lanczos1950) designed his method for linear differential and integral operators.
* **Structured matrices.** A symmetric Toeplitz matrix is constant along each diagonal, $A_{ij}=a_{|i-j|}$ for numbers $a_0,\dots,a_{d-1}$, as is the covariance matrix of a stationary time series. It is applied in $O(d\log(d))$ operations by embedding it in a circulant matrix of size $2d$, whose rows are cyclic shifts of one another and which the fast Fourier transform ([Cooley and Tukey, 1965](#ref-cooley1965)) diagonalizes. Operators that act separately along the two axes of a grid are Kronecker products. A Kronecker product of symmetric matrices $\mathbf{M}_1\in\mathbb{R}^{d_1\times d_1}$ and $\mathbf{M}_2\in\mathbb{R}^{d_2\times d_2}$ is applied without being formed, through $(\mathbf{M}_1\otimes\mathbf{M}_2)\operatorname{vec}(\mathbf{Y})=\operatorname{vec}(\mathbf{M}_2\mathbf{Y}\mathbf{M}_1^\top)$ for $\mathbf{Y}\in\mathbb{R}^{d_2\times d_1}$, where $\operatorname{vec}(\cdot)$ stacks the columns of a matrix.
* **Shifted inverses.** If linear systems with $\mathbf{A}-\sigma\mathbb{I}_d$ can be solved for a shift $\sigma$ that is not an eigenvalue, for instance with a sparse factorization computed once, then $\mathbf{x}\mapsto(\mathbf{A}-\sigma\mathbb{I}_d)^{-1}\mathbf{x}$ is a product with a symmetric matrix that has the same eigenvectors as $\mathbf{A}$ and the eigenvalues $1/(\lambda-\sigma)$. Power iteration and the Lanczos method applied to it find the eigenvalues of $\mathbf{A}$ closest to $\sigma$. This is the shift-and-invert strategy.

All these matrices are symmetric, and the results below rely on symmetry through the spectral theorem. For a nonsymmetric matrix accessed through products, such as the transition matrix of a random walk on a directed graph, whose entry $(i,j)$ is the probability of stepping from node $i$ to node $j$, power iteration can still be used under conditions not covered here, and the Arnoldi method ([Arnoldi, 1951](#ref-arnoldi1951)) replaces the Lanczos method.


## 3. Power iteration

In this part, we first establish the convergence conditions for ordinary power iteration, then describe how shifting and deflation recover additional eigenpairs. Starting from a unit vector $\mathbf{v}_0$, power iteration alternates a product with $\mathbf{A}$ and a normalization:

$$
\mathbf{v}_{k+1}=\frac{\mathbf{A}\mathbf{v}_k}{\|\mathbf{A}\mathbf{v}_k\|},
\qquad k\ge0 .
$$

The numerical analyst Alston S. Householder attributed the first treatment of the method to [Müntz (1913)](#ref-muntz1913) in his book [*The Theory of Matrices in Numerical Analysis*](#ref-householder1964) (1964), as recalled in the historical survey of [Golub and van der Vorst (2000)](#ref-golub2000). Deflation schemes built on it were analyzed early on, for example by [Osborne (1958)](#ref-osborne1958). The eigenvalue estimate of power iteration is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="rayleigh-quotient" aria-label="Define the Rayleigh quotient">Rayleigh quotient</button><span id="rayleigh-quotient" class="explanation-popover" popover="auto" role="note" aria-label="Rayleigh quotient" data-label="Definition">For a unit vector $\mathbf v$, the Rayleigh quotient $\mathbf v^\top\mathbf A\mathbf v$ is the curvature of $\mathbf x\mapsto\frac12\mathbf x^\top\mathbf A\mathbf x$ along $\mathbf v$. It lies between $\lambda_{\min}(\mathbf A)$ and $\lambda_{\max}(\mathbf A)$. It equals $\lambda$ for every unit vector $\mathbf v$ of the eigenspace $E_\lambda$, since $\mathbf A\mathbf v=\lambda\mathbf v$ gives $\mathbf v^\top\mathbf A\mathbf v=\lambda\|\mathbf v\|^2=\lambda$. The converse fails: $\mathbf v^\top\mathbf A\mathbf v=\lambda$ does not make $\mathbf v$ an eigenvector, except when $\lambda$ is $\lambda_{\min}(\mathbf A)$ or $\lambda_{\max}(\mathbf A)$.</span></span> $\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k$.

### 3.1 Proposition 1 [Convergence of power iteration]

Let $\mathbf{v}_0\in\mathbb{R}^d$ be a unit vector, and define

$$
\Lambda_0=\{\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{v}_0\neq0\}.
$$

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

Each application of $\mathbf{A}$ multiplies the component of the current vector in $E_\lambda$ by $\lambda$, so after $k$ steps the component in $E_\lambda$ carries the factor $\lambda^k$:

$$
\mathbf{A}^k\mathbf{v}_0=\textcolor{#2563eb}{\lambda_\star^k\,\Pi_{\lambda_\star}\mathbf{v}_0}+\textcolor{#ea580c}{\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda^k\,\Pi_\lambda\mathbf{v}_0}.
$$

Among the components present in $\mathbf{v}_0$, the <span style="color:#2563eb">blue</span> one, with the largest $|\lambda|$, grows fastest, and the normalization removes the common factor $|\lambda_\star|^k$, so the <span style="color:#ea580c">orange</span> components fade relative to it. Only the sign of $\lambda_\star^k$ survives, which is why the statement multiplies $\mathbf{v}_k$ by $\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k$. The limit is the normalized projection of $\mathbf{v}_0$ onto the whole eigenspace $E_{\lambda_\star}$: inside a repeated eigenspace, the direction is chosen by the initialization, not by the method.

#### Proof

Set $\mathbf{p}_\lambda=\Pi_\lambda\mathbf{v}_0$. The spectral decomposition gives

$$
\mathbf{v}_0=\sum_{\lambda\in\Lambda_0}\mathbf{p}_\lambda,
\qquad
\mathbf{A}^k\mathbf{v}_0=\sum_{\lambda\in\Lambda_0}\lambda^k\mathbf{p}_\lambda
\quad(k\ge1).
$$

**Well-definedness.** Because $\mathbf{p}_{\lambda_\star}\neq0$ and $\lambda_\star\neq0$, the component

$$
\Pi_{\lambda_\star}\mathbf{A}^k\mathbf{v}_0=\lambda_\star^k\mathbf{p}_{\lambda_\star}
$$

is nonzero for every $k\ge0$. Consequently, $\mathbf{A}^k\mathbf{v}_0\neq0$ for every $k\ge0$. Since $\|\mathbf{v}_0\|=1$, repeated normalization satisfies

$$
\mathbf{v}_k=\frac{\mathbf{A}^k\mathbf{v}_0}{\|\mathbf{A}^k\mathbf{v}_0\|}.
$$

Indeed, this identity holds at $k=0$. If it holds at iteration $k$, then

$$
\begin{aligned}
\mathbf{v}_{k+1}
&=\frac{\mathbf{A}\bigl(\mathbf{A}^k\mathbf{v}_0/\|\mathbf{A}^k\mathbf{v}_0\|\bigr)}{\bigl\|\mathbf{A}\bigl(\mathbf{A}^k\mathbf{v}_0/\|\mathbf{A}^k\mathbf{v}_0\|\bigr)\bigr\|}\\
&=\frac{\mathbf{A}^{k+1}\mathbf{v}_0}{\|\mathbf{A}^{k+1}\mathbf{v}_0\|},
\end{aligned}
$$

because the positive scalar $1/\|\mathbf{A}^k\mathbf{v}_0\|$ cancels between the numerator and the denominator.

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
&=\frac{\lambda_\star^k\bigl(\mathbf{p}_{\lambda_\star}+\mathbf{r}_k\bigr)}{|\lambda_\star|^k\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}},\\
\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k
&=\frac{\textcolor{#2563eb}{\mathbf{p}_{\lambda_\star}}+\textcolor{#ea580c}{\mathbf{r}_k}}{\sqrt{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2}}
\longrightarrow\frac{\mathbf{p}_{\lambda_\star}}{\|\mathbf{p}_{\lambda_\star}\|}.
\end{aligned}
$$

The <span style="color:#2563eb">blue term</span> is the component along the dominant eigenspace, and it keeps its size after rescaling. The <span style="color:#ea580c">orange term</span> collects every other component, and it shrinks geometrically. Since $\mathbf{p}_{\lambda_\star}\in E_{\lambda_\star}$, the limit $\mathbf{q}_\star$ satisfies $\mathbf{A}\mathbf{q}_\star=\lambda_\star\mathbf{q}_\star$.

**Rayleigh quotient.** For $k\ge1$, orthogonality gives

$$
\begin{aligned}
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k
&=\frac{(\mathbf{A}^k\mathbf{v}_0)^\top\mathbf{A}(\mathbf{A}^k\mathbf{v}_0)}{\|\mathbf{A}^k\mathbf{v}_0\|^2}\\
&=\frac{\displaystyle\sum_{\lambda\in\Lambda_0}\lambda^{2k+1}\|\mathbf{p}_\lambda\|^2}{\displaystyle\sum_{\lambda\in\Lambda_0}\lambda^{2k}\|\mathbf{p}_\lambda\|^2}\\
&=\frac{\displaystyle\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2+\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\lambda\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\displaystyle\|\mathbf{p}_{\lambda_\star}\|^2+\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}\\
&\longrightarrow\frac{\lambda_\star\|\mathbf{p}_{\lambda_\star}\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2}=\lambda_\star .
\end{aligned}
$$

The third line divides the numerator and the denominator by $\lambda_\star^{2k}>0$.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

If $\mathbf{A}\mathbf{v}_0=0$, then $\mathbf{v}_0$ is already an eigenvector with eigenvalue zero. In this case, the algorithm should return $(0,\mathbf{v}_0)$ without attempting to normalize $\mathbf{A}\mathbf{v}_0$.

#### Remark 2 [A negative dominant eigenvalue]

If $\lambda_\star>0$, [Proposition 1](#31-proposition-1-convergence-of-power-iteration) gives $\mathbf{v}_k\to\mathbf{q}_\star$. If $\lambda_\star<0$, it gives $(-1)^k\mathbf{v}_k\to\mathbf{q}_\star$, or, equivalently,

$$
\mathbf{v}_{2k}\longrightarrow\mathbf{q}_\star,
\qquad
\mathbf{v}_{2k+1}\longrightarrow-\mathbf{q}_\star .
$$

Thus, in the latter case, the even and odd subsequences converge to opposite unit vectors. The raw sequence $\{\mathbf{v}_k\}_{k\ge0}$ does not converge, but

$$
\min\bigl\{\|\mathbf{v}_k-\mathbf{q}_\star\|,\|\mathbf{v}_k+\mathbf{q}_\star\|\bigr\}\longrightarrow0 .
$$

A stopping rule that compares consecutive iterates, such as $\|\mathbf{v}_{k+1}-\mathbf{v}_k\|\le\varepsilon$ with $\varepsilon<2$, is therefore eventually never satisfied when $\lambda_\star<0$, because $\|\mathbf{v}_{k+1}-\mathbf{v}_k\|\to2$, even though the eigendirection has converged. The Rayleigh quotient converges in both cases.

#### Remark 3 [Rate of convergence]

The proof also gives the speed of convergence. Let

$$
\rho_2=\max\{|\lambda|:\lambda\in\Lambda_0\setminus\{\lambda_\star\}\},
$$

with $\rho_2=0$ when $\Lambda_0=\{\lambda_\star\}$, and let $\phi_0\in[0,\pi/2)$ be the angle between $\mathbf{v}_0$ and $E_{\lambda_\star}$, so that $\tan(\phi_0)=\|\mathbf{v}_0-\mathbf{p}_{\lambda_\star}\|/\|\mathbf{p}_{\lambda_\star}\|$. Then, for every $k\ge1$,

$$
\Bigl\|\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k-\mathbf{q}_\star\Bigr\|
\le\textcolor{#ea580c}{\left(\frac{\rho_2}{|\lambda_\star|}\right)^{k}}\tan(\phi_0),
\qquad
\bigl|\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k-\lambda_\star\bigr|
\le2|\lambda_\star|\textcolor{#ea580c}{\left(\frac{\rho_2}{|\lambda_\star|}\right)^{2k}}\tan^2(\phi_0).
$$

Indeed, the sum defining $\|\mathbf{r}_k\|^2$ gives $\|\mathbf{r}_k\|\le(\rho_2/|\lambda_\star|)^k\|\mathbf{v}_0-\mathbf{p}_{\lambda_\star}\|$, because $\sum_{\lambda\neq\lambda_\star}\|\mathbf{p}_\lambda\|^2=\|\mathbf{v}_0-\mathbf{p}_{\lambda_\star}\|^2$. Write $t_k=\|\mathbf{r}_k\|/\|\mathbf{p}_{\lambda_\star}\|$. The unit vector $\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k$ has inner product $1/\sqrt{1+t_k^2}$ with $\mathbf{q}_\star$, so

$$
\Bigl\|\bigl(\operatorname{sgn}(\lambda_\star)\bigr)^k\mathbf{v}_k-\mathbf{q}_\star\Bigr\|^2
=2-\frac{2}{\sqrt{1+t_k^2}}
\le t_k^2,
$$

using $1/\sqrt{1+t}\ge1-t/2$ for $t\ge0$. For the Rayleigh quotient, subtracting $\lambda_\star$ from the third line of the last display of the proof gives

$$
\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k-\lambda_\star
=\frac{\displaystyle\sum_{\lambda\in\Lambda_0\setminus\{\lambda_\star\}}(\lambda-\lambda_\star)\left|\frac{\lambda}{\lambda_\star}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\|\mathbf{p}_{\lambda_\star}\|^2+\|\mathbf{r}_k\|^2},
$$

and $|\lambda-\lambda_\star|<2|\lambda_\star|$ bounds the absolute value of the numerator by $2|\lambda_\star|\,\|\mathbf{r}_k\|^2$. The eigenvector error therefore shrinks by the <span style="color:#ea580c">factor</span> $\textcolor{#ea580c}{\rho_2/|\lambda_\star|}$ per product, and the eigenvalue error by its square. The factor can be arbitrarily close to one. [Remark 5](#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude) below is the limiting case $\rho_2=|\lambda_\star|$, in which convergence can fail altogether.

#### Remark 4 [Which eigenvalue is found]

The hypothesis of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) concerns distinct eigenvalues present in the initialization. It does not require the dominant eigenvalue to be simple, since the limit is the normalized projection of $\mathbf{v}_0$ onto its entire eigenspace. To obtain an eigenvalue of globally largest absolute value, the initialization must have a nonzero projection onto a corresponding eigenspace. For example, if

$$
\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d),
\qquad
\mathbf{v}_0=\frac{\mathbf{g}}{\|\mathbf{g}\|},
$$

then $\Pi_\lambda\mathbf{v}_0\neq0$ for every $\lambda\in\Lambda(\mathbf{A})$ with probability one. Indeed, for each $\lambda$, the event $\Pi_\lambda\mathbf{g}=0$ means that $\mathbf{g}$ belongs to the proper linear subspace $E_\lambda^\perp$, which has <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="gaussian-null-subspace" aria-label="Explain why a proper subspace has Gaussian probability zero">Gaussian probability zero</button><span id="gaussian-null-subspace" class="explanation-popover" popover="auto" role="note" aria-label="Gaussian measure of a proper subspace" data-label="Fact">A proper subspace $S$ lies in a hyperplane $\{\mathbf x:\mathbf c^\top\mathbf x=0\}$ with $\mathbf c\neq0$. For $\mathbf g\sim\mathcal N(0,\mathbb I_d)$, the number $\mathbf c^\top\mathbf g$ is Gaussian with variance $\|\mathbf c\|^2\gt0$, so it equals zero with probability zero.</span></span>. There are only finitely many eigenspaces. Consequently, such an initialization satisfies the hypotheses of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) whenever $\mathbf{A}\neq0$ and exactly one distinct eigenvalue of $\mathbf{A}$ has maximal absolute value.

#### Remark 5 [Failure for opposite eigenvalues of equal dominant magnitude]

The condition examined here is the strict dominance assumption in [Proposition 1](#31-proposition-1-convergence-of-power-iteration):

$$
|\lambda_\star|>|\lambda|
\qquad
\text{for every }\lambda\in\Lambda_0\setminus\{\lambda_\star\}.
$$

Suppose that this condition fails for a real symmetric matrix with a nonzero eigenvalue present in the initialization. Then the largest absolute value $\rho>0$ over $\Lambda_0$ is attained by two distinct eigenvalues present in the initialization. Two distinct real numbers with the same absolute value are opposite, so both $+\rho$ and $-\rho$ are present. We show that ordinary power iteration can then fail to converge to any eigendirection, even while its Rayleigh quotient converges.

Suppose that

$$
\mathbf{p}_+=\Pi_\rho\mathbf{v}_0\neq0,
\qquad
\mathbf{p}_-=\Pi_{-\rho}\mathbf{v}_0\neq0,
$$

and that every other eigenvalue present in $\mathbf{v}_0$ has absolute value smaller than $\rho$. For $k\ge1$, define the scaled remainder

$$
\mathbf{r}_k=\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\left(\frac{\lambda}{\rho}\right)^k\mathbf{p}_\lambda .
$$

Then

$$
\|\mathbf{r}_k\|^2=\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\left|\frac{\lambda}{\rho}\right|^{2k}\|\mathbf{p}_\lambda\|^2\longrightarrow0,
$$

and

$$
\mathbf{v}_k=\frac{\textcolor{#2563eb}{\mathbf{p}_+}+\textcolor{#ea580c}{(-1)^k\mathbf{p}_-}+\mathbf{r}_k}{\sqrt{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2+\|\mathbf{r}_k\|^2}} .
$$

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
&=\frac{\displaystyle\rho\|\mathbf{p}_+\|^2-\rho\|\mathbf{p}_-\|^2+\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\lambda\left|\frac{\lambda}{\rho}\right|^{2k}\|\mathbf{p}_\lambda\|^2}{\displaystyle\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2+\sum_{\substack{\lambda\in\Lambda_0\\|\lambda|<\rho}}\left|\frac{\lambda}{\rho}\right|^{2k}\|\mathbf{p}_\lambda\|^2}\\
&\longrightarrow\rho\,\frac{\|\mathbf{p}_+\|^2-\|\mathbf{p}_-\|^2}{\|\mathbf{p}_+\|^2+\|\mathbf{p}_-\|^2}.
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

Thus a constant or convergent Rayleigh quotient does not by itself establish convergence to an eigenpair. This obstruction motivates the squared-power construction of [Proposition 2](#34-proposition-2-power-iteration-with-sign-separation) below and the use of an eigenpair residual as a stopping criterion.

<figure class="article-figure" id="figure-power-iteration">
  <img src="/images/blog/matrix-vector-products/power-iteration.svg" width="1040" height="430" alt="Two panels of error curves against the number of matrix-vector products. Left: geometric decay of the sign-corrected eigenvector error and the faster decay of the Rayleigh-quotient error for a negative dominant eigenvalue, with the raw distance alternating near 2. Right: the residual of plain power iteration stalls when +1 and -1 tie, while squared power iteration with sign separation drives it to zero geometrically." loading="lazy" />
  <figcaption>Power iteration on two symmetric matrices of size $d=60$ with random eigenvectors. Each curve is the median over 200 Gaussian initializations, and its band contains the middle half of the runs (the interquartile range). (a) The dominant eigenvalue $\lambda_\star=-1$ has multiplicity 3 and the next largest magnitude is $\rho_2=0.8$. The sign-corrected error $\|(-1)^k\mathbf v_k-\mathbf q_\star\|$ (<span style="color:#2563eb">blue</span>) decays like $0.8^k$ and the Rayleigh-quotient error (<span style="color:#ea580c">orange</span>) like $0.8^{2k}$, below the bounds of <a href="#remark-3-rate-of-convergence">Remark 3</a> (<span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>), while the raw distance $\|\mathbf v_k-\mathbf q_\star\|$ (<span style="color:#7d8783">grey</span>) alternates between those small values and nearly 2 (<a href="#remark-2-a-negative-dominant-eigenvalue">Remark 2</a>). (b) The eigenvalues $+1$ and $-1$ tie for the largest magnitude. The residual $\|\mathbf A\mathbf v-(\mathbf v^\top\mathbf A\mathbf v)\mathbf v\|$ of plain power iteration (<span style="color:#ea580c">orange</span>) stalls at order one (<a href="#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude">Remark 5</a>), whereas squared power iteration with sign separation (<span style="color:#2563eb">blue</span>; <a href="#34-proposition-2-power-iteration-with-sign-separation">Proposition 2</a>, run as in <a href="#37-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a>) drives the residual of its candidate to zero, by a factor of about $0.8$ per product. Both horizontal axes count matrix–vector products: $\mathbf v_k$ costs $k$ products and its Rayleigh quotient one more, and each squared step costs two.</figcaption>
</figure>

### 3.2 Corollary 1 [Recovering algebraic extrema by shifting]

Let $\lambda_{\min}=\min\Lambda(\mathbf{A})$ and $\lambda_{\max}=\max\Lambda(\mathbf{A})$.

If $\alpha\ge\lambda_{\max}$ and $\alpha>\lambda_{\min}$, then power iteration applied to $\mathbf{B}=\alpha\mathbb{I}_d-\mathbf{A}$ converges to a unit vector in $E_{\lambda_{\min}}$, provided its initialization has a nonzero projection onto that eigenspace. Its Rayleigh quotient converges to $\mu_{\max}=\alpha-\lambda_{\min}$, so $\lambda_{\min}=\alpha-\mu_{\max}$.

Similarly, if $\beta\le\lambda_{\min}$ and $\beta<\lambda_{\max}$, then power iteration applied to $\mathbf{C}=\mathbf{A}-\beta\mathbb{I}_d$ converges to a unit vector in $E_{\lambda_{\max}}$, provided its initialization has a nonzero projection onto that eigenspace. Its Rayleigh quotient converges to $\nu_{\max}=\lambda_{\max}-\beta$, so $\lambda_{\max}=\beta+\nu_{\max}$.

#### Remark 6 [What a shift does to the spectrum]

Power iteration returns the eigenvalue of largest magnitude, which is $\lambda_{\max}$ or $\lambda_{\min}$ depending on the signs. A scalar shift moves the whole spectrum without changing the eigenvectors, and the shift $\alpha\mathbb{I}_d-\mathbf{A}$ also reverses its order. With $\alpha\ge\lambda_{\max}$, the spectrum of $\mathbf{B}$ lies in $[0,\alpha-\lambda_{\min}]$, so the eigenvalue of $\mathbf{A}$ farthest below $\alpha$ becomes the dominant eigenvalue of $\mathbf{B}$.

#### Proof

For any eigenpair $(\lambda,\mathbf{u})$ of $\mathbf{A}$,

$$
\mathbf{B}\mathbf{u}=(\alpha\mathbb{I}_d-\mathbf{A})\mathbf{u}=(\alpha-\lambda)\mathbf{u}.
$$

Since an orthonormal eigenbasis of $\mathbf{A}$ is also an eigenbasis of $\mathbf{B}$,

$$
\Lambda(\mathbf{B})=\{\alpha-\lambda:\lambda\in\Lambda(\mathbf{A})\}.
$$

The assumptions on $\alpha$ give

$$
0\le\alpha-\lambda\le\textcolor{#2563eb}{\alpha-\lambda_{\min}},
\qquad
\textcolor{#2563eb}{\alpha-\lambda_{\min}}>0 .
$$

Whenever $\lambda>\lambda_{\min}$,

$$
\alpha-\lambda<\alpha-\lambda_{\min}.
$$

Therefore, $\mathbf{B}$ has a positive dominant eigenvalue $\textcolor{#2563eb}{\alpha-\lambda_{\min}}$, whose eigenspace is exactly $E_{\lambda_{\min}}$. [Proposition 1](#31-proposition-1-convergence-of-power-iteration) applies, including when this eigenvalue is repeated.

For $\mathbf{C}$,

$$
\mathbf{C}\mathbf{u}=(\mathbf{A}-\beta\mathbb{I}_d)\mathbf{u}=(\lambda-\beta)\mathbf{u},
\qquad
\Lambda(\mathbf{C})=\{\lambda-\beta:\lambda\in\Lambda(\mathbf{A})\}.
$$

The assumptions on $\beta$ give

$$
0\le\lambda-\beta\le\textcolor{#2563eb}{\lambda_{\max}-\beta},
\qquad
\textcolor{#2563eb}{\lambda_{\max}-\beta}>0 .
$$

Whenever $\lambda<\lambda_{\max}$,

$$
\lambda-\beta<\lambda_{\max}-\beta .
$$

Thus $\mathbf{C}$ has a positive dominant eigenvalue $\textcolor{#2563eb}{\lambda_{\max}-\beta}$, with eigenspace $E_{\lambda_{\max}}$. [Proposition 1](#31-proposition-1-convergence-of-power-iteration) again applies.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

In particular, choosing $\alpha=\lambda_{\max}$ recovers the minimum eigenvalue from the maximum, unless $\mathbf{A}=\lambda_{\max}\mathbb{I}_d$. In that exceptional case, every eigenvalue already equals $\lambda_{\max}$. The analogous statement holds for $\beta=\lambda_{\min}$.

#### Remark 7 [The shift only needs to pass the midpoint]

The condition $\alpha\ge\lambda_{\max}$ is convenient but stronger than needed. [Proposition 1](#31-proposition-1-convergence-of-power-iteration) only requires that $\alpha-\lambda_{\min}$ strictly dominate $|\alpha-\lambda|$ for every other eigenvalue $\lambda$ present in the initialization. For $\lambda\le\alpha$, this holds because $\lambda>\lambda_{\min}$. For $\lambda>\alpha$, it reads $\lambda-\alpha<\alpha-\lambda_{\min}$, that is, $\lambda+\lambda_{\min}<2\alpha$. Hence every shift $\alpha>(\lambda_{\min}+\lambda_{\max})/2$ works. In particular, an underestimate of $\lambda_{\max}$, such as a Rayleigh quotient, is a valid shift as long as it exceeds the midpoint of the spectrum. Symmetrically, every $\beta<(\lambda_{\min}+\lambda_{\max})/2$ works for $\mathbf{C}$.

A scalar shift changes every eigenvalue. It does not generally make the next eigenvalue in a prescribed ordering dominant. To remove specified eigendirections while preserving the other eigenpairs, we instead use orthogonal deflation.

### 3.3 Lemma 1 [Orthogonal deflation]

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

1. $\mathbf{B}\mathbf{q}_i=0$ for each $i\in\{1,\dots,r\}$. These are precisely the specified eigenvectors stored as columns of $\mathbf{Q}$.
2. $\mathbf{B}\mathbf{x}=\mathbf{A}\mathbf{x}$ for every $\mathbf{x}$ satisfying $\mathbf{Q}^\top\mathbf{x}=0$.
3. The restriction of $\mathbf{B}$ to $\operatorname{range}(\mathbf{P})=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_r\}^\perp$ has the remaining eigenvalues of $\mathbf{A}$, counted with multiplicity.

#### Remark 8 [Two views of the deflated matrix]

The matrix $\mathbf{P}\mathbf{A}\mathbf{P}$ compresses $\mathbf{A}$ onto the orthogonal complement of the stored eigenvectors. Nothing changes on that complement (item 2), and the stored directions are sent to zero (item 1). The second expression, $\mathbf{A}-\mathbf{Q}\mathbf{D}\mathbf{Q}^\top$, subtracts the part of the spectral decomposition carried by the stored eigenpairs. The two expressions coincide because the $\mathbf{q}_i$ are exact eigenvectors: the proof uses $\mathbf{A}\mathbf{Q}=\mathbf{Q}\mathbf{D}$. For computed, approximate eigenvectors they differ, which is why the implementation below applies $\mathbf{P}$ explicitly.

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

Ordinary power iteration can be applied repeatedly to the deflated operator whenever the hypotheses of [Proposition 1](#31-proposition-1-convergence-of-power-iteration) hold at each stage. The following modification also handles opposite eigenvalues of equal magnitude, without requiring a spectral bound.

### 3.4 Proposition 2 [Power iteration with sign separation]

Let $\mathbf{B}\in\mathbb{R}^{d\times d}$ be symmetric, and let $\mathbf{w}_0$ be any unit vector. If $\mathbf{B}\mathbf{w}_0=0$, then $(0,\mathbf{w}_0)$ is an eigenpair. Otherwise, the iteration

$$
\mathbf{w}_{k+1}=\frac{\mathbf{B}^2\mathbf{w}_k}{\|\mathbf{B}^2\mathbf{w}_k\|}
$$

is well defined and converges to a unit vector $\mathbf{w}$ satisfying

$$
\mathbf{B}^2\mathbf{w}=\rho^2\mathbf{w}
$$

for some $\rho>0$. This value can be recovered as

$$
\rho=\|\mathbf{B}\mathbf{w}\|.
$$

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

Squaring maps $+\rho$ and $-\rho$ to the same eigenvalue $\rho^2$ of $\mathbf{B}^2$, so the obstruction of [Remark 5](#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude) disappears: $\mathbf{B}^2$ is positive semidefinite, and its largest eigenvalue present in $\mathbf{w}_0$ is strictly dominant. The price is that the limit $\mathbf{w}$ may mix the two eigenspaces of $\mathbf{B}$ for $+\rho$ and $-\rho$. One more product separates them. The limit lies in the eigenspace of $\mathbf{B}^2$ for $\rho^2$, which is the sum of these two eigenspaces, so we can write $\mathbf{w}=\textcolor{#2563eb}{\mathbf{w}_+}+\textcolor{#ea580c}{\mathbf{w}_-}$ with $\mathbf{B}\mathbf{w}_\pm=\pm\rho\,\mathbf{w}_\pm$. Then $\mathbf{B}\mathbf{w}/\rho=\textcolor{#2563eb}{\mathbf{w}_+}-\textcolor{#ea580c}{\mathbf{w}_-}$ is the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="sign-reflection" aria-label="Explain the reflection picture of sign separation">reflection</button><span id="sign-reflection" class="explanation-popover" popover="auto" role="note" aria-label="Sign separation as a reflection" data-label="Geometry">On the sum of the two eigenspaces, $\mathbf B/\rho$ fixes every vector of the $+\rho$ eigenspace and reverses every vector of the $-\rho$ eigenspace: it is the mirror reflection across the $+\rho$ eigenspace. Averaging a vector with its mirror image keeps its part on the mirror; averaging it with minus its mirror image keeps the other part.</span></span> of $\mathbf{w}$ that fixes the <span style="color:#2563eb">blue part</span> and reverses the <span style="color:#ea580c">orange part</span>, so

$$
\mathbf{u}_+=\textcolor{#2563eb}{\mathbf{w}_+},
\qquad
\mathbf{u}_-=\textcolor{#ea580c}{\mathbf{w}_-}
$$

are exactly the orthogonal projections of $\mathbf{w}$ onto the two eigenspaces. The orthogonality and norm identities in the proof below are this observation written out.

#### Proof

Let $\Pi^{\mathbf{B}}_\lambda$ denote the orthogonal projection onto the eigenspace of $\mathbf{B}$ associated with $\lambda$. Since $\mathbf{B}\mathbf{w}_0\neq0$, the initialization has a nonzero component in at least one nonzero eigenspace of $\mathbf{B}$. Define

$$
\rho=\max\bigl\{|\lambda|:\lambda\in\Lambda(\mathbf{B}),\;\Pi^{\mathbf{B}}_\lambda\mathbf{w}_0\neq0\bigr\}>0 .
$$

The eigenvalues of $\mathbf{B}^2$ are the numbers $\lambda^2$. Its largest eigenvalue present in $\mathbf{w}_0$ is therefore $\rho^2>0$, and every other distinct eigenvalue present is strictly smaller than $\rho^2$.

[Proposition 1](#31-proposition-1-convergence-of-power-iteration), applied to $\mathbf{B}^2$, shows that the iterates are well defined and converge to a unit vector $\mathbf{w}$ in the eigenspace associated with $\rho^2$:

$$
\mathbf{B}^2\mathbf{w}=\rho^2\mathbf{w}.
$$

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

$$
\textcolor{#2563eb}{\mathbf{u}_+}+\textcolor{#ea580c}{\mathbf{u}_-}=\mathbf{w}\neq0,
$$

so the two vectors cannot both be zero. More precisely,

$$
\textcolor{#2563eb}{\mathbf{u}_+}^\top\textcolor{#ea580c}{\mathbf{u}_-}=\frac14\left(\|\mathbf{w}\|^2-\frac{\|\mathbf{B}\mathbf{w}\|^2}{\rho^2}\right)=0,
$$

and

$$
\|\textcolor{#2563eb}{\mathbf{u}_+}\|^2+\|\textcolor{#ea580c}{\mathbf{u}_-}\|^2=1 .
$$

Therefore,

$$
\max\bigl\{\|\mathbf{u}_+\|,\|\mathbf{u}_-\|\bigr\}\ge\frac{1}{\sqrt{2}} .
$$

Choosing the vector with larger norm avoids normalizing a potentially very small component.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

This construction does not require forming $\mathbf{B}^2$. Its action is obtained through $\mathbf{B}^2\mathbf{x}=\mathbf{B}(\mathbf{B}\mathbf{x})$, using two applications of $\mathbf{B}$. Squaring removes the distinction between the eigenvalues $+\rho$ and $-\rho$ during power iteration. The subsequent construction of $\mathbf{u}_+$ and $\mathbf{u}_-$ separates their eigenspaces.

### 3.5 Corollary 2 [Recovery of a complete eigenbasis]

For every real symmetric matrix $\mathbf{A}$, orthogonal deflation combined with [Proposition 2](#34-proposition-2-power-iteration-with-sign-separation) recovers a complete orthonormal eigenbasis, including repeated and zero eigenvalues, when each inner power iteration is taken to its limit. The procedure is as follows.

1. Initialize $\mathbf{Q}_0\in\mathbb{R}^{d\times0}$ as an empty matrix, that is, a matrix with no columns, so that $\mathbf{Q}_0\mathbf{Q}_0^\top=0_{d\times d}$ (each entry is an empty sum).
2. At stage $j\in\{1,\dots,d\}$, define $\mathbf{P}_{j-1}=\mathbb{I}_d-\mathbf{Q}_{j-1}\mathbf{Q}_{j-1}^\top$ and $\mathbf{B}_j=\mathbf{P}_{j-1}\mathbf{A}\mathbf{P}_{j-1}$. Choose $\mathbf{z}$ such that $\mathbf{P}_{j-1}\mathbf{z}\neq0$, and set $\mathbf{w}_0=\mathbf{P}_{j-1}\mathbf{z}/\|\mathbf{P}_{j-1}\mathbf{z}\|$. For example, a fresh draw $\mathbf{z}\sim\mathcal{N}(0,\mathbb{I}_d)$ gives a nonzero denominator with probability one.
3. If $\mathbf{B}_j\mathbf{w}_0=0$, set $\mathbf{q}_j=\mathbf{w}_0$ and $\lambda_j=0$. Otherwise, apply [Proposition 2](#34-proposition-2-power-iteration-with-sign-separation) to $\mathbf{B}_j$, choose the larger of the two sign-separated components in norm, and normalize it to obtain $\mathbf{q}_j$. Recover the eigenvalue as $\lambda_j=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$.
4. Append the new eigenvector, $\mathbf{Q}_j=[\mathbf{Q}_{j-1},\mathbf{q}_j]$, and continue to the next stage.

#### Proof

At the first stage, $\mathbf{P}_0=\mathbb{I}_d$ and $\mathbf{B}_1=\mathbf{A}$, so [Proposition 2](#34-proposition-2-power-iteration-with-sign-separation) yields an eigenpair of $\mathbf{A}$.

Suppose that the first $j-1$ stages have produced orthonormal eigenvectors of $\mathbf{A}$. By [Lemma 1](#33-lemma-1-orthogonal-deflation),

$$
\mathbf{B}_j\mathbf{x}=\mathbf{A}\mathbf{x}
\qquad
\text{whenever }\mathbf{Q}_{j-1}^\top\mathbf{x}=0 .
$$

The initialization belongs to this orthogonal complement. Since $\mathbf{B}_j=\mathbf{P}_{j-1}\mathbf{A}\mathbf{P}_{j-1}$, every squared-power iterate also belongs to it. The limit vector and the sign-separated vectors therefore belong to it as well.

Consequently, the resulting unit vector $\mathbf{q}_j$ satisfies

$$
\mathbf{A}\mathbf{q}_j=\mathbf{B}_j\mathbf{q}_j=\lambda_j\mathbf{q}_j,
\qquad
\mathbf{Q}_{j-1}^\top\mathbf{q}_j=0 .
$$

The zero case is also valid: if $\mathbf{B}_j\mathbf{w}_0=0$, then

$$
\mathbf{A}\mathbf{w}_0=\mathbf{B}_j\mathbf{w}_0=0 .
$$

Thus each stage adds one unit eigenvector orthogonal to all previously stored eigenvectors.

After $d$ stages,

$$
\mathbf{Q}_d^\top\mathbf{Q}_d=\mathbb{I}_d,
\qquad
\mathbf{A}\mathbf{Q}_d=\mathbf{Q}_d\operatorname{diag}(\lambda_1,\dots,\lambda_d).
$$

Since the square matrix $\mathbf{Q}_d$ is orthogonal, $\mathbf{Q}_d\mathbf{Q}_d^\top=\mathbb{I}_d$, and hence

$$
\mathbf{A}=\mathbf{Q}_d\operatorname{diag}(\lambda_1,\dots,\lambda_d)\mathbf{Q}_d^\top .
$$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

Any admissible nonzero initialization suffices to recover some eigenpair at each stage. With fresh Gaussian initializations, the procedure selects an eigenvalue of largest remaining absolute value with probability one. Consequently, the eigenvalues are obtained in nonincreasing absolute value, with arbitrary ordering within ties. Algebraic ordering can be obtained by sorting afterward.

The exact test $\mathbf{B}_j\mathbf{w}_0=0$ identifies a zero eigenvector. It does not require concluding that the entire remaining operator is zero.

The [figure below](#figure-complete-eigenbasis) runs this procedure on the adjacency matrix of a bipartite graph, whose nonzero eigenvalues come in opposite pairs and whose eigenvalue $0$ is repeated, and compares it with deflation combined with ordinary power iteration, with and without a shift.

<figure class="article-figure article-figure-wide" id="figure-complete-eigenbasis">
  <img src="/images/blog/matrix-vector-products/complete-eigenbasis.svg" width="1400" height="460" alt="Three panels on the adjacency matrix of a bipartite graph with 50 nodes. Left: the eigenvalue found at each of the 50 deflation stages; Algorithm 1 alternates between opposite eigenvalues of decreasing magnitude and ends with ten zeros, while shifted deflation descends from the largest to the smallest eigenvalue. Middle: eigenpairs found against matrix-vector products on a log scale; Algorithm 1 finishes after about fifteen thousand products, shifted deflation after about sixty thousand, and plain deflation never finds one. Right: residuals, eigenvalue errors and inner products of Algorithm 1, all below one trillionth." loading="lazy" />
  <figcaption>A complete eigenbasis of the adjacency matrix of a random bipartite graph, applied through its edge list (<a href="#22-other-matrices-known-through-their-products">Section 2.2</a>). The graph links 30 nodes to 20 others by 148 edges, with two planted communities on each side. Because the graph is bipartite, every nonzero eigenvalue $\sigma$ comes with $-\sigma$, so plain power iteration meets the tie of <a href="#remark-5-failure-for-opposite-eigenvalues-of-equal-dominant-magnitude">Remark 5</a> at once, and $0$ is an eigenvalue of multiplicity 10. Every stage starts from a fresh Gaussian vector and, as in <a href="#35-corollary-2-recovery-of-a-complete-eigenbasis">Corollary 2</a>, is taken to its limit: it runs until the residual of its candidate for the deflated matrix $\mathbf B_j$ is below $10^{-13}$. (a) Eigenvalue found at each stage. <a href="#37-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> (<span style="color:#2563eb">blue</span>) finds the eigenvalues in nonincreasing absolute value, the two members of each pair at consecutive stages, and the ten zero eigenvalues last. Power iteration on $\mathbf A-\lambda_{\min}(\mathbf A)\,\mathbb I_d$ with deflation (<span style="color:#7c3aed">purple</span>; <a href="#32-corollary-1-recovering-algebraic-extrema-by-shifting">Corollary 1</a> and <a href="#33-lemma-1-orthogonal-deflation">Lemma 1</a>) finds them in decreasing order. <span style="color:#7d8783">Grey lines</span> mark the 41 distinct eigenvalues. (b) Number of eigenpairs found against the number of products. Algorithm 1 uses 15,296 products in all, two for each zero eigenvalue. The shifted iteration uses 61,453, even with the most favorable shift $\lambda_{\min}(\mathbf A)=-6.48$: adding $6.48$ to every eigenvalue brings the ratios that set the speed of convergence (<a href="#remark-3-rate-of-convergence">Remark 3</a>) closer to one, and each zero eigenvalue takes a full stage. Plain deflated power iteration (<span style="color:#ea580c">orange</span>) never completes its first stage, and after 20,000 products its residual is still above 1.9. (c) Accuracy of the eigenbasis of Algorithm 1, far below the tolerance $10^{-10}\|\mathbf A\|$ (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>): residuals (<span style="color:#2563eb">blue</span>) below $3.3\cdot10^{-13}$, eigenvalues within $6.3\cdot10^{-15}$ of the spectrum (<span style="color:#7c3aed">purple</span>; exact zeros drawn at $10^{-17}$), inner products between distinct eigenvectors (<span style="color:#7d8783">grey</span>) below $7.3\cdot10^{-14}$, and $\|\mathbf A-\mathbf Q_{50}\operatorname{diag}(\lambda_1,\dots,\lambda_{50})\mathbf Q_{50}^\top\|=2.2\cdot10^{-13}$.</figcaption>
</figure>

### 3.6 Lemma 2 [Residual certificate for an approximate eigenpair]

For any unit vector $\mathbf{q}$, define

$$
\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q},
\qquad
\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}.
$$

Then

$$
\min_{\lambda\in\Lambda(\mathbf{A})}|\widehat{\lambda}-\lambda|\le\|\mathbf{r}\| .
$$

#### Remark 10 [A computable bound on an unknown distance]

The residual is computable with one product, and it bounds an unknown quantity: the distance from the estimate $\widehat\lambda$ to the nearest true eigenvalue. If $\mathbf{q}$ is exactly an eigenvector, the residual is zero, and the bound says that $\widehat\lambda$ is exactly an eigenvalue.

#### Proof

Expand $\mathbf{q}$ in an orthonormal eigenbasis $\mathbf{u}_1,\dots,\mathbf{u}_d$ of $\mathbf{A}$:

$$
\mathbf{q}=\sum_{i=1}^dc_i\mathbf{u}_i,
\qquad
\mathbf{A}\mathbf{u}_i=\lambda_i\mathbf{u}_i,
\qquad
\sum_{i=1}^dc_i^2=1 .
$$

Then

$$
\mathbf{r}=\sum_{i=1}^dc_i(\lambda_i-\widehat{\lambda})\mathbf{u}_i,
$$

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

This certificate concerns proximity to some eigenvalue. A small residual alone does not certify that the eigenvalue has the largest remaining absolute value.

#### Remark 11 [Any scalar, and the best one]

The proof never uses the definition of $\widehat\lambda$. The same expansion gives

$$
\min_{\lambda\in\Lambda(\mathbf{A})}|\mu-\lambda|\le\|\mathbf{A}\mathbf{q}-\mu\mathbf{q}\|
\qquad\text{for every scalar }\mu,
$$

so the certificate applies to an eigenvalue estimate computed by any means, as long as it is paired with a unit vector. The Rayleigh quotient is the best choice of $\mu$. Indeed, expanding the square with $\|\mathbf{q}\|=1$ and $\widehat\lambda=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ gives

$$
\begin{aligned}
\|\mathbf{A}\mathbf{q}-\mu\mathbf{q}\|^2
&=\|\mathbf{A}\mathbf{q}\|^2-2\mu\widehat\lambda+\mu^2\\
&=\textcolor{#ea580c}{(\mu-\widehat\lambda)^2}+\textcolor{#2563eb}{\|\mathbf{A}\mathbf{q}\|^2-\widehat\lambda^2}\\
&=\textcolor{#ea580c}{(\mu-\widehat\lambda)^2}+\textcolor{#2563eb}{\|\mathbf{r}\|^2},
\end{aligned}
$$

where the last line is the second one at $\mu=\widehat\lambda$. The <span style="color:#2563eb">blue term</span> does not depend on $\mu$, so the certificate is smallest at $\mu=\widehat\lambda$, where it equals the residual of [Lemma 2](#36-lemma-2-residual-certificate-for-an-approximate-eigenpair), and the <span style="color:#ea580c">orange term</span> is the price of any other estimate. As a function of $\mu$, the certificate $\sqrt{(\mu-\widehat\lambda)^2+\|\mathbf{r}\|^2}$ is a hyperbola centered at the Rayleigh quotient, and it lies above the distance from $\mu$ to the spectrum, as panel (a) of the [figure below](#figure-residual-certificate) shows.

<figure class="article-figure article-figure-wide" id="figure-residual-certificate">
  <img src="/images/blog/matrix-vector-products/residual-certificate.svg" width="1400" height="460" alt="Three panels on a 60 by 60 sample covariance matrix. Left: for several power-iteration iterates, the norm of A v minus mu v as a function of mu is a hyperbola lying above the saw-tooth distance from mu to the spectrum, with its minimum at the Rayleigh quotient. Middle: the residual decreases geometrically and the distance to the nearest eigenvalue twice as fast, down to rounding level. Right: a log-log scatter of distance against residual for many iterates, all below the diagonal." loading="lazy" />
  <figcaption>The residual certificate on the sample covariance matrix $\frac1n\mathbf X^\top\mathbf X$ of $n=400$ centered observations of $d=60$ variables, applied as $\mathbf X^\top(\mathbf X\mathbf v)/n$ (<a href="#22-other-matrices-known-through-their-products">Section 2.2</a>). Its three largest eigenvalues, the variances along the first three principal components, are 9.72, 4.92 and 3.20, and the other 57 lie in $[0.40,1.80]$. (a) For the iterates $\mathbf v_k$ of power iteration from one Gaussian start, the certificate $\mu\mapsto\|\mathbf A\mathbf v_k-\mu\mathbf v_k\|$ of <a href="#remark-11-any-scalar-and-the-best-one">Remark 11</a> (<span style="background:linear-gradient(90deg,#b0dd2f,#481d6f);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span> as $k$ grows) lies above the distance from $\mu$ to the spectrum (<span style="color:#27322f">black</span>, with the eigenvalues marked in <span style="color:#ea580c">orange</span> on the axis) for every scalar $\mu$. Its minimum, marked by a dot, is attained at the Rayleigh quotient $\widehat\lambda_k$ and equals the residual $\|\mathbf r_k\|$ of <a href="#36-lemma-2-residual-certificate-for-an-approximate-eigenpair">Lemma 2</a>. (b) Medians over 200 Gaussian starts, with bands containing the middle half of the runs. The residual (<span style="color:#2563eb">blue</span>) shrinks by the factor $\lambda_2/\lambda_1\approx0.51$ per product, and the distance from $\widehat\lambda_k$ to the nearest eigenvalue (<span style="color:#ea580c">orange</span>) by its square (<a href="#remark-3-rate-of-convergence">Remark 3</a>), until rounding errors of order $10^{-15}$ stop it. The certificate is valid but pessimistic: after 20 products, the median residual is $1.2\cdot10^{-5}$ and the median distance $3.2\cdot10^{-11}$. In the first steps, $\widehat\lambda_k$ lies among the small eigenvalues, and its distance to the nearest of them (<span style="color:#ea580c">orange</span>) is much smaller than its distance to $\lambda_{\max}(\mathbf A)$ (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dotted;text-decoration-thickness:0.12em;text-underline-offset:0.2em">dotted</span>): the certificate concerns some eigenvalue, not necessarily the largest. (c) The pairs formed by the residual and the distance for 50 runs, colored by $k$ from <span style="background:linear-gradient(90deg,#fde725,#440154);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span>. None lies above the diagonal, as <a href="#36-lemma-2-residual-certificate-for-an-approximate-eigenpair">Lemma 2</a> requires, and over all 200 runs and 40 steps the largest ratio of the distance to the residual is 0.995. Distances below $10^{-15}$, at rounding level, are drawn at $10^{-15}$.</figcaption>
</figure>

### 3.7 Algorithm 1 [Deflated squared power iteration with sign separation]

For a finite implementation, squared power iteration and sign separation can be combined without explicitly constructing $\mathbf{A}$, $\mathbf{P}$ or $\mathbf{B}$. At a given deflation stage, let $\mathbf{Q}$ contain the previously computed orthonormal vectors, and implement the projection through

$$
\mathbf{P}\mathbf{x}=\mathbf{x}-\mathbf{Q}(\mathbf{Q}^\top\mathbf{x}).
$$

Choose tolerances $\varepsilon_{\mathrm{abs}}>0$ and $\varepsilon_{\mathrm{rel}}\ge0$. Starting from a unit vector $\mathbf{w}$ satisfying $\mathbf{Q}^\top\mathbf{w}=0$, repeat the following steps.

1. **<span style="color:#2563eb">First product.</span>** Compute $\textcolor{#2563eb}{\mathbf{h}_1}=\mathbf{A}\mathbf{w}$, $\mathbf{a}=\mathbf{P}\textcolor{#2563eb}{\mathbf{h}_1}$ and $\rho=\|\mathbf{a}\|$. If $\rho=0$, go to step 5.
2. **<span style="color:#ea580c">Second product.</span>** Set $\mathbf{y}=\mathbf{a}/\rho$, then compute $\textcolor{#ea580c}{\mathbf{h}_2}=\mathbf{A}\mathbf{y}$ and $\mathbf{b}=\mathbf{P}\textcolor{#ea580c}{\mathbf{h}_2}$.
3. **Candidate.** Choose the scalar sign $s=+1$ if $\mathbf{w}^\top\mathbf{y}\ge0$ and $s=-1$ if $\mathbf{w}^\top\mathbf{y}<0$. This selects whichever of $\mathbf{w}+\mathbf{y}$ and $\mathbf{w}-\mathbf{y}$ has larger norm. Form $n=\|\mathbf{w}+s\mathbf{y}\|$ and $\mathbf{q}=(\mathbf{w}+s\mathbf{y})/n$. The product of the candidate with the original matrix is available by linearity: $\mathbf{A}\mathbf{q}=(\textcolor{#2563eb}{\mathbf{h}_1}+s\textcolor{#ea580c}{\mathbf{h}_2})/n$.
4. **Check or update.** Compute $\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ and $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$. If $\|\mathbf{r}\|\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$, accept $(\widehat{\lambda},\mathbf{q})$ and append $\mathbf{q}$ to $\mathbf{Q}$. Otherwise, update $\mathbf{w}\leftarrow\mathbf{b}/\|\mathbf{b}\|$ and return to step 1.
5. **Zero projected product.** If $\rho=0$, use $\mathbf{q}=\mathbf{w}$ as the candidate, compute $\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$ and $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$, where $\mathbf{A}\mathbf{q}=\textcolor{#2563eb}{\mathbf{h}_1}$, and check the same original residual before accepting the pair.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

Because $\mathbf{w}$ and $\mathbf{y}$ are unit vectors, the normalization in step 3 is never small:

$$
n^2=\|\mathbf{w}\|^2+\|\mathbf{y}\|^2+2s\,\mathbf{w}^\top\mathbf{y}=2+2|\mathbf{w}^\top\mathbf{y}|\ge2 .
$$

For the symmetric projected operator $\mathbf{B}=\mathbf{P}\mathbf{A}\mathbf{P}$, these operations satisfy

$$
\mathbf{a}=\mathbf{B}\mathbf{w},
\qquad
\mathbf{y}=\frac{\mathbf{B}\mathbf{w}}{\rho},
\qquad
\mathbf{b}=\frac{\mathbf{B}^2\mathbf{w}}{\rho},
$$

because $\mathbf{P}\mathbf{w}=\mathbf{w}$ and $\mathbf{P}\mathbf{y}=\mathbf{y}$. Thus the update is normalized power iteration on $\mathbf{B}^2$, with a normalized intermediate vector. In exact arithmetic, $\rho>0$ implies $\mathbf{b}\neq0$, because

$$
\mathbf{w}^\top\mathbf{B}^2\mathbf{w}=\|\mathbf{B}\mathbf{w}\|^2=\rho^2>0 .
$$

The sign separation is meaningful before exact convergence. For a unit vector $\mathbf{w}$ with

$$
\rho=\|\mathbf{B}\mathbf{w}\|>0,
\qquad
n=\left\|\mathbf{w}+s\frac{\mathbf{B}\mathbf{w}}{\rho}\right\|,
\qquad
\mathbf{q}=\frac{\mathbf{w}+s\mathbf{B}\mathbf{w}/\rho}{n},
$$

direct calculation gives

$$
\begin{aligned}
(\mathbf{B}-s\rho\,\mathbb{I}_d)\mathbf{q}
&=\frac{\textcolor{#2563eb}{\mathbf{B}\mathbf{w}}+s\mathbf{B}^2\mathbf{w}/\rho-s\rho\,\mathbf{w}\textcolor{#2563eb}{-\mathbf{B}\mathbf{w}}}{n}\\
&=\frac{s\bigl(\mathbf{B}^2\mathbf{w}-\rho^2\mathbf{w}\bigr)}{\rho\,n}.
\end{aligned}
$$

The <span style="color:#2563eb">blue terms</span> cancel because $s^2=1$. Since $n\ge\sqrt2$, the eigenpair residual of the candidate for $\mathbf{B}$ is at most $\|\mathbf{B}^2\mathbf{w}-\rho^2\mathbf{w}\|/(\sqrt2\,\rho)$. Thus, as squared power iteration converges, the extracted candidate becomes an eigenvector of $\mathbf{B}$. The stopping criterion nevertheless uses the original residual $\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$, because numerically computed deflation vectors need not span an exactly invariant subspace of $\mathbf{A}$.

In step 5, a small computed value of $\rho$ does not establish that every remaining eigenvalue is zero. This is why the zero case is also accepted only through the original residual.

In floating-point arithmetic, the implementation should reorthogonalize projected vectors, impose a maximum number of inner iterations, and report nonconvergence if the original residual does not reach the requested tolerance. No vector with computed norm zero should be normalized. A numerical breakdown should instead trigger a residual check, a restart, or a report of nonconvergence.

The exact recovery guarantee of [Corollary 2](#35-corollary-2-recovery-of-a-complete-eigenbasis) concerns limiting iterates and exact deflation. With approximate deflation vectors, convergence for $\mathbf{P}\mathbf{A}\mathbf{P}$ need not drive the residual for the original matrix $\mathbf{A}$ below an arbitrarily small tolerance. Finite computations therefore return approximate eigenpairs whose original residuals must be checked.

The residual for $\mathbf{A}$ splits exactly into a part that the inner iteration reduces and a part that it inherits. Let $\widehat\lambda_1,\dots,\widehat\lambda_r$ be the eigenvalue estimates of the stored pairs, and let the columns of $\mathbf{R}=\mathbf{A}\mathbf{Q}-\mathbf{Q}\operatorname{diag}(\widehat\lambda_1,\dots,\widehat\lambda_r)$ be their residuals. For a unit candidate $\mathbf{q}$ with $\mathbf{Q}^\top\mathbf{q}=0$ and $\widehat\lambda=\mathbf{q}^\top\mathbf{A}\mathbf{q}$, we have $\mathbf{P}\mathbf{q}=\mathbf{q}$, hence $\mathbf{A}\mathbf{q}=\mathbf{B}\mathbf{q}+\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\mathbf{q}$, where $\mathbf{B}\mathbf{q}-\widehat\lambda\mathbf{q}$ lies in $\operatorname{range}(\mathbf{P})$ and $\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\mathbf{q}$ in $\operatorname{range}(\mathbf{Q})$. Since $\mathbf{Q}^\top\mathbf{q}=0$, we also have $\mathbf{Q}^\top\mathbf{A}\mathbf{q}=(\mathbf{A}\mathbf{Q})^\top\mathbf{q}=\mathbf{R}^\top\mathbf{q}$. The two parts are orthogonal, so

$$
\textcolor{#2563eb}{\|\mathbf{A}\mathbf{q}-\widehat\lambda\mathbf{q}\|^2}=\textcolor{#ea580c}{\|\mathbf{B}\mathbf{q}-\widehat\lambda\mathbf{q}\|^2}+\textcolor{#7c3aed}{\|\mathbf{R}^\top\mathbf{q}\|^2}.
$$

The inner iteration drives the <span style="color:#ea580c">orange term</span> to zero. The <span style="color:#7c3aed">purple term</span> is inherited from the stored pairs: each entry of $\mathbf{R}^\top\mathbf{q}$ is the inner product of $\mathbf{q}$ with a stored residual, so the term is at most the sum of their squared norms, but nothing in the inner iteration makes it small. A stage therefore stalls when its threshold $\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{q}\|$ is below the inherited part. This happens readily with $\varepsilon_{\mathrm{rel}}>0$, because the threshold shrinks with $|\widehat\lambda|$, whereas the inherited part comes from earlier stages, whose eigenvalues are larger in absolute value and whose accepted residuals can be nearly as large as their own thresholds. A threshold that is the same at every stage ($\varepsilon_{\mathrm{rel}}=0$) removes this systematic mismatch, but over many stages the inherited parts can still exceed it. Continuing each stage until the residual for $\mathbf{B}$ reaches rounding level, and only then checking the residual for $\mathbf{A}$, keeps every inherited part far below the threshold, at the price of more products. The [figure below](#figure-hessian-deflation) shows the stall on the Hessian of a neural network. The complete eigenbasis in the [figure of Section 3.5](#figure-complete-eigenbasis) was computed in the last way. With the same constant threshold but without the continuation, that run stalls at stage 20 of 50, where the inherited part is $6.89\cdot10^{-10}$ and the threshold $6.48\cdot10^{-10}$.

<figure class="article-figure article-figure-wide" id="figure-hessian-deflation">
  <img src="/images/blog/matrix-vector-products/hessian-deflation.svg" width="1400" height="460" alt="Three panels on the Hessian of a small neural network. Left: eight successive deflation stages, each with a residual decreasing geometrically until a constant threshold. Middle: at a stalled second stage, the residual for the deflated matrix keeps decreasing to rounding level while the residual for the original matrix levels off at the inherited part, just above the threshold. Right: the fraction of twenty runs certifying j pairs stays at one with the constant threshold and falls to 0.3 with the relative threshold." loading="lazy" />
  <figcaption><a href="#37-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> on the Hessian of a training loss, applied by automatic differentiation (<a href="#21-hessians">Section 2.1</a>). The network is a fully connected tanh network with layer widths 4, 16, 16 and 1, so that $\theta\in\mathbb R^{369}$, trained by 4,000 steps of gradient descent on $n=256$ Gaussian inputs labelled by a random teacher network, down to a squared loss of $9.4\cdot10^{-4}$. Its Hessian has five large eigenvalues, from 5.87 down to 2.79, the next ones are near 0.3, and 147 of its eigenvalues are negative but above $-0.0023$. Algorithm 1 computes the 8 eigenpairs of largest absolute value with two threshold rules: a constant one, $\varepsilon_{\mathrm{abs}}=10^{-10}\|\mathbf A\|$ and $\varepsilon_{\mathrm{rel}}=0$, where $\|\mathbf A\|=5.87$ is the absolute value of the first eigenvalue found, and a relative one, $\varepsilon_{\mathrm{abs}}=\varepsilon_{\mathrm{rel}}=10^{-10}$. (a) One run with the constant threshold: the residual of the candidate against the cumulative number of Hessian–vector products, with the stages colored from <span style="background:linear-gradient(90deg,#b0dd2f,#481d6f);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent">light to dark</span> and labelled by their eigenvalues. Each stage decreases geometrically, by a factor per product equal to the ratio of the next largest remaining absolute value to its own, until it reaches the threshold (<span style="color:#7d8783">grey</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>), and the eight stages take 1,008 products. (b) A run with the relative threshold, at its second stage. The residual for $\mathbf A$ (<span style="color:#2563eb">blue</span>), which the algorithm checks, splits into the residual for $\mathbf B$ (<span style="color:#ea580c">orange</span>, <span style="text-decoration:underline dashed;text-underline-offset:0.2em">dashed</span>), which the inner iteration drives down to $5\cdot10^{-16}$, and the inherited part $\|\mathbf Q^\top\mathbf A\mathbf q\|$ (<span style="color:#7c3aed">purple</span>, <span style="text-decoration:underline dotted;text-decoration-thickness:0.12em;text-underline-offset:0.2em">dotted</span>), which equals the residual $6.48\cdot10^{-10}$ accepted at the first stage. The threshold of the second stage, $10^{-10}(1+5.28)=6.28\cdot10^{-10}$, is just below it, so the stage never stops. (c) Fraction of 20 runs, each with its own Gaussian starting vectors, that certify the $j$ largest pairs. With the relative threshold (<span style="color:#ea580c">orange</span>), 14 runs stall: 7 at stage 2, 5 at stage 4 and 2 at stage 7. With the constant threshold (<span style="color:#2563eb">blue</span>), all 20 runs certify the 8 pairs, and in the run of panel (a) every eigenvalue is within $6\cdot10^{-15}$ of the spectrum.</figcaption>
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

Power iteration started at $\mathbf{q}_1$ keeps only the last vector $\mathbf{A}^{m-1}\mathbf{q}_1$ of this list. The Krylov subspace keeps all of them, so it contains every polynomial filter of degree at most $m-1$ applied to $\mathbf{q}_1$. By the spectral decomposition, $p(\mathbf{A})\mathbf{q}_1=\sum_\lambda p(\lambda)\Pi_\lambda\mathbf{q}_1$, so a polynomial that is large at one eigenvalue and small at the others can isolate that eigencomponent much better than the monomial $t^{m-1}$ used by power iteration.

### 4.2 How the method works

> <span class="article-kicker article-kicker--theorem">The idea</span> The Lanczos method runs the Gram–Schmidt process on the Krylov subspace
>
> $$
> \mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\bigl\{p(\mathbf{A})\mathbf{q}_1:p\text{ is a polynomial of degree at most }m-1\bigr\},
> $$
>
> one matrix–vector product per step. The numbers that Gram–Schmidt computes along the way form a small symmetric matrix, and the eigenvalues of that matrix approximate eigenvalues of $\mathbf{A}$.

This section unpacks these sentences. It first explains why the search for eigenvalues needs an orthonormal basis of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, then runs Gram–Schmidt on the Krylov subspace by hand for four steps, states the general step, and finally shows how eigenvalues and eigenvectors come out of the coefficients. The proofs are in Lemmas [3](#45-lemma-3-orthogonality-and-tridiagonal-projection) to [5](#47-lemma-5-variational-characterization-and-monotonicity) and [Proposition 3](#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run).

After $m-1$ products, every vector of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ is within reach. It is a combination $c_0\mathbf{q}_1+c_1\mathbf{A}\mathbf{q}_1+\dots+c_{m-1}\mathbf{A}^{m-1}\mathbf{q}_1=p(\mathbf{A})\mathbf{q}_1$, and by the spectral decomposition, $p(\mathbf{A})\mathbf{q}_1=\sum_{\lambda}p(\lambda)\,\Pi_\lambda\mathbf{q}_1$ reweights each eigencomponent of $\mathbf{q}_1$ by the value $p(\lambda)$. Power iteration only ever uses the polynomial $t^{m-1}$. The Lanczos method considers all of them at once. Since $\lambda_{\max}(\mathbf{A})$ and $\lambda_{\min}(\mathbf{A})$ are the largest and smallest values of the Rayleigh quotient $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ over unit vectors $\mathbf{x}$, the best estimates that $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ can provide are the largest and smallest values of $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ over its unit vectors.

Searching a subspace needs coordinates, and the convenient ones come from an orthonormal basis $\mathbf{q}_1,\dots,\mathbf{q}_m$ of $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, the columns of $\mathbf{Q}_m=[\mathbf{q}_1,\dots,\mathbf{q}_m]$. Every vector of the subspace is then $\mathbf{x}=\mathbf{Q}_m\mathbf{y}=y_1\mathbf{q}_1+\dots+y_m\mathbf{q}_m$ for a unique $\mathbf{y}\in\mathbb{R}^m$, lengths are preserved, $\|\mathbf{x}\|=\|\mathbf{y}\|$, and

$$
\mathbf{x}^\top\mathbf{A}\mathbf{x}=\mathbf{y}^\top\mathbf{T}_m\mathbf{y},
\qquad
\mathbf{T}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m\in\mathbb{R}^{m\times m}.
$$

The search over $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ thus becomes the eigenvalue problem of the $m\times m$ matrix $\mathbf{T}_m$, whose entry $\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j$ is the coordinate of $\mathbf{A}\mathbf{q}_j$ along $\mathbf{q}_i$. Gram–Schmidt applied to the spanning vectors $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$ of the Krylov subspace produces the orthonormal basis, and, as the following steps show, it computes these coordinates on the way.

**Step 1: Gram–Schmidt on $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)=\{(c_0\mathbb{I}_d+c_1\mathbf{A})\mathbf{q}_1:c_0,c_1\in\mathbb{R}\}$.** The unit vector $\mathbf{q}_1$ is the first basis vector, and it spans $\mathcal{K}_1(\mathbf{A},\mathbf{q}_1)$. The first product gives $\mathbf{A}\mathbf{q}_1$, the vector that $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$ adds. Gram–Schmidt removes its component along $\mathbf{q}_1$, whose coefficient

$$
\textcolor{#7c3aed}{\alpha_1}=\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_1
$$

is the Rayleigh quotient of $\mathbf{q}_1$. It is the first eigenvalue estimate, the same as that of power iteration. The remainder $\mathbf{r}_1=\mathbf{A}\mathbf{q}_1-\alpha_1\mathbf{q}_1$ is orthogonal to $\mathbf{q}_1$, and dividing it by its length $\textcolor{#16a34a}{\beta_1}=\|\mathbf{r}_1\|$ gives the second basis vector. Rearranged, the step reads

$$
\mathbf{A}\mathbf{q}_1=\textcolor{#7c3aed}{\alpha_1\mathbf{q}_1}+\textcolor{#16a34a}{\beta_1\mathbf{q}_2},
\qquad
\mathbf{q}_2=\frac{(\mathbf{A}-\alpha_1\mathbb{I}_d)\,\mathbf{q}_1}{\beta_1}.
$$

So $\mathbf{q}_1,\mathbf{q}_2$ is an orthonormal basis of $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$, and $\mathbf{q}_2=p_1(\mathbf{A})\mathbf{q}_1$ for the polynomial $p_1(t)=(t-\alpha_1)/\beta_1$ of degree one. In the colors used throughout this section, a <span style="color:#7c3aed">purple</span> coefficient is computed at the current step, and a <span style="color:#16a34a">green</span> one is the length of the new direction.

**Step 2: Gram–Schmidt on $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$.** The vector that $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$ adds is $\mathbf{A}^2\mathbf{q}_1$, and textbook Gram–Schmidt would orthogonalize it next. The Lanczos method applies $\mathbf{A}$ to the newest basis vector instead, which also costs one product. By Step 1,

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
\mathbf{q}_3=\frac{\mathbf{r}_2}{\beta_2},
$$

that is, $\mathbf{A}\mathbf{q}_2=\textcolor{#ea580c}{\beta_1\mathbf{q}_1}+\textcolor{#7c3aed}{\alpha_2\mathbf{q}_2}+\textcolor{#16a34a}{\beta_2\mathbf{q}_3}$. Now $\mathbf{q}_1,\mathbf{q}_2,\mathbf{q}_3$ is an orthonormal basis of $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$, and $\mathbf{q}_3=p_2(\mathbf{A})\mathbf{q}_1$ for the polynomial $p_2(t)=\bigl((t-\alpha_2)\,p_1(t)-\beta_1\bigr)/\beta_2$ of degree two. The coefficients found so far already give the matrix of the Rayleigh quotient on $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$,

$$
\mathbf{T}_2=\mathbf{Q}_2^\top\mathbf{A}\mathbf{Q}_2=\begin{bmatrix}\alpha_1&\beta_1\\\beta_1&\alpha_2\end{bmatrix},
$$

whose two eigenvalues are the smallest and largest values of $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ over the unit vectors of $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$.

**Step 3: Gram–Schmidt on $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$, where a coefficient vanishes.** The third product gives $\mathbf{A}\mathbf{q}_3$. Its coefficient on $\mathbf{q}_1$ is

$$
\mathbf{q}_1^\top\mathbf{A}\mathbf{q}_3=(\mathbf{A}\mathbf{q}_1)^\top\mathbf{q}_3=(\alpha_1\mathbf{q}_1+\beta_1\mathbf{q}_2)^\top\mathbf{q}_3=\textcolor{#7d8783}{0},
$$

because $\mathbf{A}\mathbf{q}_1$ lies in $\mathcal{K}_2(\mathbf{A},\mathbf{q}_1)$, to which $\mathbf{q}_3$ is orthogonal. Gram–Schmidt has nothing to subtract along $\mathbf{q}_1$, and <span style="color:#7d8783">grey</span> marks such vanishing coefficients. The coefficient on $\mathbf{q}_2$ is $(\mathbf{A}\mathbf{q}_2)^\top\mathbf{q}_3=\textcolor{#ea580c}{\beta_2}$, known from Step 2, and the one on $\mathbf{q}_3$ is the new number $\textcolor{#7c3aed}{\alpha_3}=\mathbf{q}_3^\top\mathbf{A}\mathbf{q}_3$. Two subtractions and a normalization give $\mathbf{q}_4$:

$$
\mathbf{A}\mathbf{q}_3=\textcolor{#ea580c}{\beta_2\mathbf{q}_2}+\textcolor{#7c3aed}{\alpha_3\mathbf{q}_3}+\textcolor{#16a34a}{\beta_3\mathbf{q}_4},
$$

and $\mathbf{q}_1,\dots,\mathbf{q}_4$ is an orthonormal basis of $\mathcal{K}_4(\mathbf{A},\mathbf{q}_1)$.

**Step 4: Gram–Schmidt on $\mathcal{K}_5(\mathbf{A},\mathbf{q}_1)$, and the pattern.** The fourth product gives $\mathbf{A}\mathbf{q}_4$. Its coefficients on $\mathbf{q}_1$ and $\mathbf{q}_2$ vanish, since $\mathbf{A}\mathbf{q}_1$ and $\mathbf{A}\mathbf{q}_2$ lie in $\mathcal{K}_3(\mathbf{A},\mathbf{q}_1)$, to which $\mathbf{q}_4$ is orthogonal. Its coefficient on $\mathbf{q}_3$ is $\textcolor{#ea580c}{\beta_3}$, known from Step 3, and the new number is $\textcolor{#7c3aed}{\alpha_4}$, so that $\mathbf{A}\mathbf{q}_4=\textcolor{#ea580c}{\beta_3\mathbf{q}_3}+\textcolor{#7c3aed}{\alpha_4\mathbf{q}_4}+\textcolor{#16a34a}{\beta_4\mathbf{q}_5}$. The four steps have computed the coordinates of $\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}\mathbf{q}_4$ in the basis:

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

**The general step.** The same reasoning holds at every step $j$. By symmetry, the coefficient of $\mathbf{A}\mathbf{q}_j$ on an earlier basis vector $\mathbf{q}_i$ is $(\mathbf{A}\mathbf{q}_i)^\top\mathbf{q}_j$, and $\mathbf{A}\mathbf{q}_i$ lies in $\mathcal{K}_{i+1}(\mathbf{A},\mathbf{q}_1)=\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{i+1}\}$, to which $\mathbf{q}_j$ is orthogonal as soon as $i\le j-2$. Only two coefficients survive: $\beta_{j-1}$, computed as a length at the previous step, and the new number $\alpha_j=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j$. Step $j$ of [Definition 2](#44-definition-2-lanczos-recurrence-and-exact-breakdown) below is therefore the following Gram–Schmidt step.

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

$$
\mathbf{A}\mathbf{q}_j=\textcolor{#ea580c}{\beta_{j-1}\mathbf{q}_{j-1}}+\textcolor{#7c3aed}{\alpha_j\mathbf{q}_j}+\textcolor{#16a34a}{\beta_j\mathbf{q}_{j+1}}.
$$

The <span style="color:#ea580c">orange term</span> was found at the previous step, the <span style="color:#7c3aed">purple term</span> is the only coefficient computed at step $j$, and the <span style="color:#16a34a">green term</span> is the new direction. In terms of polynomials, $\mathbf{q}_{j+1}=p_j(\mathbf{A})\mathbf{q}_1$, where $p_0=1$, $p_{-1}=0$ and

$$
\beta_j\,p_j(t)=(t-\alpha_j)\,p_{j-1}(t)-\beta_{j-1}\,p_{j-2}(t).
$$

Gram–Schmidt on $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ is thus Gram–Schmidt on the polynomials $1,t,t^2,\dots$ for the inner product $\langle p,r\rangle=\bigl(p(\mathbf{A})\mathbf{q}_1\bigr)^\top r(\mathbf{A})\mathbf{q}_1=\sum_\lambda p(\lambda)\,r(\lambda)\,\|\Pi_\lambda\mathbf{q}_1\|^2$, and the polynomials $p_j$ are orthonormal for it.

Compared with textbook Gram–Schmidt on $\mathbf{q}_1,\mathbf{A}\mathbf{q}_1,\mathbf{A}^2\mathbf{q}_1,\dots$, the method makes two changes, and neither changes the subspaces. First, it multiplies the newest basis vector instead of forming the next power. In general, $\mathbf{q}_j$ has a nonzero coefficient on $\mathbf{A}^{j-1}\mathbf{q}_1$, since otherwise it would lie in $\mathcal{K}_{j-1}(\mathbf{A},\mathbf{q}_1)$, to which it is orthogonal. Hence $\mathbf{A}\mathbf{q}_j$ is a nonzero multiple of $\mathbf{A}^j\mathbf{q}_1$ plus a vector of $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, and the two bring the same new direction, as in Step 2. This avoids a loss of accuracy. Up to scaling, the powers $\mathbf{A}^j\mathbf{q}_1$ are the iterates of power iteration, so they turn toward the same dominant eigenvector and become nearly parallel, and extracting a new direction from nearly parallel vectors loses accuracy to cancellation. Second, symmetry leaves two subtractions per step instead of $j$, so every step costs the same. In floating-point arithmetic, the vanishing coefficients are only approximately zero and the orthogonality they protect is gradually lost, which is why [Algorithm 2](#411-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks) below reorthogonalizes.

**Eigenvalues and eigenvectors from the coefficients.** Writing the relations $\mathbf{A}\mathbf{q}_j=\beta_{j-1}\mathbf{q}_{j-1}+\alpha_j\mathbf{q}_j+\beta_j\mathbf{q}_{j+1}$ for $j=1,\dots,m$ as the columns of one matrix identity gives

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\textcolor{#16a34a}{\beta_m\mathbf{q}_{m+1}\mathbf{e}_m^\top},
$$

where $\mathbf{e}_m$ is the last standard basis vector of $\mathbb{R}^m$. Inside $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, the matrix $\mathbf{A}$ acts on coordinates as the small matrix $\mathbf{T}_m$, except for the <span style="color:#16a34a">green term</span>, through which the last basis vector leaks out of the subspace. Every eigenpair $(\vartheta,\mathbf{y})$ of $\mathbf{T}_m$, with $\|\mathbf{y}\|=1$, therefore gives a candidate eigenpair $(\vartheta,\mathbf{Q}_m\mathbf{y})$ of $\mathbf{A}$, a Ritz pair of [Lemma 4](#46-lemma-4-ritz-pairs-and-their-residuals):

$$
\underbrace{\mathbf{A}}_{d\times d}
\;\longrightarrow\;
\underbrace{\mathbf{T}_m=\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m}_{m\times m}
\;\longrightarrow\;
(\vartheta,\mathbf{y})
\;\longrightarrow\;
(\vartheta,\mathbf{Q}_m\mathbf{y}).
$$

Multiplying the identity by $\mathbf{y}$ and using $\mathbf{T}_m\mathbf{y}=\vartheta\mathbf{y}$ shows that the residual of the Ritz pair is exactly the leak:

$$
\mathbf{A}\mathbf{Q}_m\mathbf{y}-\vartheta\,\mathbf{Q}_m\mathbf{y}=\textcolor{#16a34a}{\beta_m\,y_m\,\mathbf{q}_{m+1}},
\qquad
\bigl\|\mathbf{A}\mathbf{Q}_m\mathbf{y}-\vartheta\,\mathbf{Q}_m\mathbf{y}\bigr\|=\beta_m|y_m|,
$$

where $y_m$ is the last entry of $\mathbf{y}$. The residual is available without any further product, and [Lemma 2](#36-lemma-2-residual-certificate-for-an-approximate-eigenpair) turns it into a certificate. The basis vectors are building blocks, not approximate eigenvectors. An approximate eigenvector $\mathbf{Q}_m\mathbf{y}$ combines all of them, so the basis must be stored to form it, whereas the Ritz values need only the numbers $\alpha_j$ and $\beta_j$.

Gram–Schmidt also explains why the Ritz values approach eigenvalues. At step $m$, the part of $\mathbf{A}^m\mathbf{q}_1$ that Gram–Schmidt keeps is what remains after removing its orthogonal projection onto $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$, and it points along $\mathbf{q}_{m+1}$. Every vector $\mathbf{A}^m\mathbf{q}_1-\mathbf{k}$ with $\mathbf{k}\in\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ is $\pi(\mathbf{A})\mathbf{q}_1$ for a monic polynomial $\pi$ of degree $m$, that is, one whose coefficient of $t^m$ is $1$, and the orthogonal projection gives the closest $\mathbf{k}$. The remainder is therefore $\pi_m(\mathbf{A})\mathbf{q}_1=\beta_1\beta_2\cdots\beta_m\,\mathbf{q}_{m+1}$, where $\pi_m$ is the monic polynomial of degree $m$ that makes

$$
\|\pi(\mathbf{A})\mathbf{q}_1\|^2=\sum_{\lambda\in\Lambda(\mathbf{A})}\pi(\lambda)^2\,\|\Pi_\lambda\mathbf{q}_1\|^2
$$

as small as possible. To keep this sum small, $\pi_m$ must be small at every eigenvalue that carries a large share of $\mathbf{q}_1$. The Ritz values at step $m$ are exactly the roots of $\pi_m$: expanding $\det(t\,\mathbb{I}_m-\mathbf{T}_m)$ along its last row gives the same three-term recurrence as $\beta_1\cdots\beta_m\,p_m(t)$, so $\det(t\,\mathbb{I}_m-\mathbf{T}_m)=\pi_m(t)$. The Ritz values therefore sit near the eigenvalues at which $\pi_m$ has to be small. Once $m$ equals the number $s$ of distinct eigenvalues present in $\mathbf{q}_1$, the polynomial $\prod_\lambda(t-\lambda)$ over those eigenvalues makes the sum zero. The remainder then vanishes, $\beta_s=0$, and the Ritz values are exactly those eigenvalues ([Proposition 3](#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run)). [Lemma 5](#47-lemma-5-variational-characterization-and-monotonicity) adds that the extreme Ritz values move monotonically toward $\lambda_{\min}(\mathbf{A})$ and $\lambda_{\max}(\mathbf{A})$.

### 4.3 A three-dimensional example

The steps of [Section 4.2](#42-how-the-method-works) can be followed with numbers on a small matrix. Take

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

and their eigenvalues, the Ritz values, are $\{2\}$, then $\{2-\sqrt{2/3},\,2+\sqrt{2/3}\}\approx\{1.18,\,2.82\}$, then $\{1,2,3\}$. The extreme Ritz values move outward toward the extreme eigenvalues $1$ and $3$, as [Lemma 5](#47-lemma-5-variational-characterization-and-monotonicity) guarantees. At $m=3$, they are exactly the eigenvalues of $\mathbf{A}$, as [Proposition 3](#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) guarantees for a starting vector with a nonzero component on every eigenvector. At $m=2$, the eigenvectors of $\mathbf{T}_2$ are $\frac{1}{\sqrt2}[1,\pm1]^\top$, so both Ritz pairs have residual $\beta_2|y_2|=\frac{1}{\sqrt3}\cdot\frac{1}{\sqrt2}\approx0.41$ ([Lemma 4](#46-lemma-4-ritz-pairs-and-their-residuals)). This bounds their actual distances to the spectrum, $0.18$ in both cases ([Lemma 2](#36-lemma-2-residual-certificate-for-an-approximate-eigenpair)). For comparison, power iteration from the same $\mathbf{q}_1$ estimates $\lambda_{\max}(\mathbf{A})=3$ after the same two products by $\mathbf{v}_1^\top\mathbf{A}\mathbf{v}_1=36/14\approx2.57$, with $\mathbf{v}_1=\mathbf{A}\mathbf{q}_1/\|\mathbf{A}\mathbf{q}_1\|$, which is farther from $3$ than the Ritz value $2.82$. In the polynomial reading of Section 4.2, the Ritz values $2\pm\sqrt{2/3}$ are the roots of $\pi_2(t)=(t-2)^2-2/3$, the monic quadratic that makes $\|\pi(\mathbf{A})\mathbf{q}_1\|$ smallest, and at $m=3$ the polynomial $(t-1)(t-2)(t-3)$ makes it zero, which is the breakdown.

### 4.4 Definition 2 [Lanczos recurrence and exact breakdown]

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

The vectors $\mathbf{q}_j$ are basis vectors, not generally eigenvectors of $\mathbf{A}$. Each step orthogonalizes $\mathbf{A}\mathbf{q}_j$ against the two most recent basis vectors only, for the reasons given in [Section 4.2](#42-how-the-method-works). This <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="three-term-recurrence" aria-label="Explain why a three-term recurrence suffices">three-term recurrence</button><span id="three-term-recurrence" class="explanation-popover" popover="auto" role="note" aria-label="Three-term recurrence" data-label="Linear algebra">Orthogonalizing $\mathbf A\mathbf q_j$ against every previous basis vector, as in Gram–Schmidt, would cost more at every step. For symmetric $\mathbf A$, the coefficients against $\mathbf q_1,\dots,\mathbf q_{j-2}$ vanish ([Lemma 3](#45-lemma-3-orthogonality-and-tridiagonal-projection)), so the two coefficients $\alpha_j$ and $\beta_{j-1}$ suffice.</span></span> is enough because $\mathbf{A}$ is symmetric. [Lemma 3](#45-lemma-3-orthogonality-and-tridiagonal-projection) shows that, in exact arithmetic, $\mathbf{A}\mathbf{q}_j$ is automatically orthogonal to $\mathbf{q}_1,\dots,\mathbf{q}_{j-2}$, so the projected matrix is tridiagonal and each step costs one product plus $O(d)$ further operations.

### 4.5 Lemma 3 [Orthogonality and tridiagonal projection]

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

$$
\begin{aligned}
\alpha_j
&=\mathbf{q}_j^\top\bigl(\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}\bigr)\\
&=\mathbf{q}_j^\top\mathbf{A}\mathbf{q}_j .
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\mathbf{q}_j^\top\mathbf{r}_j
&=\mathbf{q}_j^\top\mathbf{z}_j-\alpha_j\mathbf{q}_j^\top\mathbf{q}_j\\
&=\alpha_j-\alpha_j\\
&=0 .
\end{aligned}
$$

*The case $i=j-1$, for $j\ge2$.* The previous recurrence gives

$$
\mathbf{A}\mathbf{q}_{j-1}=\beta_{j-2}\mathbf{q}_{j-2}+\alpha_{j-1}\mathbf{q}_{j-1}+\beta_{j-1}\mathbf{q}_j .
$$

Using symmetry,

$$
\begin{aligned}
\mathbf{q}_{j-1}^\top\mathbf{r}_j
&=\mathbf{q}_{j-1}^\top\mathbf{A}\mathbf{q}_j-\beta_{j-1}\\
&=\textcolor{#2563eb}{(\mathbf{A}\mathbf{q}_{j-1})^\top\mathbf{q}_j}-\beta_{j-1}\\
&=\beta_{j-1}-\beta_{j-1}\\
&=0 .
\end{aligned}
$$

In the first line, $\mathbf{q}_{j-1}^\top\mathbf{q}_{j-1}=1$ and the term $\alpha_j\mathbf{q}_{j-1}^\top\mathbf{q}_j$ vanishes.

*The case $i\le j-2$.* The recurrence at step $i$ shows that

$$
\mathbf{A}\mathbf{q}_i\in\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{i+1}\}.
$$

Since $i+1\le j-1$, this vector is orthogonal to $\mathbf{q}_j$. Hence

$$
\begin{aligned}
\mathbf{q}_i^\top\mathbf{r}_j
&=\mathbf{q}_i^\top\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_i^\top\mathbf{q}_{j-1}-\alpha_j\mathbf{q}_i^\top\mathbf{q}_j\\
&=\textcolor{#2563eb}{(\mathbf{A}\mathbf{q}_i)^\top\mathbf{q}_j}\\
&=0 .
\end{aligned}
$$

The <span style="color:#2563eb">blue terms</span> are where symmetry enters. Moving $\mathbf{A}$ from $\mathbf{q}_j$ onto an older basis vector $\mathbf{q}_i$ turns the inner product into one with $\mathbf{A}\mathbf{q}_i$, and an earlier step of the recurrence has already expanded $\mathbf{A}\mathbf{q}_i$ in the basis. This is why orthogonalizing against the two most recent vectors suffices.

Thus $\mathbf{r}_j$ is orthogonal to every column of $\mathbf{Q}_j$. If $\beta_j>0$, normalizing $\mathbf{r}_j$ extends the orthonormal family.

**Krylov subspaces.** The identity holds at $m=1$. Suppose it holds at iteration $j$ and $\beta_j>0$. Then

$$
\mathbf{q}_{j+1}=\frac{\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}-\alpha_j\mathbf{q}_j}{\beta_j}\in\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1),
$$

because $\mathbf{q}_{j-1}$ and $\mathbf{q}_j$ belong to $\mathcal{K}_j(\mathbf{A},\mathbf{q}_1)$, whose image under $\mathbf{A}$ lies in $\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1)$. Consequently,

$$
\operatorname{span}\{\mathbf{q}_1,\dots,\mathbf{q}_{j+1}\}\subseteq\mathcal{K}_{j+1}(\mathbf{A},\mathbf{q}_1).
$$

The left-hand side has dimension $j+1$, because its vectors are orthonormal. The right-hand side is generated by $j+1$ vectors, so its dimension is at most $j+1$. The two spaces are therefore equal.

**Matrix identity.** For each completed iteration $j<m$,

$$
\mathbf{A}\mathbf{q}_j=\beta_{j-1}\mathbf{q}_{j-1}+\alpha_j\mathbf{q}_j+\beta_j\mathbf{q}_{j+1}.
$$

For the final column,

$$
\mathbf{A}\mathbf{q}_m=\beta_{m-1}\mathbf{q}_{m-1}+\alpha_m\mathbf{q}_m+\mathbf{r}_m .
$$

Collecting these identities column by column gives

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\mathbf{r}_m\mathbf{e}_m^\top .
$$

Multiplying on the left by $\mathbf{Q}_m^\top$, and using

$$
\mathbf{Q}_m^\top\mathbf{Q}_m=\mathbb{I}_m,
\qquad
\mathbf{Q}_m^\top\mathbf{r}_m=0,
$$

gives

$$
\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m=\mathbf{T}_m .
$$

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

The residual form of this identity remains valid when $\beta_m=0$. When $\beta_m>0$, it can also be written as

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m+\beta_m\mathbf{q}_{m+1}\mathbf{e}_m^\top .
$$

No vector $\mathbf{q}_{m+1}$ needs to be defined at an exact breakdown.

### 4.6 Lemma 4 [Ritz pairs and their residuals]

Let

$$
\mathbf{T}_m\mathbf{y}=\vartheta\,\mathbf{y},
\qquad
\|\mathbf{y}\|=1,
$$

and define

$$
\mathbf{x}=\mathbf{Q}_m\mathbf{y}.
$$

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

$$
\begin{aligned}
\|\mathbf{x}\|^2
&=\mathbf{y}^\top\mathbf{Q}_m^\top\mathbf{Q}_m\mathbf{y}\\
&=\mathbf{y}^\top\mathbf{y}\\
&=1 .
\end{aligned}
$$

Also,

$$
\begin{aligned}
\mathbf{x}^\top\mathbf{A}\mathbf{x}
&=\mathbf{y}^\top\mathbf{Q}_m^\top\mathbf{A}\mathbf{Q}_m\mathbf{y}\\
&=\mathbf{y}^\top\mathbf{T}_m\mathbf{y}\\
&=\vartheta\,\mathbf{y}^\top\mathbf{y}\\
&=\vartheta .
\end{aligned}
$$

Using [Lemma 3](#45-lemma-3-orthogonality-and-tridiagonal-projection),

$$
\begin{aligned}
\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}
&=\mathbf{A}\mathbf{Q}_m\mathbf{y}-\vartheta\mathbf{Q}_m\mathbf{y}\\
&=\mathbf{Q}_m\mathbf{T}_m\mathbf{y}+\mathbf{r}_m\mathbf{e}_m^\top\mathbf{y}-\vartheta\mathbf{Q}_m\mathbf{y}\\
&=\mathbf{Q}_m\textcolor{#2563eb}{(\mathbf{T}_m\mathbf{y}-\vartheta\mathbf{y})}+\mathbf{r}_m y_m\\
&=\mathbf{r}_m y_m .
\end{aligned}
$$

The <span style="color:#2563eb">blue factor</span> vanishes because $\mathbf{y}$ is an eigenvector of $\mathbf{T}_m$. Taking norms gives

$$
\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|=\|\mathbf{r}_m\|\,|y_m|=\beta_m|y_m| .
$$

The eigenvalue bound is [Lemma 2](#36-lemma-2-residual-certificate-for-an-approximate-eigenpair) applied to the unit vector $\mathbf{x}$, whose Rayleigh quotient is $\vartheta$.

<details class="article-disclosure">
<summary>The eigenbasis expansion, written out for the Ritz vector</summary>
<div class="article-disclosure__body">

Expand $\mathbf{x}$ in an orthonormal eigenbasis $\mathbf{u}_1,\dots,\mathbf{u}_d$ of $\mathbf{A}$:

$$
\mathbf{x}=\sum_{i=1}^dc_i\mathbf{u}_i,
\qquad
\mathbf{A}\mathbf{u}_i=\lambda_i\mathbf{u}_i,
\qquad
\sum_{i=1}^dc_i^2=1 .
$$

Then

$$
\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}=\sum_{i=1}^dc_i(\lambda_i-\vartheta)\mathbf{u}_i,
$$

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

### 4.7 Lemma 5 [Variational characterization and monotonicity]

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

The <span style="color:#16a34a">extreme Ritz values</span> are the extreme Rayleigh quotients over a growing family of subspaces. They can therefore only move outward, toward the extreme eigenvalues of $\mathbf{A}$, and never overshoot them, as the <span style="color:#16a34a">green curves</span> of panel (a) in the [figure of Section 4.9](#figure-lanczos) show.

#### Proof

Every vector in $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$ has a unique representation

$$
\mathbf{x}=\mathbf{Q}_m\mathbf{y}.
$$

Moreover,

$$
\|\mathbf{x}\|^2=\mathbf{y}^\top\mathbf{Q}_m^\top\mathbf{Q}_m\mathbf{y}=\|\mathbf{y}\|^2,
$$

and

$$
\mathbf{x}^\top\mathbf{A}\mathbf{x}=\mathbf{y}^\top\mathbf{T}_m\mathbf{y}.
$$

Let $\mathbf{s}_1,\dots,\mathbf{s}_m$ be an orthonormal eigenbasis of $\mathbf{T}_m$, with eigenvalues $\vartheta_1,\dots,\vartheta_m$. For any unit vector

$$
\mathbf{y}=\sum_{i=1}^mc_i\mathbf{s}_i,
\qquad
\sum_{i=1}^mc_i^2=1,
$$

we have

$$
\mathbf{y}^\top\mathbf{T}_m\mathbf{y}=\sum_{i=1}^m\vartheta_i c_i^2 .
$$

Thus

$$
\min_i\vartheta_i\le\mathbf{y}^\top\mathbf{T}_m\mathbf{y}\le\max_i\vartheta_i .
$$

The lower and upper bounds are attained by eigenvectors associated with the smallest and largest eigenvalues. This proves the two variational identities.

Since

$$
\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\subseteq\mathcal{K}_{m+1}(\mathbf{A},\mathbf{q}_1),
$$

minimizing over the larger set can only decrease the minimum, while maximizing over the larger set can only increase the maximum. Therefore,

$$
\vartheta_{\min}^{(m+1)}\le\vartheta_{\min}^{(m)},
\qquad
\vartheta_{\max}^{(m)}\le\vartheta_{\max}^{(m+1)}.
$$

Finally, every unit vector $\mathbf{x}\in\mathbb{R}^d$ satisfies

$$
\lambda_{\min}(\mathbf{A})\le\mathbf{x}^\top\mathbf{A}\mathbf{x}\le\lambda_{\max}(\mathbf{A}),
$$

as follows from the same eigenbasis expansion for $\mathbf{A}$. This gives the remaining bounds.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

This characterization provides a precise connection with power iteration. Started from $\mathbf{v}_0=\mathbf{q}_1$, and whenever $\mathbf{A}^{m-1}\mathbf{q}_1\neq0$, the normalized power iterate

$$
\mathbf{v}_{m-1}=\frac{\mathbf{A}^{m-1}\mathbf{q}_1}{\|\mathbf{A}^{m-1}\mathbf{q}_1\|}
$$

belongs to $\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)$. Consequently,

$$
\vartheta_{\min}^{(m)}\le\mathbf{v}_{m-1}^\top\mathbf{A}\mathbf{v}_{m-1}\le\vartheta_{\max}^{(m)}.
$$

Lanczos optimizes the Rayleigh quotient over all unit vectors in the available Krylov subspace. The normalized power vector is one particular member of that subspace.

Both estimates cost $m$ matrix–vector products: $\mathbf{T}_m$ needs $\mathbf{A}\mathbf{q}_1,\dots,\mathbf{A}\mathbf{q}_m$, and the Rayleigh quotient of $\mathbf{v}_{m-1}$ needs $\mathbf{A}\mathbf{v}_0,\dots,\mathbf{A}\mathbf{v}_{m-1}$. The eigenvalue of largest magnitude is $\lambda_{\max}(\mathbf{A})$ or $\lambda_{\min}(\mathbf{A})$. In the first case, $\mathbf{v}_{m-1}^\top\mathbf{A}\mathbf{v}_{m-1}\le\vartheta_{\max}^{(m)}\le\lambda_{\max}(\mathbf{A})$, and in the second, $\lambda_{\min}(\mathbf{A})\le\vartheta_{\min}^{(m)}\le\mathbf{v}_{m-1}^\top\mathbf{A}\mathbf{v}_{m-1}$. Either way, at an equal number of products, the extreme Ritz value on the side of the dominant eigenvalue is at least as close to it as the power-iteration estimate (middle panel of the figure in [Section 4.9](#49-corollary-3-random-initialization-and-repeated-eigenvalues)).

This variational statement does not imply that Lanczos has a lower total runtime for every matrix. Constructing and retaining the basis introduces additional work and storage.

### 4.8 Proposition 3 [Exact termination and the eigenvalues recovered by one run]

For each $\lambda\in\Lambda(\mathbf{A})$, let $\Pi_\lambda$ be the orthogonal projection onto $E_\lambda$. Define

$$
\Lambda_1=\{\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{q}_1\neq0\},
\qquad
s=|\Lambda_1| .
$$

Here $\Lambda_1$ contains distinct eigenvalues, irrespective of their multiplicities.

A single Lanczos run terminates at exactly iteration $s$:

$$
\beta_j>0\quad(1\le j<s),
\qquad
\beta_s=0 .
$$

At termination,

$$
\mathbf{A}\mathbf{Q}_s=\mathbf{Q}_s\mathbf{T}_s,
$$

and the eigenvalues of $\mathbf{T}_s$ are precisely the elements of $\Lambda_1$, each appearing once.

#### Remark 15 [One direction per distinct eigenvalue]

A run can only discover the eigenvalues present in $\mathbf{q}_1$, and it discovers each of them once, however large its multiplicity. Inside the Krylov subspace, the whole eigenspace $E_\lambda$ is represented by the single vector $\Pi_\lambda\mathbf{q}_1$. The number of steps before an exact breakdown is therefore the number of distinct eigenvalues present in $\mathbf{q}_1$, not $d$.

#### Proof

Write

$$
\mathbf{p}_\lambda=\Pi_\lambda\mathbf{q}_1,
\qquad
\lambda\in\Lambda_1,
$$

and define

$$
W=\operatorname{span}\{\mathbf{p}_\lambda:\lambda\in\Lambda_1\}.
$$

The vectors $\mathbf{p}_\lambda$ are nonzero and mutually orthogonal, so

$$
\dim W=s .
$$

**The Krylov subspaces lie in $W$.** Every power of $\mathbf{A}$ applied to $\mathbf{q}_1$ belongs to $W$, because

$$
\mathbf{q}_1=\sum_{\lambda\in\Lambda_1}\mathbf{p}_\lambda
$$

and, for $k\ge1$,

$$
\mathbf{A}^k\mathbf{q}_1=\sum_{\lambda\in\Lambda_1}\lambda^k\mathbf{p}_\lambda .
$$

Therefore,

$$
\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)\subseteq W
\qquad
\text{for every }m .
$$

**The Krylov subspace of order $s$ equals $W$.** For each $\lambda\in\Lambda_1$, define the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="lagrange-polynomial" aria-label="Explain the Lagrange interpolation polynomial">Lagrange polynomial</button><span id="lagrange-polynomial" class="explanation-popover" popover="auto" role="note" aria-label="Lagrange interpolation polynomial" data-label="Definition">$\ell_\lambda$ is the unique polynomial of degree at most $s-1$ that equals $1$ at $\lambda$ and $0$ at the other $s-1$ points of $\Lambda_1$. Applied to $\mathbf A$, it keeps the component of $\mathbf q_1$ along $E_\lambda$ and removes every other component present in $\mathbf q_1$.</span></span>

$$
\ell_\lambda(t)=\prod_{\substack{\mu\in\Lambda_1\\\mu\neq\lambda}}\frac{t-\mu}{\lambda-\mu}.
$$

When $s=1$, the empty product is defined as $1$. This polynomial has degree at most $s-1$ and satisfies

$$
\ell_\lambda(\mu)=
\begin{cases}
1,&\mu=\lambda,\\
0,&\mu\neq\lambda,
\end{cases}
\qquad\mu\in\Lambda_1 .
$$

Consequently,

$$
\begin{aligned}
\ell_\lambda(\mathbf{A})\mathbf{q}_1
&=\sum_{\mu\in\Lambda_1}\ell_\lambda(\mu)\mathbf{p}_\mu\\
&=\mathbf{p}_\lambda .
\end{aligned}
$$

Since $\ell_\lambda$ has degree at most $s-1$,

$$
\mathbf{p}_\lambda\in\mathcal{K}_s(\mathbf{A},\mathbf{q}_1).
$$

Thus

$$
W\subseteq\mathcal{K}_s(\mathbf{A},\mathbf{q}_1),
$$

and hence

$$
\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)=W .
$$

**No breakdown before iteration $s$.** Suppose a breakdown occurred at some $m<s$. [Lemma 3](#45-lemma-3-orthogonality-and-tridiagonal-projection) would give

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_m\mathbf{T}_m,
$$

so

$$
\mathcal{K}_m(\mathbf{A},\mathbf{q}_1)=\operatorname{range}(\mathbf{Q}_m)
$$

would be invariant under $\mathbf{A}$. Since it contains $\mathbf{q}_1$, it would then contain every power $\mathbf{A}^k\mathbf{q}_1$, and therefore

$$
\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)\subseteq\mathcal{K}_m(\mathbf{A},\mathbf{q}_1).
$$

This is impossible, because

$$
\dim\mathcal{K}_s(\mathbf{A},\mathbf{q}_1)=s>m=\dim\mathcal{K}_m(\mathbf{A},\mathbf{q}_1).
$$

Thus no breakdown occurs before iteration $s$.

**Breakdown at iteration $s$.** At iteration $s$, all columns of $\mathbf{Q}_s$ belong to $W$. Since $W$ is spanned by eigenvectors, it is invariant under $\mathbf{A}$. Therefore,

$$
\mathbf{r}_s=\mathbf{A}\mathbf{q}_s-\beta_{s-1}\mathbf{q}_{s-1}-\alpha_s\mathbf{q}_s\in W .
$$

But [Lemma 3](#45-lemma-3-orthogonality-and-tridiagonal-projection) also gives

$$
\mathbf{r}_s\perp\operatorname{range}(\mathbf{Q}_s)=W .
$$

Hence $\mathbf{r}_s=0$, and $\beta_s=0$.

**The eigenvalues of $\mathbf{T}_s$.** The normalized vectors

$$
\mathbf{u}_\lambda=\frac{\mathbf{p}_\lambda}{\|\mathbf{p}_\lambda\|},
\qquad
\lambda\in\Lambda_1,
$$

form an orthonormal eigenbasis of $W$. Set

$$
\mathbf{y}_\lambda=\mathbf{Q}_s^\top\mathbf{u}_\lambda .
$$

Because $\mathbf{u}_\lambda\in W=\operatorname{range}(\mathbf{Q}_s)$,

$$
\mathbf{Q}_s\mathbf{Q}_s^\top\mathbf{u}_\lambda=\mathbf{u}_\lambda .
$$

It follows that

$$
\begin{aligned}
\mathbf{T}_s\mathbf{y}_\lambda
&=\mathbf{Q}_s^\top\mathbf{A}\mathbf{Q}_s\mathbf{Q}_s^\top\mathbf{u}_\lambda\\
&=\mathbf{Q}_s^\top\mathbf{A}\mathbf{u}_\lambda\\
&=\lambda\,\mathbf{Q}_s^\top\mathbf{u}_\lambda\\
&=\lambda\,\mathbf{y}_\lambda .
\end{aligned}
$$

Moreover, for $\lambda,\mu\in\Lambda_1$,

$$
\begin{aligned}
\mathbf{y}_\lambda^\top\mathbf{y}_\mu
&=\mathbf{u}_\lambda^\top\mathbf{Q}_s\mathbf{Q}_s^\top\mathbf{u}_\mu\\
&=\mathbf{u}_\lambda^\top\mathbf{u}_\mu\\
&=\delta_{\lambda\mu} .
\end{aligned}
$$

Thus the $s$ vectors $\mathbf{y}_\lambda$ form an orthonormal basis of $\mathbb{R}^s$. The corresponding elements of $\Lambda_1$ are therefore all the eigenvalues of $\mathbf{T}_s$, each appearing once.

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

### 4.9 Corollary 3 [Random initialization and repeated eigenvalues]

If $\mathbf{g}\sim\mathcal{N}(0,\mathbb{I}_d)$ and $\mathbf{q}_1=\mathbf{g}/\|\mathbf{g}\|$, then, with probability one, a terminating Lanczos run recovers every distinct eigenvalue of $\mathbf{A}$. If $\mathbf{A}$ has $d$ distinct eigenvalues, the run reaches $s=d$ and yields a complete eigenbasis. If eigenvalues are repeated, a single run yields only one eigenvector from each corresponding eigenspace. With probability one, the eigenvector found in $E_\lambda$ at termination is the normalized projection of $\mathbf{q}_1$ onto $E_\lambda$.

#### Proof

For each $\lambda\in\Lambda(\mathbf{A})$, the event $\Pi_\lambda\mathbf{g}=0$ means that $\mathbf{g}$ lies in the proper linear subspace $E_\lambda^\perp$, which has Gaussian probability zero. Since there are finitely many distinct eigenvalues, the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="union-bound" aria-label="State the union bound">union bound</button><span id="union-bound" class="explanation-popover" popover="auto" role="note" aria-label="Union bound" data-label="Fact">For finitely many events, the probability that at least one of them occurs is at most the sum of their probabilities. A finite union of probability-zero events therefore has probability zero.</span></span> gives

$$
\begin{aligned}
\mathbb{P}\bigl(\exists\lambda\in\Lambda(\mathbf{A}):\Pi_\lambda\mathbf{g}=0\bigr)
&\le\sum_{\lambda\in\Lambda(\mathbf{A})}\mathbb{P}(\Pi_\lambda\mathbf{g}=0)\\
&=0 .
\end{aligned}
$$

Thus $\Lambda_1=\Lambda(\mathbf{A})$ with probability one, and [Proposition 3](#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) applies.

For any polynomial $p$, $\Pi_\lambda p(\mathbf{A})\mathbf{q}_1=p(\lambda)\Pi_\lambda\mathbf{q}_1$. Therefore, the projection of the entire Krylov subspace onto $E_\lambda$ is contained in $\operatorname{span}\{\Pi_\lambda\mathbf{q}_1\}$, which is one-dimensional when $\Pi_\lambda\mathbf{q}_1\neq0$. A single run cannot recover additional independent directions in that eigenspace. At termination, the eigenvectors $\mathbf{y}_\lambda$ of $\mathbf{T}_s$ from the proof of [Proposition 3](#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run) lift to the Ritz vectors $\mathbf{Q}_s\mathbf{y}_\lambda=\mathbf{u}_\lambda$, the normalized projections of $\mathbf{q}_1$.

When the spectrum is simple and every eigenvalue is present in the initialization, $s=d$. In this case, $\mathbf{Q}_d$ is an orthogonal matrix, and

$$
\mathbf{A}=\mathbf{Q}_d\mathbf{T}_d\mathbf{Q}_d^\top .
$$

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
  <figcaption>The Lanczos method on a symmetric matrix of size $d=60$ with random eigenvectors and only $s=12$ distinct eigenvalues, of multiplicities 1 to 9 (labels on the right of panel a), including the eigenvalue 0 with multiplicity 8. (a) One run: Ritz values at each step $m$ (<span style="color:#27322f">black dots</span>) against the distinct eigenvalues (<span style="color:#7d8783">grey lines</span>). The extreme Ritz values (<span style="color:#16a34a">green</span>) move outward monotonically (<a href="#47-lemma-5-variational-characterization-and-monotonicity">Lemma 5</a>), the power-iteration estimate after the same number of products (<span style="color:#ea580c">orange</span>) stays between them, and at $m=s=12$ the Ritz values are exactly the 12 distinct eigenvalues, each appearing once (<a href="#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run">Proposition 3</a>). (b) Distance from $\lambda_{\max}(\mathbf A)=2$ of the largest Ritz value (<span style="color:#16a34a">green</span>) and of the power-iteration estimate (<span style="color:#ea580c">orange</span>), at equal numbers of matrix–vector products. (c) The coefficient $\beta_m$ falls from about $0.3$ at $m=11$ to about $10^{-12}$ at $m=12$: the exact breakdown of <a href="#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run">Proposition 3</a>, up to floating-point error. For every starting vector, the 12 Ritz values at $m=12$ match the 12 distinct eigenvalues to within $3\cdot10^{-15}$. Panels (b) and (c) show medians over 200 Gaussian starting vectors, with bands containing the middle half of the runs (the interquartile range), and all runs use full reorthogonalization.</figcaption>
</figure>

### 4.10 Proposition 4 [Complete decomposition by continuation with new starting vectors]

Let $\mathbf{A}\in\mathbb{R}^{d\times d}$ be symmetric. Initialize $\mathbf{q}_0=0$, $\beta_0=0$, and any unit vector $\mathbf{q}_1$. Run the Lanczos recurrence for $j=1,\dots,d$, with the following rule whenever $j<d$:

$$
\mathbf{q}_{j+1}=
\begin{cases}
\mathbf{r}_j/\beta_j,&\beta_j>0,\\
\text{any unit vector orthogonal to }\mathbf{q}_1,\dots,\mathbf{q}_j,&\beta_j=0 .
\end{cases}
$$

At a breakdown, retain $\beta_j=0$ and all previously generated basis vectors.

Then $\mathbf{Q}_d=[\mathbf{q}_1,\dots,\mathbf{q}_d]$ is orthogonal,

$$
\beta_d=0,
\qquad
\mathbf{A}\mathbf{Q}_d=\mathbf{Q}_d\mathbf{T}_d,
$$

and

$$
\mathbf{A}=\mathbf{Q}_d\mathbf{T}_d\mathbf{Q}_d^\top .
$$

The matrix $\mathbf{T}_d$ is symmetric and tridiagonal, with a zero off-diagonal entry at every boundary between runs.

If

$$
\mathbf{T}_d=\mathbf{Z}\mathbf{D}\mathbf{Z}^\top,
\qquad
\mathbf{Z}^\top\mathbf{Z}=\mathbb{I}_d,
$$

with $\mathbf{D}\in\mathbb{R}^{d\times d}$ the diagonal matrix of the eigenvalues of $\mathbf{T}_d$, then

$$
\mathbf{V}=\mathbf{Q}_d\mathbf{Z}
$$

is a complete orthogonal eigenvector matrix of $\mathbf{A}$:

$$
\mathbf{V}^\top\mathbf{V}=\mathbb{I}_d,
\qquad
\mathbf{A}\mathbf{V}=\mathbf{V}\mathbf{D}.
$$

#### Remark 16 [Why continuation completes the basis]

Each run discovers one direction in every eigenspace that its starting vector reaches, then stops at an invariant subspace ([Proposition 3](#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run)). Continuing with a new starting vector in the orthogonal complement, which is invariant as well, lets the next run find the directions that the previous runs could not see.

#### Proof

At an exact breakdown after iteration $j$,

$$
\mathbf{A}\mathbf{Q}_j=\mathbf{Q}_j\mathbf{T}_j .
$$

Thus $\operatorname{range}(\mathbf{Q}_j)$ is invariant under $\mathbf{A}$.

Its orthogonal complement is also invariant. Indeed, if $\mathbf{Q}_j^\top\mathbf{x}=0$, symmetry gives

$$
\begin{aligned}
\mathbf{Q}_j^\top\mathbf{A}\mathbf{x}
&=(\mathbf{A}\mathbf{Q}_j)^\top\mathbf{x}\\
&=(\mathbf{Q}_j\mathbf{T}_j)^\top\mathbf{x}\\
&=\mathbf{T}_j\mathbf{Q}_j^\top\mathbf{x}\\
&=0 .
\end{aligned}
$$

A new unit vector in this complement therefore starts another Lanczos run entirely within the complement. Since $\beta_j=0$, the first recurrence of that run contains no contribution from the preceding block.

Within each run, [Lemma 3](#45-lemma-3-orthogonality-and-tridiagonal-projection) ensures orthogonality. Vectors from different runs are orthogonal because they lie in orthogonal invariant subspaces. Each run adds at least one vector, and the construction can continue whenever fewer than $d$ vectors have been obtained. Thus it produces $d$ orthonormal vectors.

At the final iteration, $\mathbf{r}_d$ is orthogonal to every column of $\mathbf{Q}_d$. These columns form a basis of $\mathbb{R}^d$, so

$$
\mathbf{r}_d=0,
\qquad
\beta_d=0 .
$$

Collecting all recurrences, including the zero couplings at block boundaries, gives

$$
\mathbf{A}\mathbf{Q}_d=\mathbf{Q}_d\mathbf{T}_d .
$$

Multiplying on the right by $\mathbf{Q}_d^\top$ gives

$$
\mathbf{A}=\mathbf{Q}_d\mathbf{T}_d\mathbf{Q}_d^\top .
$$

Finally,

$$
\begin{aligned}
\mathbf{A}(\mathbf{Q}_d\mathbf{Z})
&=\mathbf{Q}_d\mathbf{T}_d\mathbf{Z}\\
&=\mathbf{Q}_d\mathbf{Z}\mathbf{D},
\end{aligned}
$$

and

$$
\begin{aligned}
(\mathbf{Q}_d\mathbf{Z})^\top(\mathbf{Q}_d\mathbf{Z})
&=\mathbf{Z}^\top\mathbf{Q}_d^\top\mathbf{Q}_d\mathbf{Z}\\
&=\mathbf{Z}^\top\mathbf{Z}\\
&=\mathbb{I}_d .
\end{aligned}
$$

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

### 4.11 Algorithm 2 [Lanczos with reorthogonalization and residual checks]

For finite computations, the preceding identities require additional care. In particular, the short recurrence does not maintain exact orthogonality in floating-point arithmetic. A direct implementation can reorthogonalize each newly computed residual against the stored basis,

$$
\mathbf{r}_j\leftarrow\mathbf{r}_j-\mathbf{Q}_j(\mathbf{Q}_j^\top\mathbf{r}_j),
$$

repeating the projection when needed before computing $\beta_j$ and normalizing. Such <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="full-reorthogonalization" aria-label="Show the source on full reorthogonalization">full reorthogonalization</button><span id="full-reorthogonalization" class="explanation-popover" popover="auto" role="note" aria-label="Full reorthogonalization" data-label="Source">See the discussion of full reorthogonalization in Bai et al., <a href="https://netlib.org/utk/people/JackDongarra/etemplates/node109.html"><em>Templates for the Solution of Algebraic Eigenvalue Problems</em></a>. The projection is carried out with classical or modified Gram–Schmidt, and it is repeated when the norm of the new vector drops below about $1/\sqrt2$ of its previous value, which rarely happens.</span></span> is a standard part of practical Lanczos implementations ([Bai et al., 2000](#ref-bai2000)). An implementable procedure for estimating selected eigenpairs is therefore the following.

1. **Fix the operator.** Provide a routine computing $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$, and keep everything that defines $\mathbf{A}$ fixed throughout the run. For a Hessian $\nabla_\theta^2f(\theta)$, these are $\theta$ and the function $f$, including the data it is evaluated on.
2. **Initialize the basis.** Choose a unit vector $\mathbf{q}_1$, and set $\mathbf{q}_0=0$ and $\beta_0=0$.
3. **Extend the basis.** Compute the Lanczos recurrence, reorthogonalize the residual, and obtain the next basis vector when its norm is sufficiently large.
4. **Solve the small eigenvalue problem.** At selected iterations, assemble $\mathbf{T}_m$ and compute its desired eigenpairs $\mathbf{T}_m\mathbf{y}_i=\vartheta_i\mathbf{y}_i$.
5. **Check candidates using the original operator.** Form $\mathbf{x}_i=\mathbf{Q}_m\mathbf{y}_i/\|\mathbf{Q}_m\mathbf{y}_i\|$, and recompute $\widehat{\lambda}_i=\mathbf{x}_i^\top\mathbf{A}\mathbf{x}_i$ and $\varepsilon_i=\|\mathbf{A}\mathbf{x}_i-\widehat{\lambda}_i\mathbf{x}_i\|$. Accept an approximate eigenpair when $\varepsilon_i\le\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\|\mathbf{A}\mathbf{x}_i\|$, where $\varepsilon_{\mathrm{abs}}>0$ and $\varepsilon_{\mathrm{rel}}\ge0$ are chosen tolerances.
6. **Stop or continue.** Stop when the requested candidates satisfy the tolerance or an iteration limit is reached. If the objective is a complete decomposition, use additional orthogonal starting vectors when a run terminates ([Proposition 4](#410-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors)).

$$
\begin{array}{r}
\blacksquare\Box
\end{array}
$$

In exact arithmetic, the inexpensive quantity

$$
\beta_m\bigl|(\mathbf{y}_i)_m\bigr|
$$

equals the residual norm ([Lemma 4](#46-lemma-4-ritz-pairs-and-their-residuals)). In floating-point arithmetic, loss of orthogonality and the corrections introduced by reorthogonalization perturb the exact tridiagonal identity. The inexpensive expression can be used to screen candidates, while the residual computed with the original operator supplies the final accuracy check. The normalization by $\|\mathbf{Q}_m\mathbf{y}_i\|$ in step 5 matters for the same reason: when the computed basis is not exactly orthonormal, $\mathbf{Q}_m\mathbf{y}_i$ need not have unit norm, and the inexpensive estimate can then be too optimistic (see the implementation discussion of the [Lanczos algorithm](https://netlib.org/utk/people/JackDongarra/etemplates/node104.html) in [Bai et al., 2000](#ref-bai2000)).

A small computed $\beta_j$ indicates an approximate breakdown. It is not an exact proof of invariance or spectral completeness. Likewise, satisfying a residual tolerance certifies an approximate eigenpair, not that no more extreme eigenvalue has been missed.

Each basis-extension step requires one matrix–vector product. If $C_{\mathbf{A}}$ denotes its cost, $m$ steps require $m$ such products, with total cost $mC_{\mathbf{A}}$, in addition to any products used for explicit residual checks. Retaining $\mathbf{Q}_m$ requires $O(dm)$ storage. Full reorthogonalization costs $O(dj)$ operations at iteration $j$, and therefore

$$
\sum_{j=1}^mO(dj)=O(dm^2)
$$

operations over the run, in addition to solving the projected eigenvalue problems.

When $\mathbf{A}$ is available only through products, as in the examples of [Section 2](#2-matrices-known-only-through-their-products), the run never forms $\mathbf{A}$. Nevertheless, a complete decomposition with $m=d$ still requires $O(d^2)$ storage for the basis, and full reorthogonalization then costs $O(d^3)$ operations. The practical benefit is greatest when the required eigenpairs reach the desired accuracy with $m\ll d$.

The [figure below](#figure-lanczos-vs-deflation) compares the two algorithms on the network Hessian of the [figure of Section 3.7](#figure-hessian-deflation) and on the bipartite graph of the [figure of Section 3.5](#figure-complete-eigenbasis).

<figure class="article-figure" id="figure-lanczos-vs-deflation">
  <img src="/images/blog/matrix-vector-products/lanczos-vs-deflation.svg" width="1040" height="430" alt="Two panels comparing Algorithm 2 (Lanczos) with Algorithm 1. Left: Hessian-vector products needed to certify the j largest eigenpairs of a network Hessian, from about 13 to 35 for Lanczos and from about 200 to 1000 for Algorithm 1. Right: eigenpairs certified against products for a 50-node bipartite graph; Lanczos with continuation reaches all 50 after 50 products, Algorithm 1 after about fifteen thousand." loading="lazy" />
  <figcaption><a href="#411-algorithm-2-lanczos-with-reorthogonalization-and-residual-checks">Algorithm 2</a> (<span style="color:#16a34a">green</span>) against <a href="#37-algorithm-1-deflated-squared-power-iteration-with-sign-separation">Algorithm 1</a> (<span style="color:#2563eb">blue</span>), with the same constant threshold $10^{-10}\|\mathbf A\|$ on the residuals for $\mathbf A$. (a) The network Hessian of the <a href="#figure-hessian-deflation">figure of Section 3.7</a>: Hessian–vector products used until the $j$ eigenpairs of largest absolute value are certified, medians over 20 starting vectors with bands containing the middle half of the runs. For Algorithm 1, the count runs over stages 1 to $j$. For Algorithm 2, it is the number of Lanczos steps until the $j$ Ritz pairs of largest absolute value pass the inexpensive test $\beta_m|(\mathbf y_i)_m|\le10^{-10}\|\mathbf A\|$ of <a href="#46-lemma-4-ritz-pairs-and-their-residuals">Lemma 4</a>, plus one product per pair for the explicit check of step 5, whose residuals were all below $5.4\cdot10^{-10}$. The eight pairs cost 35 products with Algorithm 2 and about 1,000 with Algorithm 1. (b) The complete eigenbasis of the bipartite graph of the <a href="#figure-complete-eigenbasis">figure of Section 3.5</a>. The first Lanczos run breaks down at step 41, the number of distinct eigenvalues (<a href="#48-proposition-3-exact-termination-and-the-eigenvalues-recovered-by-one-run">Proposition 3</a>), where its 41 Ritz pairs all pass the test. Each of the nine continuation runs of <a href="#410-proposition-4-complete-decomposition-by-continuation-with-new-starting-vectors">Proposition 4</a> starts in the eigenspace of $0$, orthogonally to the basis built so far, and breaks down after one step. The 50 products, plus 50 explicit checks (<span style="color:#16a34a">green square</span>), give all 50 eigenpairs with residuals below $1.2\cdot10^{-11}$, against 15,296 products for Algorithm 1 with every stage taken to its limit.</figcaption>
</figure>

## References

- <span id="ref-arnoldi1951"></span>W. E. Arnoldi, [“The Principle of Minimized Iterations in the Solution of the Matrix Eigenvalue Problem”](https://doi.org/10.1090/qam/42792), *Quarterly of Applied Mathematics* 9(1):17–29, 1951.
- <span id="ref-bai2000"></span>Zhaojun Bai, James Demmel, Jack Dongarra, Axel Ruhe, and Henk van der Vorst (editors), [*Templates for the Solution of Algebraic Eigenvalue Problems: A Practical Guide*](https://netlib.org/utk/people/JackDongarra/etemplates/book.html), SIAM, 2000. Sections used here: the Hermitian [Lanczos algorithm](https://netlib.org/utk/people/JackDongarra/etemplates/node104.html) and [full reorthogonalization](https://netlib.org/utk/people/JackDongarra/etemplates/node109.html).
- <span id="ref-cooley1965"></span>James W. Cooley and John W. Tukey, [“An Algorithm for the Machine Calculation of Complex Fourier Series”](https://doi.org/10.1090/S0025-5718-1965-0178586-1), *Mathematics of Computation* 19(90):297–301, 1965.
- <span id="ref-dembo1983"></span>Ron S. Dembo and Trond Steihaug, [“Truncated-Newton Algorithms for Large-Scale Unconstrained Optimization”](https://doi.org/10.1007/BF02592055), *Mathematical Programming* 26(2):190–212, 1983.
- <span id="ref-golub2000"></span>Gene H. Golub and Henk A. van der Vorst, [“Eigenvalue Computation in the 20th Century”](https://doi.org/10.1016/S0377-0427%2800%2900413-1), *Journal of Computational and Applied Mathematics* 123(1–2):35–65, 2000.
- <span id="ref-hestenes1952"></span>Magnus R. Hestenes and Eduard Stiefel, [“Methods of Conjugate Gradients for Solving Linear Systems”](https://doi.org/10.6028/jres.049.044), *Journal of Research of the National Bureau of Standards* 49(6):409–436, 1952.
- <span id="ref-householder1964"></span>Alston S. Householder, *The Theory of Matrices in Numerical Analysis*, Blaisdell, New York, 1964. Reprinted by Dover, 1975.
- <span id="ref-lanczos1950"></span>Cornelius Lanczos, [“An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators”](https://doi.org/10.6028/jres.045.026), *Journal of Research of the National Bureau of Standards* 45(4):255–282, 1950.
- <span id="ref-vonluxburg2007"></span>Ulrike von Luxburg, [“A Tutorial on Spectral Clustering”](https://doi.org/10.1007/s11222-007-9033-z), *Statistics and Computing* 17(4):395–416, 2007.
- <span id="ref-muntz1913"></span>Ch. Müntz, “Solution directe de l’équation séculaire et de quelques problèmes analogues transcendants”, *Comptes Rendus de l’Académie des Sciences*, Paris, 1913.
- <span id="ref-osborne1958"></span>Elmer E. Osborne, [“On Acceleration and Matrix Deflation Processes Used with the Power Method”](https://www.jstor.org/stable/2098697), *Journal of the Society for Industrial and Applied Mathematics* 6(3):279–287, 1958.
- <span id="ref-pearlmutter1994"></span>Barak A. Pearlmutter, [“Fast Exact Multiplication by the Hessian”](https://doi.org/10.1162/neco.1994.6.1.147), *Neural Computation* 6(1):147–160, 1994.
