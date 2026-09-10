---
title: "Generating Random Variables and Stochastic Processes, Generative Flow Networks (GFlowNets)"
date: "2022-04-14"
category: "Sampling"
image: "/gifs/gflownet.gif"
tags:
  - Inverse Transform Sampling
  - Acceptance-Rejection Method
  - MCMC
  - Metropolis-Hasting
  - Gibbs sampling
  - Metropolis-adjusted Langevin
  - Important Sampling
  - GflowNets
excerpt: "Practical tutorial about GFlowNets, MCMC, Metropolis-Hasting, Gibbs sampling, and related stochastic simulation methods."
---

**Note:** It's better to read all the updates below before clicking on any link.

## Practical tutorial

<a href="https://github.com/Tikquuss/GflowNets_Tutorial">Here</a> here is the practical tutorial (theory & code)
I wrote in **Winter 2022** about GflowNets [1], MCMC, Metropolis-Hasting, Gibbs sampling, Metropolis-adjusted Langevin,
Inverse Transform Sampling, Acceptance-Rejection Method and Important Sampling.

I received a lot of positive feedback  on this tutorial, which has been the starting point for many in their
learning of GflowNets.

## More resources

<ul>
  <li>To go in depth with GflowNets : <i>GflowNets foundations</i> paper [2]
    or <i>Trajectory Balance</i> paper [3] (very pedagogical paper).</li>
  <li>For Variational Bayes, I recomment the paper <i>A practical tutorial on Variational Bayes</i> [4]</li>
  <li>See also <i>MCMC and Bayesian Modeling</i>, 2017, Martin Haugh, Columbia University </li>
</ul>

## Update: I met Pierre L’Ecuyer

In **Fall 2022**, wanting to update my level in probability and statistics,
I took <a href="https://www.iro.umontreal.ca/~lecuyer/ift6561.html">"IFT6561 : Stochastic Simulation"</a>, taught at the Université de Montréal by
the eminent Pierre L'Écuyer. This course is clearly a masterclass.
It's very theoretical and very practical at the same time.

Pierre L'Écuyer is the 2nd best teacher I've known in my life so far. I was very close to switching to another field,
since he was planning to take me on as a student; but unfortunately I was already being supervised.

His book, <i>"Stochastic Simulation and Monte Carlo Methods"</i>, a masterclass, is not yet public.
But if you ask for access he will send it to you.

Here are the book's headlines, captured from my reading plan. Click on each image to open the full-size version in a new tab.

<div class="media-grid">
  <a href="/images/tutorials/pierre-lecuyer-1.png" target="_blank" rel="noopener"><img src="/images/tutorials/pierre-lecuyer-1.png" alt="Reading plan for Pierre L'Écuyer's book, parts 1 and 2" loading="lazy"></a>
  <a href="/images/tutorials/pierre-lecuyer-2.png" target="_blank" rel="noopener"><img src="/images/tutorials/pierre-lecuyer-2.png" alt="Reading plan for Pierre L'Écuyer's book, parts 3, 4, and 5" loading="lazy"></a>
  <a href="/images/tutorials/pierre-lecuyer-3.png" target="_blank" rel="noopener"><img src="/images/tutorials/pierre-lecuyer-3.png" alt="Reading plan for Pierre L'Écuyer's book, parts 6 and 7" loading="lazy"></a>
</div>

**Note:** I mention this section because I'm supposed to add a section on <i>Gibbs sampling</i>,
<i>Metropolis-adjusted Langevin</i> and <i>Important Sampling</i> to my tutorial by now, from the book of Pierre.
I'll find the time to do it so that the tutorial can be complete.

## Update: Class presentation

<a href="/presentations/gflownets-ift6169.pdf">This</a> is
the presentation I gave in **Winter 2023** during the class
<a href="https://mitliagkas.github.io/ift6085-dl-theory-class/">"IFT6169: Theoretical principles for deep learning"</a>
taught in Mila by the masterful Ioannis Mitliagkas.

<object type="application/pdf" data="/presentations/gflownets-ift6169.pdf" title="GFlowNets class presentation" width="100%" height="800">
  <p><a href="/presentations/gflownets-ift6169.pdf">Open the GFlowNets class presentation (PDF)</a>.</p>
</object>

## References

<ul>
  <li>
    [1]
    <a href="https://arxiv.org/abs/2106.04399">
      Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation
    </a>
  </li>
  <li>
    [2]
    <a href="https://arxiv.org/pdf/2111.09266">
      GFlowNet Foundations
    </a>
  </li>
  <li>
    [3]
    <a href="https://arxiv.org/pdf/2201.13259">
      Trajectory balance: Improved credit assignment in GFlowNets
    </a>
  </li>
  <li>
    [3]
    <a href="https://arxiv.org/pdf/2103.01327">
      A practical tutorial on Variational Bayes
    </a>
  </li>
</ul>
