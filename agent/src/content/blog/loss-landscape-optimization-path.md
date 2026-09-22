---
title: "Visualization of the loss landscape and optimization path of a neural network"
date: "2022-05-01"
category: "Research Notes"
image: "/images/blog/loss-landscape.png"
tags:
  - deep learning
  - loss landscape
excerpt: "How low-dimensional subspaces can preserve an optimization trajectory and make a neural network’s loss landscape interpretable."
---

<p class="article-lede">A useful loss-landscape plot should preserve the geometry of the optimization path, not merely project a high-dimensional surface onto an arbitrary plane.</p>

Neural-network <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="loss-landscape" aria-label="Define a loss landscape">loss landscapes</button><span id="loss-landscape" class="explanation-popover" popover="auto" role="note" aria-label="Loss landscape" data-label="Definition">A loss landscape is the scalar training objective viewed as a function of all model parameters. Its dimension therefore equals the number of trainable parameters.</span></span> live in very high-dimensional parameter spaces, whereas visualization is limited to one-dimensional curves or two-dimensional surfaces. Several methods attempt to close this dimensionality gap. The central idea is to choose a <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="linear-subspace" aria-label="Define the linear subspace used for visualization">low-dimensional linear subspace</button><span id="linear-subspace" class="explanation-popover" popover="auto" role="note" aria-label="Low-dimensional linear subspace" data-label="Geometry">Choose one or two directions in parameter space and evaluate the loss only on the line or plane they span. The resulting slice can be plotted directly.</span></span> that <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="trajectory-shape" aria-label="Explain preserving the optimization trajectory shape">maximally preserves the optimization trajectory’s shape</button><span id="trajectory-shape" class="explanation-popover" popover="auto" role="note" aria-label="Preserving trajectory shape" data-label="Goal">The projection should retain as much as possible of the distances, turns, and relative arrangement of the parameter iterates, so patterns in the plot remain meaningful for the original trajectory.</span></span>.

> <span class="article-kicker article-kicker--paper">Source</span> The longer draft is available in the [HackMD version](https://hackmd.io/@6LQ4mvRtS4Sc3LHkNEvDXQ/SkT-VIxyj). This web version is still being formatted and expanded.
