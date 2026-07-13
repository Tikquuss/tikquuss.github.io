---
title: "Model Merging via Data-Free Covariance Estimation"
authors: "Marawan Gamal Abdel Hameed, Derek Tam, Pascal Jr Tikeng Notsawo, Colin Raffel, Guillaume Rabusseau"
venue: "CATS (Continual Adaptation at Scale) Workshop, ICML 2026 (Oral); Conference on Language Modeling (COLM 2026)"
date: "2026-04-01"
paperUrl: "https://arxiv.org/abs/2604.01329"
image: "/images/publications/actmat/actmat.png"
tags: ["Model Merging", "Covariance Estimation", "Transfer Learning"]
abstract: "Model merging provides a way of cheaply combining individual models to produce a model that inherits each individual's capabilities."
---

Model merging provides a way of cheaply combining individual models to produce a model that inherits each individual's capabilities. While some merging methods can approach the performance of multitask training, they are often heuristically motivated and lack theoretical justification. A principled alternative is to pose model merging as a layer-wise optimization problem that directly minimizes interference between tasks. However, this formulation requires estimating per-layer covariance matrices from data, which may not be available when performing merging. In contrast, many of the heuristically-motivated methods do not require auxiliary data, making them practically advantageous. In this work, we revisit the interference minimization framework and show that, under certain conditions, covariance matrices can be estimated directly from *difference matrices*, eliminating the need for data while also reducing computational costs. We validate our approach across vision and language benchmarks on models ranging from $86M$ parameters to $7B$ parameters, outperforming previous data-free state-of-the-art merging methods.

Paper : [https://arxiv.org/abs/2604.01329](https://arxiv.org/abs/2604.01329)
