---
title: "Grokking Finite-Dimensional Algebra"
authors: "Pascal Jr. Tikeng Notsawo, Guillaume Dumas, Guillaume Rabusseau"
venue: "Forty-Third International Conference on Machine Learning (ICML)"
date: "2026-02-23"
paperUrl: "https://arxiv.org/abs/2602.19533"
image: "/images/publications/grokking-beyond-l2.png"
tags: ["Grokking", "Algebra", "Representation Learning", "Generalization"]
abstract: "This paper investigates the grokking phenomenon, which refers to the sudden transition from a long memorization to generalization observed during neural networks training, in the context of learning multiplication in finite-dimensional algebras (FDA)."
---

This paper investigates the grokking phenomenon, which refers to the sudden transition from a long memorization to generalization observed during neural networks training, in the context of learning multiplication in finite-dimensional algebras (FDA). While prior work on grokking has focused mainly on group operations, we extend the analysis to more general algebraic structures, including non-associative, non-commutative, and non-unital algebras. We show that learning group operations is a special case of learning FDA, and that learning multiplication in FDA amounts to learning a bilinear product specified by the algebra's structure tensor. For algebras over the reals, we connect the learning problem to matrix factorization with an implicit low-rank bias, and for algebras over finite fields, we show that grokking emerges naturally as models must learn discrete representations of algebraic elements. This leads us to experimentally investigate the following core questions: (i) how do algebraic properties such as commutativity, associativity, and unitality influence both the emergence and timing of grokking, (ii) how structural properties of the structure tensor of the FDA, such as sparsity and rank, influence generalization, and (iii) to what extent generalization correlates with the model learning latent embeddings aligned with the algebra's representation. Our work provides a unified framework for grokking across algebraic structures and new insights into how mathematical structure governs neural network generalization dynamics.

Paper : [https://arxiv.org/abs/2602.19533](https://arxiv.org/abs/2602.19533)
