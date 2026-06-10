---
title: "Stochastic Average Gradient : A Simple Empirical Investigation"
authors: "Pascal Junior Tikeng Notsawo"
venue: "IFT6512, Stochastic programming, Université de Montréal"
date: "2023-01-31"
paperUrl: "https://hal.science/hal-03966263/document"
image: "/images/publications/loss-landscape.png"
tags: ["Optimization", "Stochastic Gradient", "Convergence", "SAG"]
abstract: "Despite the recent growth of theoretical studies and empirical successes of neural networks, gradient backpropagation is still the most widely used algorithm for training such networks."
---

Despite the recent growth of theoretical studies and empirical successes of neural networks, gradient backpropagation is still the most widely used algorithm for training such networks. On the one hand, we have deterministic or full gradient (FG) approaches that have a cost proportional to the amount of training data used but have a linear convergence rate, and on the other hand, stochastic gradient (SG) methods that have a cost independent of the size of the dataset, but have a less optimal convergence rate than the determinist approaches. To combine the cost of the stochastic approach with the convergence rate of the deterministic approach, a stochastic average gradient (SAG) has been proposed. SAG is a method for optimizing the sum of a finite number of smooth convex functions. Like SG methods, the SAG method's iteration cost is independent of the number of terms in the sum. In this work, we propose to compare SAG to some standard optimizers used in machine learning. SAG converges faster than other optimizers on simple toy problems and performs better than many other optimizers on simple machine learning problems. We also propose a combination of SAG with the momentum algorithm and Adam. These combinations allow empirically higher speed and obtain better performance than the other methods, especially when the landscape of the function to optimize presents obstacles or is ill-conditioned.

Paper : [hal.science/hal-03966263](https://hal.science/hal-03966263/document)
