---
title: "Quantifying Overclaiming Propensity in Frontier LLM Agents"
authors: "Nolan Smyth, Yorguin-Jose Mantilla-Ramos, Pascal Jr Tikeng Notsawo, Saskia Helbling, Alberto Tosato, Mohamed Amine Merzouk, Nouha Dziri, Gauthier Gidel, Tommaso Tosato"
authorContributions:
  lead:
    - "Nolan Smyth"
    - "Yorguin-Jose Mantilla-Ramos"
    - "Tommaso Tosato"
  core:
    - "Pascal Jr Tikeng Notsawo"
    - "Saskia Helbling"
    - "Alberto Tosato"
  highlighted:
    - "Pascal Jr Tikeng Notsawo"
venues:
  - "Preprint, under review"
date: "2026-09-20"
paperUrl: "https://arxiv.org/abs/2609.20812"
image: "/images/publications/overclaiming-overview.png"
tags: ["AI Safety", "Overclaiming", "Benchmarks", "LLM Agents", "Agent Evaluation"]
abstract: Frontier coding agents often claim to have reviewed every file when they have not, leaving critical issues they missed unreported.
---

## Abstract

Frontier coding agents are increasingly trusted to work autonomously for long periods, yet an agent's final response is often the only account of that work a user sees. We quantify the propensity of frontier agents to *overclaim* task completion, a misrepresentation that can mislead the user. An agent overclaims when its final response contradicts information in its context. This definition requires no inference about intent and is independent of task success.
We introduce *OverclaimBench*, an evaluation suite composed of five file-review scenarios, transcript-based coverage measurements, and registered planted defects. We evaluate eight proprietary frontier models in their own production command-line interfaces, and four open-weight models under a single fixed harness on OverclaimBench and find that: (1) agents do not read all the files they were asked to review in 67.9% of runs; (2) among runs where not all files are read, agents are *misleading* 80.4% of the time (59-96% per model), either falsely claiming to have read all files or omitting that coverage is incomplete; (3) requiring delegation to subagents increased reading coverage, but among reviews that remained incomplete, a large majority were still misleading; and (4) agents that falsely claimed a complete review missed planted defects at about 1.8 times the rate of agents that read every file, showing that claims of completion can conceal substantive failures. Together, these results show that agents' final responses are not reliable accounts of their actions.
