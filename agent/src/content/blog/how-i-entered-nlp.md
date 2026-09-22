---
title: "Word embeddings"
date: "2020-08-07"
category: "Personal Notes"
image: "/images/tutorials/word-embeddings.png"
tags:
  - NLP
  - Glove
  - Word2Vec
  - Bag of words
  - TF-IDF
excerpt: "The genesis of my word embeddings tutorial, and what led me to machine learning research."
---

<p class="article-lede">This is the story of how a machine-learning competition in Cameroon led me from software engineering to NLP research—and to writing my tutorial on word embeddings.</p>

See [the tutorial](/tutorials/word-embeddings/). Below is its genesis.

In 2019, I was just a programmer. At the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="nasey-note" aria-label="About NASEY">engineering school I attended</button><span id="nasey-note" class="explanation-popover" popover="auto" role="note" aria-label="National Advanced School of Engineering Yaoundé" data-label="Context">The National Advanced School of Engineering Yaoundé (NASEY) is an engineering school in Cameroon.</span></span>, we first completed two intense years of preparatory classes, with substantial mathematics, physical sciences, and algorithms. Then everyone chose a department: Computer Engineering, Electrical Engineering, Mechanical Engineering, Civil Engineering, Telecommunications Engineering, or Industrial Engineering. I chose computer science engineering and was already in the second year of that program—my fourth year at university.

When <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="james-moudie-note" aria-label="About James Assiene Moudie">James Assiene Moudie</button><span id="james-moudie-note" class="explanation-popover" popover="auto" role="note" aria-label="James Assiene Moudie" data-label="Person">James Assiene Moudie launched the first MLPC and later became a research engineer at DeepMind.</span></span> launched the first edition of <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="mlpc-note" aria-label="Explain MLPC">MLPC</button><span id="mlpc-note" class="explanation-popover" popover="auto" role="note" aria-label="Machine Learning Project Competition" data-label="Context"><strong>MLPC</strong> stands for Machine Learning Project Competition. NASEY students use machine learning to address a local problem in Africa.</span></span>, I had the idea of creating a system to translate local Cameroonian languages automatically. Unfortunately, at that time, the closest we had come to artificial intelligence at school was:

- September (Fall) 2016 to  July (Summer) 2018

    In general, all our mathematics training was useful for understanding machine learning theory: Real Analysis, Linear Algebra, Euclidean Affine Geometry, Probability and Statistics, Series and Generalized Integrals, Multilinear Algebra-Curves and Surface, Analysis in finite-dimensional vector spaces, Numerical Analyses.

    > <span class="article-kicker article-kicker--remark">Personal note</span> NDONG NGUEMA Eugène Patrice, who taught us Series and Generalized Integrals (Fall 2017) and Numerical Analysis (Winter and Summer 2017), is the best teacher I have ever known. Beyond that, he is a genius. Unfortunately, his work as a teacher in Cameroon receives far less visibility than it deserves.

- September (Fall) 2018 to July (Summer) 2019
    - Formal Systems and Foundations of Artificial Intelligence
    - Mathematical Tools for Computer Science ...
    - Science of information: (Shannon) Entropy, (huffman ...) encoding...
    - Basic mathematics: measure theory, Laplace transform ...

- September (Fall) 2019 to July (Summer) 2020:

    - Data Analysis, Theory and Practice (with Wilson Toussile, Fall 2019):
        * Statistical Learning : Parametric Estimation (Maximum likelihood estimator ...), Confidence interval ...
        * Supervised, unsupervised and semi-supervised learning formalism
        * Bayesian classification, linear and quadratic regression, bias-variance risk decomposition, cross-validation, k-nearest neighbors, K-means classification,
        * Hierarchical classification (hierarchical ascending classification...), Hard classification, Fuzzy classification, Similarity and dissimilarity measures, Clustering by mixture models (Gaussian case), EM algorithm.

    - Artificial Intelligence and Applications : Multi-Agent Systems (really old school) ...
    - Grammars and Languages: Chomsky hierarchy of grammars, Canonical automata, etc...

- September-December (Fall) 2020 :
    * Advanced Machine Learning: there was nothing advanced, the professor just took Ian Goodfellow, Yoshua Bengio and Aaron Courville's book and came to explain worse than what was in the book.
    * Image processing, GIS and WebMapping
    * Data mining: unfortunately, Professor Henri Gwet, who taught us this class (a good teacher), passed away a few months after the end of the session.

> <span class="article-kicker article-kicker--remark">Context</span> Unlike in Montreal (UdeM), where I am supposed to take two or three courses per term in Fall and Winter, we had about twenty courses per school year in Cameroon. There were no electives: we took all the courses in the program.

> <span class="article-kicker article-kicker--paper">Source</span> I have listed only the courses directly related to my learning of machine learning. The [complete list of courses](https://hackmd.io/@6LQ4mvRtS4Sc3LHkNEvDXQ/Hy_XJZ53h) is available separately.

So, in 2019, I already knew how to make computers work with real-valued vectors—for example, through linear regression—but I did not know how to make them process text. That is where I got my first taste of NLP: first with <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="classical-text-representations" aria-label="Explain GloVe, Word2Vec, bag of words, and TF-IDF">GloVe, Word2Vec, bag of words, and TF–IDF</button><span id="classical-text-representations" class="explanation-popover" popover="auto" role="note" aria-label="Classical text representations" data-label="Methods"><strong>Bag of words</strong> counts tokens; <strong>TF–IDF</strong> reweights counts by how distinctive a term is; <strong>Word2Vec</strong> and <strong>GloVe</strong> learn dense vectors whose geometry reflects patterns of word co-occurrence.</span></span>, and later with BERT and its variants. BERT had been released only recently and was attracting a great deal of attention.

If for the first approaches (GloVe, etc), understanding was quick, learning how *Transformer* works by myself until I could implement it was not an easy task for me.

> <span class="article-kicker article-kicker--remark">Learning note</span> I did not like following tutorials because I found many of them ineffective. I preferred to read the papers: difficult for a beginner, but once I understood an idea from the paper, I felt that I had really understood it. It is possible to follow ten tutorials on a concept and still miss its mechanism.

Faced with the difficulty of understanding how Transformer works :
- I went back to the original account of the <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="vanilla-attention" aria-label="Explain the original attention mechanism">attention mechanism</button><span id="vanilla-attention" class="explanation-popover" popover="auto" role="note" aria-label="Bahdanau attention" data-label="Paper">Bahdanau attention constructs a context vector as a learned weighted average of encoder states at every decoding step. See Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, <a href="https://arxiv.org/abs/1409.0473"><em>Neural Machine Translation by Jointly Learning to Align and Translate</em></a>, ICLR 2015.</span></span>.
- I read Minh-Thang Luong's PhD thesis: *NEURAL MACHINE TRANSLATION*, STANFORD UNIVERSITY, December 2016
- The *Deep Learning* book, Ian Goodfellow, Yoshua Bengio, Aaron Courville
- Coursera's Machine Learning and NLP specialization
- ...
- [Here's](https://docs.google.com/document/d/1YgnNYwWSEK1VdkEcdlwwQ89hZZDHtPlQGXfCp7BfEKw/edit?usp=drive_link) the document that lists all the papers I've read this period (briefly, from 2019 to 2022), and that helped me get started first in NLP, then in machine learning in general.

During this learning period, I wrote this [tutorial](/tutorials/word-embeddings/) on word embedding.

Back to 2020. I had completed my project, and we published [*On the Use of Linguistic Similarities to Improve Neural Machine Translation for African Languages*](/publications/african-nmt/). We proposed a dataset with parallel text for vernaculars absent from commonly used datasets such as JW300. We grouped related languages using their histories, morphologies, geographical and cultural distributions, and population migrations, and proposed a similarity metric that requires paragraph-level rather than word-level parallelism. Combining multitask learning with <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="mlm-tlm" aria-label="Explain masked and translation language modelling">MLM and TLM</button><span id="mlm-tlm" class="explanation-popover" popover="auto" role="note" aria-label="Masked and translation language modelling" data-label="Methods"><strong>Masked language modelling (MLM)</strong> predicts hidden tokens from their context. <strong>Translation language modelling (TLM)</strong> applies the same objective to aligned sentences from two languages so that information can cross the language boundary.</span></span> on clusters of similar languages substantially improved individual translation pairs. In particular, we gained <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="bleu-score" aria-label="Explain BLEU">29 BLEU points</button><span id="bleu-score" class="explanation-popover" popover="auto" role="note" aria-label="BLEU score" data-label="Metric">BLEU is an automatic machine-translation metric based mainly on clipped $n$-gram precision, with a brevity penalty. Higher is better, although human evaluation remains important.</span></span> on Bafia–Ewondo relative to earlier methods that did not exploit multilingualism.

I also completed an internship at WL Research from July to December 2020 with <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="mohamed-kane-note" aria-label="About Mohamed Hassan Kane">Mohamed Hassan Kane</button><span id="mohamed-kane-note" class="explanation-popover" popover="auto" role="note" aria-label="Mohamed Hassan Kane" data-label="Source">See the video <a href="https://youtu.be/7nHz8yhwyHg"><em>Independent AI Research in Africa: What Role for the Diaspora?</em></a></span></span>. We developed and deployed a machine-learning system that reviews <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="eula-note" aria-label="Define EULA">end-user license agreements (EULAs)</button><span id="eula-note" class="explanation-popover" popover="auto" role="note" aria-label="End-user license agreement" data-label="Definition">An EULA is the contract that specifies how an end user may use a piece of software. Our system flagged terms and conditions deemed unacceptable to the government.</span></span>. I also worked on <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="derivative-supervision" aria-label="Explain derivative-supervised learning methods">supervised learning with derivatives</button><span id="derivative-supervision" class="explanation-popover" popover="auto" role="note" aria-label="Supervised learning with derivatives" data-label="Methods">Sobolev training and differential machine learning supervise derivatives as well as function values. SIREN uses sinusoidal activations to represent signals and their derivatives accurately.</span></span>, including Sobolev training, differential machine learning, and SIREN.

I made my first contact with Mila in February 2021. Many thanks to <span class="explanation-note"><button type="button" class="explanation-trigger" popovertarget="dianbo-liu-note" aria-label="About Dianbo Liu">Dianbo Liu</button><span id="dianbo-liu-note" class="explanation-popover" popover="auto" role="note" aria-label="Dianbo Liu" data-label="Person">Dianbo Liu was a postdoctoral researcher with Prof. Yoshua Bengio and led the Humanitarian AI team at Mila–Quebec AI Institute.</span></span>, who introduced me to research.
