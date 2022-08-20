---
title: "Word embeddings"
excerpt: "<br/><img src='/images/tutorials/word_embeddings.png'>"
date: 2021-01-07
collection: portfolio
---

Code : [https://github.com/Tikquuss/word_embeddings](https://github.com/Tikquuss/word_embeddings)

## 1) **GloVe**

GloVe, coined from Global Vectors, is a model for distributed word representation. The model is an unsupervised learning algorithm for obtaining vector representations for words. The GloVe model learns word vectors by examining word co-occurrences within a text corpus.  

Let the matrix of word-word co-occurrence counts be denoted by $X$, whose entries $X_{ij}$ tabulate the number of times word $j$ occurs in the context of word $i$. Let $X_i = \sum_k X_{ik}$ be the number of times any word appears
in the context of word $i$. Finally, let $P_{ij} = X_{ij}/X_i$ be the probability that word $j$ appear in the context of word $i$.

The relationship between two words $i$ and $j$ can be examined by studying the ratio of their co-occurrence probabilities with various probe words, $k$. For words $k$ related to $i$ but not $j$, we expect the ratio $P_{ik}/P_{jk}$ will be large. Similarly, for words $k$ related to $j$ but not $i$, the ratio should be small. For words $k$ that are either related to both $i$ and $j$, or to neither, the ratio should be close to one. 

An example relating to the concepts of thermodinamics is given in the original paper with $i = ice$, $j = steam$ and $k \in \{solid, gas, water, fashion\}$

The above argument suggests that the appropriate starting point for word vector learning should be with ratios of co-occurrence probabilities rather than the probabilities themselves. Noting that the ratio $P_{ik}/P_{jk}$ depends on three words $i$, $j$, and $k$, the most general model takes the form $F(w_i, w_j,\tilde{w_{k}}) = P_{ik}/P_{jk}$ where $w \in \mathbb{R}^d$ are word vectors and $\tilde{w} \in \mathbb{R}^d$ are separate context word vectors.

To enforce $F$ to encode the information present the ratio $P_{ik}/P_{jk}$ in the word vector space, the authors restrict $F$
to depend only on the difference of the two target words $i$ and $j$, since vector spaces are inherently linear structures.
To also avoid $F$ to obfuscate the linear structure we are trying to capture as it transforms vectors into scalars, the authors passed the dot product $(w_i - w_j)^T \tilde{w}_k$ as an $F$ parameter instead of $w_i - w_j$ and $\tilde{w}_k$ themselves.

$$F((w_i - w_j)^T \tilde{w}_k) = P_{ik}/P_{jk}$$ 
$\text{ then }$ 
$$F(w_i^T \tilde{w}_k) = P_{ik} = X_{ik}/X_i \text{ (A)}$$

The authors require that $F$ be a homomorphism between the groups $(\mathbb{R},+)$ and $(\mathbb{R}_{>0}, ×)$, i.e.,

$$F((w_i - w_j)^T \tilde{w}_k) = F(w_i^T \tilde{w}_k - w_j^T \tilde{w}_k) = \frac{F(w_i^T \tilde{w}_k)}{F(w_j^T \tilde{w}_k)}$$ 
$\text{ then }$ 
$$F = exp \text{ (B)}$$

$$\text{(A) and (B)} \Rightarrow w_i^T \tilde{w}_k = log(P_{ik}) = log(X_{ik}) - log(X_i)$$
 
We will then produce vectors with a soft constraint that for each word pair of word $i$ and word $j$

$$w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log X_{ij}$$

where $b_i$ and $\tilde{b}_j$ are scalar bias terms associated with words $i$ and $j$, respectively. 

We’ll do this by minimizing an objective function $J$, which evaluates the sum of all squared errors based on the above equation, weighted with a function $f$:

$$J=\sum_{i=1}^{V} \sum_{j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

We choose an $f$ that helps prevents common word pairs (i.e., those with large $X_{ij}$ values) from skewing our objective too much:
$$
f(X_{ij}) = \left\{
    \begin{array}{ll}
        \bigg(\frac{X_{ij}}{x_{max}}\bigg)^{\alpha} & \text{if } X_{ij} \lt x_{max} \\
        1 & \text{otherwise}
    \end{array}
\right.
$$

When we encounter extremely common word pairs (where $X_{ij} \gt x_{max}$) this function will cut off its normal output and simply return $1$. For all other word pairs, we return some weight in the range $(0,1)$, where the distribution of weights in this range is decided by $\alpha$.

The authors use $x_{max} = 100 \text{ and } \alpha = 3/4$

## 2) **Word2Vec**

Word2vec is a group of related models that are used to produce word embeddings. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.

- ***Continuous Bag of Words (CBOW)*** : In the continuous bag-of-words architecture, the model predicts the current word from a window of surrounding context words. The order of context words does not influence prediction (bag-of-words assumption). 

- ***Continuous Skipgram (CS)*** : In the continuous skip-gram architecture, the model uses the current word to predict the surrounding window of context words. The skip-gram architecture weighs nearby context words more heavily than more distant context words. This other architecture tries to guess neighboring words using the current word. 

According to the authors, CBOW is faster while skip-gram is slower but does a better job for infrequent words.

For a sentence $x = x_1 ... x_T$, we want to maximize the given likelihood :

$$\mathcal{L} (\theta) = \prod_{t=1}^{T} \prod_{-m \le j \le m } f_{\theta}(w_t, w_{t+j})$$

where :

$$
f_{\theta}(w_i, w_j) = \left\{
    \begin{array}{ll}
        P_{\theta} (w_i | w_j) & \text{for CBOW} \\
        P_{\theta} (w_j | w_i) & \text{for CS}
    \end{array}
\right.
$$

Hence, our objective function can be the average negative log likelihood :
$$\mathcal{J} (\theta) = - \frac{1}{T} log(\mathcal{L} (\theta)) = - \frac{1}{T} \sum_{t=1}^{T} \sum_{-m \le j \le m } f_{\theta}(w_t, w_{t+j})$$

For each word $w$, we define two different vector representation $v_w$ and $u_w$ :
- $v_w$ is used when $w$ is a center word
- $u_w$ is used when $w$ is a context word

Hence, for a center word $c$ and a context word $o$ :

$P_{\theta} (o \| c) = \frac{e^{u_o^Tv_c}}{\sum_{w \in V} e^{u_w^Tv_c}}$

## **3) Bag of words**   

1. Find *N* most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
2. For each title in the corpora create a zero vector with the dimension equals to *N*.
3. For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.  

Drawbacks : 
- vocabulary size
- contain many 0s (thereby resulting in a sparse matrix)
- We are retaining no information on the grammar of the sentences nor on the ordering of the words in the text.


Let's try to do it for a toy example. Imagine that we have *N* = 4 and the list of the most popular words is 

    ['hi', 'you', 'me', 'are']

Then we need to numerate them, for example, like this: 

    {'hi': 0, 'you': 1, 'me': 2, 'are': 3}

And we have the text, which we want to transform to the vector:

    'hi how are you'

For this text we create a corresponding zero vector 

    [0, 0, 0, 0]
    
And iterate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:

    'hi':  [1, 0, 0, 0]
    'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
    'are': [1, 0, 0, 1]
    'you': [1, 1, 0, 1]

The resulting vector will be 

    [1, 1, 0, 1]

## 4) **TF-IDF (Term Frequency-Inverse Document Frequency)**

TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.	

- *Term Frequency (TF)* : It is a measure of how frequently a term, $t$, appears in a document, $d$:	
$$tf (t, d) = \frac{\text{number of times the term “t” appears in the document “d”}}{\text{number of terms in the document "d"}}$$

- *Inverse Document Frequency (IDF)* : IDF is a measure of how important a term is. We need the IDF value because computing just the TF alone is not sufficient to understand the importance of words.

$$idf (t) = log \bigg( \frac{\text{numbers of document}}{\text{number of document with the term "t"}} \bigg)$$

- We can now compute the TF-IDF score for each word in the corpus. Words with a higher score are more important, and those with a lower score are less important.

$$tf\_idf(t, d) = tf (t, d) * idf (t)$$

TF-IDF takes into account total frequencies of words in the corpora. It helps to penalize too frequent words and provide better features space. 

