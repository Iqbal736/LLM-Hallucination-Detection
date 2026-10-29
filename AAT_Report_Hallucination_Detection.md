# B. M. S. COLLEGE OF ENGINEERING
### Autonomous Institute, Affiliated to VTU, Belagavi
## DEPARTMENT OF MACHINE LEARNING
### Program: B.E. In Artificial Intelligence and Machine Learning
**Academic Year:** 2026 - 27, **Session:** July 2026 - Dec 2026

---

# Natural Language Processing for LLM
### (24AM7PCLLM)
## Alternative Assessment Tool (AAT) Report

### **Topic: Automated Hallucination & Factual Consistency Detection in LLM Generations using Lexical Overlap + Logistic Regression and DistilRoBERTa Cross-Encoder**

---

### Submitted by:

| Student Name | USN | Semester & Section |
| :--- | :--- | :---: |
| **Team Member 1:** [Student Name 1] | **1BM23AI...** | **7 B** |
| **Team Member 2:** [Student Name 2] | **1BM23AI...** | **7 B** |

---

### Valuation Report
*(to be filled by faculty)*

| Score: | |
| :--- | :--- |
| **Comments:** | |
| **Faculty In-charge:** | **Dr Monika Puttaramaiah** |
| **Faculty Signature with date:** | |

\newpage

---

# Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Background of Natural Language Processing](#11-background-of-natural-language-processing)
  - [1.2 Importance of the Selected NLP Topic](#12-importance-of-the-selected-nlp-topic)
  - [1.3 Objective of the Study](#13-objective-of-the-study)
  - [1.4 Scope of the Report](#14-scope-of-the-report)
- [2. Selected NLP Topic Overview](#2-selected-nlp-topic-overview)
  - [2.1 Description of the NLP Task](#21-description-of-the-nlp-task)
  - [2.2 Real-World Use Cases](#22-real-world-use-cases)
  - [2.3 Relevance to the Syllabus and Industry](#23-relevance-to-the-syllabus-and-industry)
  - [2.4 Key Challenges in This NLP Task](#24-key-challenges-in-this-nlp-task)
- [3. Real-Time Application / API Identified](#3-real-time-application--api-identified)
  - [3.1 Application or API Name](#31-application-or-api-name)
  - [3.2 Purpose and Functionality](#32-purpose-and-functionality)
  - [3.3 Architecture or Platform on Which It Is Built](#33-architecture-or-platform-on-which-it-is-built)
  - [3.4 Input/Output Behavior](#34-inputoutput-behavior)
  - [3.5 Limitations and Practical Considerations](#35-limitations-and-practical-considerations)
- [4. NLP Model / Architecture Study](#4-nlp-model--architecture-study)
  - [4.1 Theoretical Foundations of the Model](#41-theoretical-foundations-of-the-model)
  - [4.2 Algorithm(s) Used](#42-algorithms-used)
  - [4.3 Architecture Diagram / Layer-Wise Design](#43-architecture-diagram--layer-wise-design)
  - [4.4 Data Flow in the Model](#44-data-flow-in-the-model)
  - [4.5 Mathematical Formulation](#45-mathematical-formulation)
  - [4.6 Why This Model Suits the Selected NLP Task](#46-why-this-model-suits-the-selected-nlp-task)
- [5. Dataset and Preprocessing](#5-dataset-and-preprocessing)
  - [5.1 Dataset Description (Source, Size, Structure)](#51-dataset-description-source-size-structure)
  - [5.2 Data Cleaning Steps](#52-data-cleaning-steps)
  - [5.3 Tokenization, Lemmatization, Stopwords Removal](#53-tokenization-lemmatization-stopwords-removal)
  - [5.4 Train/Test Split Strategy](#54-traintest-split-strategy)
  - [5.5 Feature Engineering or Embedding Approach](#55-feature-engineering-or-embedding-approach)
- [6. Implementation Methodology](#6-implementation-methodology)
  - [6.1 Tools, Libraries, and Environment Used](#61-tools-libraries-and-environment-used)
  - [6.2 Model Implementation Steps](#62-model-implementation-steps)
  - [6.3 API Integration](#63-api-integration)
  - [6.4 Pseudocode / Workflow Diagram](#64-pseudocode--workflow-diagram)
  - [6.5 Hyperparameters and Training Configuration](#65-hyperparameters-and-training-configuration)
- [7. Results and Evaluation](#7-results-and-evaluation)
  - [7.1 Performance Metrics](#71-performance-metrics)
  - [7.2 Visualization of Results](#72-visualization-of-results)
  - [7.3 Qualitative and Quantitative Analysis](#73-qualitative-and-quantitative-analysis)
  - [7.4 Error Analysis](#74-error-analysis)
  - [7.5 Comparison Between Approaches](#75-comparison-between-approaches)
- [8. Discussion](#8-discussion)
  - [8.1 Interpretation of Results](#81-interpretation-of-results)
  - [8.2 Strengths and Weaknesses of the Model](#82-strengths-and-weaknesses-of-the-model)
  - [8.3 Challenges Encountered During Implementation](#83-challenges-encountered-during-implementation)
  - [8.4 Impact of Dataset Quality and Preprocessing](#84-impact-of-dataset-quality-and-preprocessing)
  - [8.5 How API or Application Enhances the NLP Task](#85-how-api-or-application-enhances-the-nlp-task)
- [9. Conclusion](#9-conclusion)
  - [9.1 Summary of the Work Done](#91-summary-of-the-work-done)
  - [9.2 Key Learnings and Insights](#92-key-learnings-and-insights)
  - [9.3 Real-World Applicability](#93-real-world-applicability)
  - [9.4 Future Improvements or Extensions](#94-future-improvements-or-extensions)
- [10. References](#10-references)

\newpage

---

# 1. Introduction

### 1.1 Background of Natural Language Processing
Natural Language Processing (NLP) is the branch of Artificial Intelligence concerned with computational understanding, interpretation, and synthesis of human language. Early NLP methodologies were predominantly rule-based, leveraging handcrafted grammars, regular expressions, and curated lexicons. These systems evolved into statistical machine learning frameworks during the 1990s and 2000s, where models such as Naive Bayes, Support Vector Machines (SVM), and Logistic Regression learned discriminative boundaries over sparse vector representations including Bag-of-Words (BoW) and Term Frequency–Inverse Document Frequency (TF-IDF).

Over the past decade, the discipline underwent a paradigm shift toward continuous distributed representations—beginning with static word embeddings (Word2Vec, GloVe), advancing through recurrent architectures (RNNs, LSTMs), and culminating in the **Transformer** architecture introduced by Vaswani et al. (2017). Transformers eliminated sequential recurrence in favor of multi-head self-attention mechanisms, enabling bidirectional, parallelized contextual modeling over expansive token horizons. 

Modern Large Language Models (LLMs)—including decoder-only generative models (GPT, Llama, Gemini) and encoder-based language models (BERT, RoBERTa, DeBERTa)—now represent the frontier of modern AI. However, as generative LLMs are increasingly deployed in high-stakes domains such as healthcare, jurisprudence, and finance, a pervasive pathology has emerged: **hallucination**. Generative models frequently generate grammatically fluent, superficially authoritative text that is factually ungrounded, logically contradictory, or fabricated with respect to verified source evidence. Consequently, automated hallucination and factual consistency detection has become one of the most critical challenges in contemporary NLP for LLMs.

### 1.2 Importance of the Selected NLP Topic
Hallucination in generative AI compromises automated systems by introducing misinformation, liability risks, and incorrect decision-making. In Retrieval-Augmented Generation (RAG) architectures, search systems, and automated medical summarization pipelines, ensuring that the generated text strictly adheres to the retrieved reference documents is paramount. 

Automated factual consistency classification empowers software systems to:
1. **Serve as an Automated LLM Guardrail / Firewall:** Intercept ungrounded or fabricated generations before they are presented to end-users.
2. **Facilitate Retrieval-Augmented Generation (RAG) Filtering:** Rank, filter, and re-prompt generative pipelines when factual alignment is below an acceptable threshold.
3. **Automate Document Summarization Verification:** Audit executive, legal, and clinical summaries against lengthy original records without exhaustive human review.
4. **Mitigate Legal and Clinical Liability:** Prevent medical diagnostic misstatements, incorrect medication dosages, and fabricated legal citations (hallucinated case law).

Factual consistency verification is challenging: hallucinations frequently exhibit high lexical overlap with the reference passage, differing only by an inverted verb polarity, a swapped named entity, or an altered chronological date. Classical Bag-of-Words and surface similarity methods are easily deceived by this lexical overlap, making this an ideal domain to benchmark traditional machine learning against bidirectional Transformer cross-encoders.

### 1.3 Objective of the Study
The primary objectives of this Alternative Assessment Tool (AAT) project are:
1. To formulate and implement an end-to-end factual consistency and hallucination detection system for paired premise-hypothesis texts.
2. To build an interpretable, domain-agnostic classical machine learning baseline using **Dense Lexical Overlap Features (Unigram/Bigram Jaccard, Coverage, Negation Discrepancies) with L2-Regularized Logistic Regression**, tuned across validation folds.
3. To evaluate a deep bidirectional Transformer cross-encoder model (**DistilRoBERTa fine-tuned on Natural Language Inference**) on the exact same held-out test split under out-of-domain (OOD) conditions.
4. To conduct an exhaustive quantitative and qualitative comparison between the classical baseline and the Transformer across accuracy, precision, recall, macro F1-score, model parameter size, training cost, and single-sample inference latency.
5. To probe both models against hand-crafted adversarial rubric test cases representing subtle entity substitutions, temporal shifts, and polarity reversals.
6. To package and deploy both architectures within a unified, interactive **Streamlit web application** providing real-time side-by-side inference, agreement consensus flags, and diagnostic token-overlap metrics.

### 1.4 Scope of the Report
This report provides an in-depth examination of the theoretical foundations, implementation methodology, empirical evaluation, and qualitative error analysis of both approaches. The experimental study is scoped to binary classification: **Class 1 (Factual / Grounded / Consistent)** versus **Class 0 (Hallucinated / Unfaithful / Contradictory)** on paired context-generation instances. 

To prevent data leakage and simulate real-world generalization, the dataset is partitioned using an **Out-of-Domain (OOD) Stratified Split Strategy**: the training set spans 20 distinct scientific and historical knowledge domains (4,000 pairs), while validation (500 pairs) and held-out testing (500 pairs) evaluate performance exclusively on previously unseen domains. The Transformer model utilizes a pre-trained, NLI-specialized checkpoint (`cross-encoder/nli-distilroberta-base`, 82M parameters) evaluated in batch inference mode on local GPU/CPU hardware.

\newpage

---

# 2. Selected NLP Topic Overview

### 2.1 Description of the NLP Task
The selected NLP task is **Natural Language Inference (NLI)** (historically termed **Recognizing Textual Entailment — RTE**), formulated as a **Sentence-Pair Factual Consistency Classification** task. 

Unlike single-sequence classification (e.g., sentiment analysis or topic categorization), sentence-pair classification operates over two structurally distinct text segments:
$$\text{Input: } (C, G) \quad \text{where} \quad C = \text{Source Context / Reference Passage (Premise)}, \; G = \text{LLM Generation / Claim (Hypothesis)}$$

The objective is to learn a mapping function $f(C, G) \rightarrow y \in \{0, 1\}$ with an associated posterior confidence score $P(y \mid C, G)$:
* **Class 1 (Factual / Grounded):** The information asserted in generation $G$ is fully supported by and logically entailed by reference context $C$.
* **Class 0 (Hallucinated / Unfaithful):** Generation $G$ contains factual contradictions, entity swaps, temporal fabrications, or unsupported extrapolations that violate the ground truth in $C$.

```
+-------------------------------------------------------------------------------+
|                       Hallucination Benchmark Dataset                         |
|                    (5,000 Verified Paired Text Instances)                    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        Stratified OOD Partitioning                            |
|             Train: 4,000 Pairs | Val: 500 Pairs | Test: 500 Pairs             |
+-------------------+---------------------------------------+-------------------+
                    |                                       |
    [Model 1: Classical Lexical ML]           [Model 2: Transformer Cross-Encoder]
                    |                                       |
                    v                                       v
+---------------------------------------+ +-------------------------------------+
|        Text Normalization &           | |      Sentence-Pair Tokenization     |
|         Contraction Cleanup           | |   [CLS] Context [SEP] Claim [SEP]   |
+-------------------+-------------------+ +-----------------+-------------------+
                    |                                       |
                    v                                       v
+---------------------------------------+ +-------------------------------------+
|        Dense Overlap Extraction       | |     DistilRoBERTa Cross-Encoder     |
|    (Jaccard, N-grams, Negation)       | |   (6 layers, 12 heads, 82M params)  |
+-------------------+-------------------+ +-----------------+-------------------+
                    |                                       |
                    v                                       v
+---------------------------------------+ +-------------------------------------+
|      Logistic Regression (C=0.01)     | |        NLI Classification Head      |
|           w·x + b -> Sigmoid          | |   Softmax: Contradiction / Entail   |
+-------------------+-------------------+ +-----------------+-------------------+
                    \                                       /
                     \                                     /
                      v                                   v
+-------------------------------------------------------------------------------+
|                 Empirical Evaluation, Error Analysis & Metrics                |
|                    (Accuracy, F1, Confusion Matrices, Latency)                |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                  Interactive Streamlit Real-Time Application                  |
+-------------------------------------------------------------------------------+
```
*Figure 1: Overall workflow of the proposed factual consistency detection system*

### 2.2 Real-World Use Cases
1. **RAG Guardrail Gateways:** In enterprise search and question-answering systems (e.g., LangChain, LlamaIndex pipelines), the detector acts as an automatic validation gate: if the generated response exhibits low entailment relative to retrieved chunks, it is discarded or routed for secondary regeneration.
2. **Clinical Decision Support & EHR Summarization:** Clinical LLMs summarizing patient electronic health records (EHR) cannot tolerate hallucinations regarding dosages, drug names, or diagnoses. Automated consistency verification prevents clinical malpractice.
3. **Legal Document Discovery & Compliance:** Contract analysis tools must verify that automated clause summaries accurately reflect binding stipulations rather than fabricated indemnities.
4. **Financial Earnings Analysis:** Verifies that financial summaries accurately convey reported revenue, margins, and EBITDA numbers without hallucinating percentage changes.
5. **Content Moderation & News Aggregation:** Filters algorithmic summaries across news wires to guarantee reporting fidelity.

### 2.3 Relevance to the Syllabus and Industry
This project maps directly to the curriculum of **Natural Language Processing for LLM (24AM7PCLLM)** in the Department of Machine Learning:
* **Lexical Semantics & Vector Space Models:** Explores tokenization, n-gram extraction, Jaccard similarity, and vector normalizations.
* **Discriminative Linear Models:** Covers convex optimization, L2 regularization parameter tuning ($C$), sigmoid probability calibration, and cross-entropy loss.
* **Transformer Architectures & Self-Attention:** Formulates Scaled Dot-Product Attention, Multi-Head Attention, positional encodings, and bidirectional contextual embeddings.
* **Transfer Learning & Distillation:** Employs knowledge distillation concepts (DistilRoBERTa retaining 97% of RoBERTa's language understanding at half the parameter footprint).
* **LLM Evaluation & Alignment:** Directly addresses factuality evaluation, hallucination quantification, and inference latency trade-offs expected of modern NLP engineers in industry.

### 2.4 Key Challenges in This NLP Task
1. **The Lexical Overlap Mirage:** Hallucinated claims often borrow 80% to 95% of their tokens directly from the context. For instance, swapping only the subject (*"Louis Pasteur discovered penicillin in 1928"*) yields nearly identical bag-of-words overlap to the factual claim (*"Alexander Fleming discovered penicillin in 1928"*), severely misleading classical models.
2. **Syntactic Negation & Polarity Inversions:** Minimal morphological changes (e.g., adding *"not"*, *"failed to"*, or *"ineffective"*) reverse the logical truth value while maintaining high bag-of-words similarity.
3. **Numerical and Chronological Precision:** Dates, currency metrics, and dosages require granular token-level alignment rather than topical clustering.
4. **Out-of-Domain Generalization:** Factual consistency verifiers must generalize across completely unseen scientific and technical vocabularies without memorizing specific domain entities.
5. **Inference Latency Bottlenecks:** Cross-encoders require quadratic self-attention over concatenated sequences ($\mathcal{O}((L_1 + L_2)^2)$), creating latency constraints in high-throughput production environments.

\newpage

---

# 3. Real-Time Application / API Identified

### 3.1 Application or API Name
The project uses the **Hugging Face Transformers API** and **Model Hub** (`cross-encoder/nli-distilroberta-base`) as the underlying deep learning engine and delivers both models through a responsive, interactive **Streamlit web application** titled *Real-Time LLM Hallucination & Factuality Detection*.

### 3.2 Purpose and Functionality
The application provides an interactive audit environment where users can submit any ground-truth reference passage (premise) alongside an LLM-generated claim (hypothesis) to inspect consistency. Key capabilities include:
1. **Live Dual-Model Inference:** Generates predictions from both the Lexical Baseline and DistilRoBERTa simultaneously.
2. **Consensus & Disagreement Flagging:** Visual badges highlight whether both models agree or if an edge case has triggered a model disagreement.
3. **Lexical Diagnostic Readouts:** Exposes Jaccard token similarity, claim coverage percentages, bigram precision, and negation conflict flags.
4. **Probabilistic Confidence Meters:** Interactive progress bars displaying calibrated posterior probabilities for Factual ($P(\text{Fact})$) vs. Hallucinated ($P(\text{Hallu})$).
5. **Benchmark & Evaluation Dashboard:** Embedded tab presenting the measured empirical evaluation metrics (Table 8), confusion matrices (Figure 7), metric bar charts (Figure 8 & 9), and latency profiles (Figure 11).

### 3.3 Architecture or Platform on Which It Is Built
The project is built on Python 3.13 utilizing PyTorch with CUDA acceleration (NVIDIA RTX 3050 Laptop GPU) and scikit-learn. The codebase is organized modularly:

*Table 1: Project modules and their responsibilities*

| Module | Responsibility |
| :--- | :--- |
| `src/dataset.py` | Generates 25-domain benchmark dataset, partitions OOD train/val/test splits, generates EDA plots |
| `src/preprocessing.py` | Text normalization, contraction expansion, 8-dimensional dense lexical feature engineering |
| `src/baseline_model.py` | Hyperparameter tuning of $C$, training, and evaluation of Lexical Logistic Regression baseline |
| `src/transformer_model.py` | Loads DistilRoBERTa Cross-Encoder checkpoint, runs CUDA batch evaluation on held-out test split |
| `src/evaluation.py` | Builds comparison tables, generates Figures 7, 8, 9, 10, 11, and compiles `error_analysis.csv` |
| `src/predict.py` | `HallucinationPredictor` class: encapsulates unified inference across both architectures |
| `src/generate_diagrams.py` | Generates architectural workflow and pipeline figures (Figure 1, 2, 3) |
| `tests/test_cases.py` | Rubric test suite probing entity swaps, date shifts, and polarity reversals (Table 10) |
| `app/app.py` | Full-stack interactive Streamlit front-end application with dual view modes |

### 3.4 Input/Output Behavior
*Table 2: Input/output behavior of both models*

| Aspect | Lexical Overlap + Logistic Regression | DistilRoBERTa Cross-Encoder |
| :--- | :--- | :--- |
| **Input Format** | Paired text strings: Context $(C)$ and Claim $(G)$ | Concatenated sequence: `[CLS] Context [SEP] Claim [SEP]` |
| **Internal Representation** | 8-dimensional normalized dense overlap vector | Sequence of 768-dimensional contextualized embeddings |
| **Sequence Length** | Arbitrary length (token set operations) | Truncated/Padded to 256 tokens |
| **Output Format** | Label $\in \{\text{Hallu}, \text{Fact}\}$, $P(\text{Hallu}), P(\text{Fact})$ | Label $\in \{\text{Hallu}, \text{Fact}\}$, Softmax over 3 NLI logits |
| **Execution Hardware** | CPU (Single thread) | CUDA GPU (RTX 3050) / CPU fallback |

### 3.5 Limitations and Practical Considerations
1. **Computational Latency:** DistilRoBERTa inference requires approximately $3.37\text{ ms}$ per sample on CUDA and $\sim 45\text{ ms}$ on CPU, which is $\approx 2,592\times$ slower than the sub-millisecond ($0.0013\text{ ms}$) classical baseline.
2. **Token Truncation Horizon:** Inputs exceeding 256 subword tokens are truncated; critical facts situated at the tail of lengthy passages could be missed.
3. **Absence of Real-Time Web Knowledge Retrieval:** Both models evaluate factual grounding strictly with respect to the *provided context* rather than external open-domain world knowledge.

\newpage

---

# 4. NLP Model / Architecture Study

### 4.1 Theoretical Foundations of the Model
The classical baseline operationalizes **Lexical Semantics and Set-Theoretic Information Retrieval**. Factual grounding is framed as a function of token containment, bigram alignment, and polarity consistency. When an LLM generates a faithful statement, a substantial proportion of its constituent unigrams and bigrams must be contained within the reference context.

Conversely, the Transformer model is founded upon **Natural Language Inference (NLI)** modeled via a **Cross-Encoder Architecture**. Unlike dual-encoder architectures (Sentence-BERT) which encode premise and hypothesis into independent summary vectors and compute cosine similarity, a Cross-Encoder passes both sequences through the full transformer stack simultaneously. Every token in the generation directly attends to every token in the reference context across all attention heads and layers, capturing non-linear semantic entailment, syntactic dependencies, and contradictory entity bindings.

### 4.2 Algorithm(s) Used
*Table 3: Summary of the two algorithms*

| Property | Model 1: Classical Baseline | Model 2: Transformer Cross-Encoder |
| :--- | :--- | :--- |
| **Algorithm** | Dense Lexical Overlap + Logistic Regression | DistilRoBERTa Cross-Encoder (`cross-encoder/nli-distilroberta-base`) |
| **Model Type** | Linear Discriminative Classifier | Deep Bidirectional Transformer Encoder |
| **Features** | 8-d dense overlap vector (Jaccard, Coverage, Negation) | Word-level Byte-Pair Encoding (BPE) contextual embeddings |
| **Parameter Count** | 9 learnable parameters (8 weights + 1 bias) | $\approx 82\text{ million}$ parameters |
| **Pre-training** | None (Trained from scratch on 4,000 domain pairs) | Pre-trained on English corpora (BooksCorpus, English Wikipedia), fine-tuned on SNLI/MNLI |
| **Optimization** | L-BFGS, $\text{max\_iter}=1000$, L2 penalty ($C=0.01$) | AdamW optimizer during original fine-tuning ($\text{lr}=2\times 10^{-5}$) |

### 4.3 Architecture Diagram / Layer-Wise Design
Figure 2 outlines the classical feature extraction and classification pipeline. Figure 3 illustrates the internal layer-wise architecture of DistilRoBERTa.

```
+------------------+     +-------------------+     +---------------------+
|  Raw Text Pair   | --> | clean_text()      | --> | Dense Lexical       |
| (Context, Claim) |     | Normalization     |     | Feature Vector (8d) |
+------------------+     +-------------------+     +----------+----------+
                                                              |
                                                              v
+------------------+     +-------------------+     +----------+----------+
|  Verdict + Conf  | <-- | Logistic Reg.     | <-- | StandardScaler()    |
| (Factual / Hallu)|     | σ(w·x + b)        |     | Z-score Normalizer  |
+------------------+     +-------------------+     +---------------------+
```
*Figure 2: Lexical Overlap + Logistic Regression baseline pipeline*

DistilRoBERTa consists of an embedding layer followed by 6 Transformer encoder blocks. Each block contains a 12-head multi-head self-attention sub-layer (hidden dimension $d=768$) and a position-wise feed-forward sub-layer (intermediate dimension $d_{ff}=3072$ with GELU activation), wrapped in residual connections and layer normalization:

```
+-------------------------------------------------------------------------------+
|     Classification Head: Softmax -> P(Contradiction), P(Entailment), P(Neutral)|
+---------------------------------------+---------------------------------------+
                                        ^
+---------------------------------------+---------------------------------------+
|        [CLS] Representation (768-d) -> Pooler Dense Layer + Dropout (0.1)     |
+---------------------------------------+---------------------------------------+
                                        ^
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
'                        Transformer Encoder Block (x 6)                        '
'  +-------------------------------------------------------------------------+  '
'  |                       Add & Layer Normalization                         |  '
'  +------------------------------------+------------------------------------+  '
'                                       ^                                       '
'  +------------------------------------+------------------------------------+  '
'  |       Position-wise Feed-Forward Network (768 -> 3072 -> 768, GELU)     |  '
'  +------------------------------------+------------------------------------+  '
'                                       ^                                       '
'  +------------------------------------+------------------------------------+  '
'  |                       Add & Layer Normalization                         |  '
'  +------------------------------------+------------------------------------+  '
'                                       ^                                       '
'  +------------------------------------+------------------------------------+  '
'  |          Multi-Head Cross-Self-Attention (12 heads, 768-d)              |  '
'  +-------------------------------------------------------------------------+  '
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
                                        ^
+---------------------------------------+---------------------------------------+
|            Token Embeddings + Learned Positional Embeddings (768-d)           |
+---------------------------------------+---------------------------------------+
                                        ^
+---------------------------------------+---------------------------------------+
|     RoBERTa BPE Tokenizer -> [CLS] Context Tokens [SEP] Claim Tokens [SEP]    |
+---------------------------------------+---------------------------------------+
                                        ^
+---------------------------------------+---------------------------------------+
|           Input Text Pair: Source Context (P) and Generated Claim (H)         |
+-------------------------------------------------------------------------------+
```
*Figure 3: Layer-wise architecture of DistilRoBERTa Cross-Encoder*

### 4.4 Data Flow in the Model
* **Model 1 (Baseline):** $(C, G) \rightarrow \text{clean\_text}() \rightarrow \text{Token Tokenization} \rightarrow \mathbf{x} \in \mathbb{R}^8 \rightarrow \text{StandardScaler}() \rightarrow z = \mathbf{w}^T \mathbf{x} + b \rightarrow \sigma(z) = \frac{1}{1 + e^{-z}} \rightarrow \hat{y} = \mathbb{I}(\sigma(z) \ge 0.5)$.
* **Model 2 (Transformer):** $(C, G) \rightarrow \text{BPE Tokenizer} \rightarrow \text{Input IDs } \mathbf{x}_{1:N} \text{ and Attention Mask} \rightarrow \mathbf{E} \in \mathbb{R}^{N \times 768} \rightarrow \text{6 Transformer Encoder Blocks} \rightarrow \mathbf{h}_{[CLS]} \in \mathbb{R}^{768} \rightarrow \mathbf{W}_c \mathbf{h}_{[CLS]} + \mathbf{b}_c \rightarrow \text{Softmax}(\mathbf{z}) \rightarrow \hat{y} = \mathbb{I}(P(\text{Entail}) \ge P(\text{Contra}))$.

### 4.5 Mathematical Formulation

#### 1. Lexical Overlap Formulations:
Let $T_C$ and $T_G$ denote the unique set of unigrams in Context $C$ and Generation $G$. The unigram Jaccard coefficient $J(C, G)$ and Token Coverage (Precision) $P_{cov}(C, G)$ are defined as:
$$J(C, G) = \frac{|T_C \cap T_G|}{|T_C \cup T_G|}, \qquad P_{cov}(C, G) = \frac{|T_C \cap T_G|}{|T_G|}$$

Let $B_C$ and $B_G$ denote the set of bigram tuples. Bigram Jaccard and Bigram Precision are:
$$J_{bi}(C, G) = \frac{|B_C \cap B_G|}{|B_C \cup B_G|}, \qquad P_{bi}(C, G) = \frac{|B_C \cap B_G|}{|B_G|}$$

Negation mismatch indicator $\delta_{neg}$ where $\mathcal{N}$ is the set of explicit negation tokens:
$$\delta_{neg} = \mathbb{I}\left( (T_C \cap \mathcal{N} \neq \emptyset) \neq (T_G \cap \mathcal{N} \neq \emptyset) \right)$$

#### 2. Logistic Regression Probability and Loss:
The posterior probability of factual grounding ($y=1$) given feature vector $\mathbf{x} \in \mathbb{R}^8$ is:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

The objective function minimized during training incorporates L2 regularization parameterized by $C$:
$$\mathcal{L}(\mathbf{w}) = \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{M} \left[ -y_i \log p_i - (1 - y_i) \log(1 - p_i) \right]$$

#### 3. Scaled Dot-Product Self-Attention:
Given input projection matrices $\mathbf{Q}$ (Queries), $\mathbf{K}$ (Keys), and $\mathbf{V}$ (Values) with key dimension $d_k = 64$:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

Multi-head attention across $h=12$ heads projects representations through parameter matrices $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{768 \times 64}$ and output matrix $\mathbf{W}^O \in \mathbb{R}^{768 \times 768}$:
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_{12}) \mathbf{W}^O$$
$$\text{where } \text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V)$$

The final 3-class NLI probability distribution over $\mathcal{C} = \{\text{Contradiction}, \text{Entailment}, \text{Neutral}\}$ is computed via:
$$P(c_j \mid C, G) = \frac{e^{z_j}}{\sum_{k=1}^3 e^{z_k}}, \qquad \mathbf{z} = \mathbf{W}_c \mathbf{h}_{[CLS]} + \mathbf{b}_c$$

### 4.6 Why This Model Suits the Selected NLP Task
1. **Full Cross-Attention Resolves Entity Swaps:** In a Cross-Encoder, the entity token *"Michael Collins"* in the claim directly cross-attends to *"Neil Armstrong and Buzz Aldrin"* in the context. The attention heads detect the binding discrepancy, producing high contradiction probability.
2. **Robustness to High Lexical Overlap:** Surface word-matching baselines cannot distinguish between identical words arranged factually versus contradictory statements. DistilRoBERTa uses contextual position embeddings to identify semantic distortions.
3. **Efficiency of Distillation:** DistilRoBERTa reduces the 12 encoder layers of RoBERTa-base to 6 layers via distillation loss, cutting inference latency by $\sim 50\%$ while preserving $97\%$ of its reasoning capacity.

\newpage

---

# 5. Dataset and Preprocessing

### 5.1 Dataset Description (Source, Size, Structure)
The benchmark dataset consists of 5,000 paired text records constructed across 25 diverse scientific, historical, medical, and financial domains. Each instance comprises:
* `context`: A verified factual reference paragraph (Premise).
* `generation`: A candidate LLM-generated summary, claim, or answer (Hypothesis).
* `label`: Ground-truth binary target ($0 = \text{Hallucinated}, 1 = \text{Factual}$).
* `category`: Descriptive label string.

*Table 4: Hallucination benchmark dataset summary statistics*

| Property | Value |
| :--- | :--- |
| **Source** | Curated Multi-Domain Scientific & Historical Factuality Benchmark |
| **Total text pairs** | 5,000 paired instances |
| **Class Distribution** | Perfectly balanced: 2,500 Factual (50%), 2,500 Hallucinated (50%) |
| **Mean generation length** | 20.3 words (137 characters) |
| **Median generation length** | 20 words |
| **Shortest / longest generation** | 13 words / 26 words |
| **Mean context length** | 26.3 words (182 characters) |
| **Knowledge Domains** | 25 distinct scientific, historical, technical, and biomedical fields |

The dataset exhibits equal class representation across factual groundings and adversarial hallucinations.

### 5.2 Data Cleaning Steps
Preprocessing executed by `clean_text()` in `src/preprocessing.py` ensures standardized token boundaries:
1. Strips HTML markup, XML tags, and trailing escape characters (`\n`, `\r`, `\t`).
2. Expands contractions (`can't` $\rightarrow$ `can not`, `won't` $\rightarrow$ `will not`, `n't` $\rightarrow$ `not`) to make negation tokens explicit for unigram extraction.
3. Normalizes repeated whitespace into single space characters.
4. Preserves semantic punctuation (`.`, `!`, `?`) to demarcate clause boundaries.

### 5.3 Tokenization, Lemmatization, Stopwords Removal
1. **Stopwords Removal:** Deliberately **omitted**. Standard stopword lexicons strip negation particles (*"not"*, *"never"*, *"no"*), pronouns (*"he"*, *"she"*, *"it"*), and auxiliary verbs (*"was"*, *"did"*). In hallucination detection, removing stopwords eliminates the very words that dictate factual truth (turning *"did not land"* into *"land"*).
2. **Lemmatization:** Omitted for the baseline, as tense alterations (e.g., *"was discovered"* vs. *"will be discovered"*) constitute temporal hallucinations that lemmatization would obscure.
3. **Transformer Tokenization:** DistilRoBERTa utilizes Byte-Pair Encoding (BPE) with a 50,265 subword vocabulary. Subwords preserve out-of-vocabulary technical terminology (e.g., *"cyclooxygenase"* $\rightarrow$ `['cyclo', '##oxygen', '##ase']`).

### 5.4 Train/Test Split Strategy
To evaluate real-world generalization, an **Out-of-Domain (OOD) Partitioning Strategy** is enforced:

*Table 5: Out-of-Domain stratified train / validation / test split*

| Split | Number of Pairs | Class Balance | Domain Scope | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 4,000 | 2,000 Fact / 2,000 Hallu | 20 Base Knowledge Domains | Training Lexical Baseline features |
| **Validation** | 500 | 250 Fact / 250 Hallu | 3 Unseen Domains (Transistors, Black Holes, Plasticity) | Hyperparameter tuning of $C$ |
| **Test (Held-Out)** | 500 | 250 Fact / 250 Hallu | 2 Strictly Held-Out Domains (Base Editing, LIGO) | Final independent model evaluation |
| **Total** | **5,000** | **2,500 Fact / 2,500 Hallu** | **25 Distinct Domains** | **Zero topic overlap between Train & Test** |

Evaluating on completely unseen domains ensures that neither model relies on memorizing specific entity names.

### 5.5 Feature Engineering or Embedding Approach
* **Model 1 (Baseline):** Transforms the text pair into an 8-dimensional continuous vector capturing unigram Jaccard, claim token coverage, recall overlap, bigram Jaccard, bigram precision, negation discrepancy, length ratio, and absolute token overlap count. Vectors are standardized via z-score normalization ($z = \frac{x - \mu}{\sigma}$).
* **Model 2 (Transformer):** Maps BPE tokens to 768-dimensional learned embeddings plus position embeddings. The 6 encoder layers transform these into dynamic, contextual embeddings where token representations vary based on surrounding context.

\newpage

---

# 6. Implementation Methodology

### 6.1 Tools, Libraries, and Environment Used
*Table 6: Tools, libraries, and experimental environment*

| Category | Tool / Library | Version | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13.7 | Entire project implementation |
| **Data Handling** | pandas, NumPy | 2.3.3 / 2.4.4 | Dataframe manipulation, array operations, serialization |
| **Classical ML** | scikit-learn | 1.8.0 | Feature standardization, Logistic Regression, metric scoring |
| **Deep Learning** | PyTorch, Transformers | 2.6.0+cu124 / 5.21.0 | Model loading, CUDA GPU acceleration, cross-attention inference |
| **Visualization** | Matplotlib, Seaborn | 3.10.9 / 0.13.2 | EDA plots, confusion matrices, comparison bar charts |
| **Persistence** | joblib, JSON | 1.5.3 / Built-in | Model pipeline serialization, metric logging |
| **Deployment** | Streamlit | 1.61.1 | Interactive web application with real-time UI |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU | 6 GB VRAM | Accelerated batch tensor evaluation |

### 6.2 Model Implementation Steps
1. **Dataset Generation (`src/dataset.py`):** Generate 5,000 balanced pairs across 25 domains; split into OOD Train (4,000), Validation (500), and Test (500); produce Figures 4 and 5.
2. **Feature Preprocessing (`src/preprocessing.py`):** Implement `extract_pair_features()` calculating the 8 dense lexical signals.
3. **Baseline Training & Tuning (`src/baseline_model.py`):** Grid search $C \in \{0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$ over validation split; fit final Logistic Regression pipeline; save to `models/baseline_pipeline.joblib`.
4. **Transformer Evaluation (`src/transformer_model.py`):** Tokenize 500 test pairs; execute batch inference on CUDA under `torch.no_grad()`; record metrics, latency, and predictions.
5. **Evaluation Compilation (`src/evaluation.py`):** Merge model outputs; compute agreement metrics; compile `results/error_analysis.csv`; generate Figures 7, 8, 9, 10, 11.
6. **Rubric Probing (`tests/test_cases.py`):** Benchmark models against hand-crafted linguistic edge cases to populate Table 10.
7. **Deployment (`app/app.py`):** Launch interactive Streamlit web dashboard via `streamlit run app/app.py`.

### 6.3 API Integration
The Hugging Face Transformers API loads the tokenizer and pre-trained cross-encoder via:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base").to(device)
```
In the Streamlit application, models are wrapped in a singleton `HallucinationPredictor` class decorated with `@st.cache_resource`, ensuring network retrieval occurs once per session and weights remain memory-resident.

### 6.4 Pseudocode / Workflow Diagram

```
STEP 1: Prepare the Dataset (src/dataset.py)
1. Initialize 25 knowledge domains with factual and adversarial claims
2. Sample 4,000 pairs from 20 base domains for training (2,000 Fact, 2,000 Hallu)
3. Sample 500 pairs from 3 unseen domains for validation (250 Fact, 250 Hallu)
4. Sample 500 pairs from 2 held-out domains for testing (250 Fact, 250 Hallu)
5. Save train.csv, val.csv, and test.csv

STEP 2: Train Model 1 - Lexical Logistic Regression (src/baseline_model.py)
1. For each pair (C, G), extract 8-d feature vector:
     [Jaccard_uni, Prec_uni, Rec_uni, Jaccard_bi, Prec_bi, Neg_mismatch, Len_ratio, Overlap]
2. Scale features via StandardScaler()
3. For each C in {0.01, 0.1, 0.5, 1, 2, 5, 10}:
     train LogisticRegression on training features
     evaluate macro F1-score on validation features
4. Select optimal C = 0.01
5. Train final model; evaluate on 500 held-out test pairs; record metrics
6. Serialize pipeline to baseline_pipeline.joblib

STEP 3: Evaluate Model 2 - DistilRoBERTa Cross-Encoder (src/transformer_model.py)
1. Load checkpoint "cross-encoder/nli-distilroberta-base" onto CUDA GPU
2. Concatenate [CLS] Context [SEP] Claim [SEP]; tokenize with max_length=256
3. Run batch forward pass in batches of 16 under torch.no_grad()
4. Softmax logits -> [P(Contradiction), P(Entailment), P(Neutral)]
5. If P(Entailment) >= P(Contradiction): predict Factual (1); else Hallucinated (0)
6. Compute accuracy, macro precision, recall, F1, and per-pair inference latency

STEP 4: Cross-Model Comparison & Real-Time App (src/evaluation.py, app/app.py)
1. Merge baseline and transformer test predictions into error_analysis.csv
2. Generate confusion matrices (Figure 7), metric bar charts (Figure 8, 9),
   agreement breakdown (Figure 10), and latency curves (Figure 11)
3. In Streamlit: provide text areas for Context and Generation, execute live inference,
   render side-by-side probability gauges, and display consensus badges
```

### 6.5 Hyperparameters and Training Configuration
*Table 7: Hyperparameters and experimental configuration*

| Hyperparameter / Setting | Model 1: Lexical Baseline | Model 2: DistilRoBERTa Cross-Encoder |
| :--- | :--- | :--- |
| **Input Feature Dimension** | 8 dense lexical features | 768-dimensional token embeddings |
| **Feature Scaling** | Standard Z-score normalization | Layer Normalization ($\epsilon = 10^{-5}$) |
| **Optimization Solver** | L-BFGS | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Regularization / Weight Decay** | L2 penalty ($C=0.01$, selected via tuning) | Weight decay = 0.01 |
| **Max Sequence Length** | Unbounded | 256 subword tokens |
| **Batch Size** | Full-batch optimization | 16 pairs per batch |
| **Hardware Accelerator** | Single-thread CPU | NVIDIA GeForce RTX 3050 (6GB VRAM) |
| **Random Seed** | 42 | 42 |

\newpage

---

# 7. Results and Evaluation

### 7.1 Performance Metrics
Both models were evaluated on the exact same 500 held-out test pairs. The empirical results are summarized in Table 8:

*Table 8: Performance of both models on the held-out test set*

| Metric | Lexical + Logistic Regression | DistilRoBERTa Cross-Encoder | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **82.80%** | **94.00%** | **+11.20%** |
| **Macro Precision** | 83.16% | 94.64% | +11.48% |
| **Macro Recall** | 82.80% | 94.00% | +11.20% |
| **Macro F1-score** | 82.75% | 93.98% | +11.23% |
| **Weighted F1-score** | 82.75% | 93.98% | +11.23% |
| **Hallucinated Class (0) F1** | 81.86% | 94.34% | +12.48% |
| **Factual Class (1) F1** | 83.65% | 93.62% | +9.97% |
| **Training Time** | **0.01 seconds** | Pre-trained checkpoint | — |
| **Inference Latency** | **0.0013 ms / pair** | **3.37 ms / pair** | $\approx 2,592\times$ slower |

### 7.2 Visualization of Results

*Table 9: Confusion matrix values on 500 held-out test pairs*

| Model Architecture | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **Lexical + Logistic Regression** | 194 | 56 | 30 | 220 |
| **DistilRoBERTa Cross-Encoder** | **250** | **0** | 30 | 220 |

As detailed in Table 9, the Lexical baseline yielded **56 False Positives** (misclassifying 56 hallucinated claims as factual groundings due to high word overlap). In contrast, DistilRoBERTa produced **0 False Positives**, demonstrating strong precision on adversarial hallucinations.

### 7.3 Qualitative and Quantitative Analysis
* **Quantitative Analysis:** DistilRoBERTa outperforms the baseline across every quantitative evaluation metric. The most pronounced performance gain occurs on the **Hallucinated Class (0)**, where F1-score surges from $81.86\%$ to $94.34\%$ ($+12.48\%$). The baseline's susceptibility to false positives stems from its reliance on lexical overlap; when a hallucination reuses $80\%+$ of context tokens but alters a key entity, the baseline incorrectly flags it as factual. DistilRoBERTa eliminated all 56 false positives on the test set.
* **Qualitative Rubric Probing:** A specialized rubric test suite comprising 4 challenging cases was evaluated across both models to probe specific linguistic phenomena:

*Table 10: Rubric test cases and model predictions (confidence in brackets)*

| Case | Category | Context Excerpt | LLM Generation / Claim | Expected | Baseline (LR) | DistilRoBERTa | Agreement |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **TC-1** | Factual Paraphrase | LIGO detected gravitational waves on Sept 14, 2015 from merging black holes. | LIGO detected physical ripples in spacetime from colliding black holes in Sept 2015. | Factual | Factual (69.0%) | Factual (52.9%) | **True** |
| **TC-2** | Contradiction / Polarity Flip | Metformin decreases hepatic glucose production and increases insulin sensitivity. | Metformin significantly increases hepatic glucose production and induces severe insulin resistance. | Hallucinated | **Factual (74.5%)** *(Error)* | **Hallucinated (99.9%)** | **False** |
| **TC-3** | Entity Swap (High Overlap) | Apollo 11 landed Armstrong and Aldrin on Moon; Collins orbited. | Apollo 11 landed Neil Armstrong and Michael Collins on the lunar surface. | Hallucinated | Hallucinated (95.6%) | Hallucinated (99.3%) | **True** |
| **TC-4** | Temporal / Date Fabrication | Base editing was developed by David Liu's lab in 2016. | Base editing was developed by David Liu's lab in 1985. | Hallucinated | Hallucinated (87.1%) | Hallucinated (99.8%) | **True** |

In **TC-2**, the generation re-uses $63.6\%$ of the premise's tokens (*"metformin"*, *"hepatic"*, *"glucose"*, *"production"*, *"insulin"*). The classical baseline is misled by the token match and predicts **Factual (74.5% confidence)**. DistilRoBERTa's self-attention captures the antonymous verb shift (*"decreases"* $\rightarrow$ *"increases"*), correctly classifying it as **Hallucinated (99.9% confidence)**.

### 7.4 Error Analysis
Comparative cross-tabulation of predictions on all 500 test pairs is stored in `results/error_analysis.csv`:
* **Both Models Correct:** 414 pairs ($82.8\%$). Both architectures reliably identify standard factual groundings and low-overlap hallucinations.
* **Only DistilRoBERTa Correct:** 56 pairs ($11.2\%$). These cases correspond entirely to high-overlap adversarial hallucinations (entity and temporal swaps) where the baseline suffered false positive errors.
* **Only Baseline Correct:** 0 pairs ($0.0\%$). There were no instances where the baseline succeeded while the Transformer failed.
* **Both Models Wrong:** 30 pairs ($6.0\%$). All 30 shared errors were False Negatives (actual factual statements misclassified as hallucinations). Manual inspection reveals that these were highly abstract paraphrases where vocabulary diverged substantially from the context, causing the NLI model to allocate probability mass to the `neutral` class rather than `entailment`.

### 7.5 Comparison Between Approaches
*Table 11: Comprehensive comparison of classical and transformer approaches*

| Criterion | Lexical Overlap + Logistic Regression | DistilRoBERTa Cross-Encoder |
| :--- | :--- | :--- |
| **Test Accuracy** | 82.80% | **94.00% (Superior)** |
| **Adversarial Robustness** | Vulnerable to high-overlap hallucinations | **Highly Robust via cross-attention** |
| **Training Latency** | **0.01 seconds on CPU** | Pre-trained (Hours on GPU) |
| **Inference Speed** | **0.0013 ms / pair (Ultra-fast)** | 3.37 ms / pair on GPU (~45 ms on CPU) |
| **Model Footprint** | **< 10 KB (.joblib file)** | $\approx 320\text{ MB}$ checkpoint |
| **Interpretability** | **High:** Explicit coefficients on overlap features | **Low:** Deep neural attention weights |
| **Hardware Requirement** | Any commodity CPU | CUDA GPU recommended for batch workloads |

\newpage

---

# 8. Discussion

### 8.1 Interpretation of Results
The experimental findings illuminate the fundamental trade-off between lexical surface similarity and deep semantic cross-attention. The classical baseline's $82.80\%$ accuracy shows that domain-agnostic overlap metrics provide a useful first-pass heuristic. However, the $11.20\%$ performance gap separating the baseline from DistilRoBERTa ($94.00\%$) exposes the ceiling of bag-of-words representations. In factual consistency detection, semantic truth is governed by relation-binding rather than token presence.

DistilRoBERTa's ability to achieve zero false positives on the held-out test set demonstrates that multi-head attention successfully tracks dependencies across the sequence pair, confirming that pre-trained language models fine-tuned on NLI transfer effectively to hallucination detection.

### 8.2 Strengths and Weaknesses of the Model
*Table 12: Strengths and weaknesses of the evaluated architectures*

| Model Architecture | Strengths | Weaknesses |
| :--- | :--- | :--- |
| **Lexical + Logistic Regression** | • Near-instantaneous training ($0.01\text{ s}$)<br>• Microsecond inference ($0.0013\text{ ms}$)<br>• Ultra-lightweight footprint ($<10\text{ KB}$)<br>• Fully interpretable decision boundary | • Blind to syntactic dependencies and syntax<br>• Easily deceived by high-overlap fabrications<br>• Incapable of semantic abstraction |
| **DistilRoBERTa Cross-Encoder** | • Superior classification accuracy ($94.00\%$)<br>• Robust against subtle entity/date swaps<br>• Handles complex paraphrasing and synonymy<br>• Eliminates false positive errors | • Substantially higher latency ($3.37\text{ ms}$)<br>• Requires GPU acceleration for large batches<br>• 256-token context window truncation limit<br>• Opaque decision process (black box) |

### 8.3 Challenges Encountered During Implementation
1. **Preventing Out-of-Domain Leakage:** Initial random partitioning produced identical context topics in train and test splits, causing TF-IDF to trivially memorize specific entities. Structuring an Out-of-Domain (OOD) partition across 25 isolated domains resolved this issue and yielded realistic evaluation.
2. **The NLI Neutrality Dilemma:** Standard NLI models output three classes: *Entailment*, *Contradiction*, and *Neutral*. In generative QA, statements that are partially unmentioned fall under *Neutral*. Establishing the decision boundary $\mathbb{I}(P(\text{Entailment}) \ge P(\text{Contradiction}))$ proved critical for factual consistency scoring.
3. **Hardware Utilization:** Optimizing PyTorch batch tensors on the RTX 3050 laptop GPU reduced inference duration from $45\text{ ms}$ on CPU down to $3.37\text{ ms}$ per sample.

### 8.4 Impact of Dataset Quality and Preprocessing
Preserving negation tokens while expanding contractions was the single most impactful preprocessing decision for the baseline. Stripping stopwords or applying aggressive stemming degrades accuracy by removing the grammatical markers that convey polarity and temporal reference.

### 8.5 How API or Application Enhances the NLP Task
The Hugging Face API allowed rapid integration of an NLI cross-encoder without requiring weeks of foundational pre-training. Wrapping both pipelines within an interactive Streamlit application bridges research and deployment: users can visualize real-time consensus badges, probe diagnostic lexical signals, and inspect posterior confidence distributions.

\newpage

---

# 9. Conclusion

### 9.1 Summary of the Work Done
This AAT project designed, implemented, and empirically compared two distinct architectures for automated hallucination and factual consistency detection in LLM generations. A balanced, multi-domain benchmark comprising 5,000 paired texts across 25 scientific and historical fields was evaluated under out-of-domain conditions. An 8-dimensional Lexical Overlap baseline coupled with regularized Logistic Regression ($C=0.01$) achieved $82.80\%$ test accuracy in $0.0013\text{ ms}$ per sample. The DistilRoBERTa Cross-Encoder reached **$94.00\%$ accuracy** and **$93.98\%$ macro F1-score**, eliminating all false positives on adversarial hallucinations. Both models were unified and deployed in a functional Streamlit web application.

### 9.2 Key Learnings and Insights
1. **Lexical Overlap is Insufficient for Factuality:** High word similarity does not imply factual alignment. LLMs frequently generate adversarial hallucinations that maintain high token overlap while reversing meaning.
2. **Cross-Attention Resolves Relation Discrepancies:** By computing joint attention between context and generation tokens, cross-encoders effectively detect entity mismatches and numerical fabrications.
3. **Engineering Trade-offs:** The $11.20\%$ accuracy improvement achieved by DistilRoBERTa requires a $\approx 2,592\times$ increase in computational latency, highlighting the trade-off between deployment efficiency and verification accuracy.

### 9.3 Real-World Applicability
In production RAG pipelines, an optimal deployment architecture would leverage a **two-stage cascaded verification pipeline**:
* **Stage 1 (High-Throughput Filter):** The microsecond Lexical Baseline rapidly screens candidate generations. Pairs exhibiting clear negation conflicts or low overlap are rejected immediately at virtually zero computational cost.
* **Stage 2 (High-Precision Audit):** Borderline, high-overlap generations are routed to the DistilRoBERTa Cross-Encoder for deep semantic validation before delivery to the user.

### 9.4 Future Improvements or Extensions
1. **Long-Context Document Chunking:** Integrate sliding-window token chunking or hierarchical transformers (Longformer) to process multi-page reference documents exceeding 256 tokens.
2. **Model Quantization & ONNX Runtime Export:** Apply INT8 post-training quantization to reduce DistilRoBERTa's memory footprint and accelerate CPU inference.
3. **Span-Level Attribution:** Incorporate attention-rollout mechanisms or Integrated Gradients to highlight the exact hallucinated tokens within the user interface.

\newpage

---

# 10. References

1. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
2. Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A Robustly Optimized BERT Pretraining Approach," *arXiv preprint arXiv:1907.11692*, 2019.
3. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in *NeurIPS EMC^2 Workshop*, 2019.
4. W. Kryscinski, B. McCann, C. Xiong, and R. Socher, "Evaluating the Factual Consistency of Abstractive Text Summarization," in *Proc. of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2020, pp. 9332–9346.
5. T. Goyal and G. Durrett, "Evaluating Factuality in Generation with Dependency-level Entailment," in *Proc. of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2020, pp. 3592–3603.
6. A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding," in *Proc. of the 2018 EMNLP Workshop BlackboxNLP*, 2018, pp. 353–355.
7. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2019.
8. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
9. T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proc. of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38–45.
10. D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft, Stanford University, 2024.
11. H. Zhang et al., "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models," in *Proc. of EMNLP*, 2023.
12. Streamlit Inc., "Streamlit Documentation & Component Reference," https://docs.streamlit.io, 2026.

