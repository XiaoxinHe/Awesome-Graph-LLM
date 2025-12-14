# Awesome-Graph-LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME things about **Graph-Related Large Language Models (LLMs)**.

Large Language Models (LLMs) have shown remarkable progress in natural language processing tasks. However, their integration with graph structures, which are prevalent in real-world applications, remains relatively unexplored. This repository aims to bridge that gap by providing a curated list of research papers that explore the intersection of graph-based techniques with LLMs.


## Table of Contents

- [Awesome-Graph-LLM ](#awesome-graph-llm-)
  - [Table of Contents](#table-of-contents)
  - [Datasets, Benchmarks \& Surveys](#datasets-benchmarks--surveys)
  - [Prompting](#prompting)
  - [General Graph Model](#general-graph-model)
  - [Large Multimodal Models (LMMs)](#large-multimodal-models-lmms)
  - [Applications](#applications)
    - [Basic Graph Reasoning](#basic-graph-reasoning)
    - [Node Classification](#node-classification)
    - [Knowledge Graph](#knowledge-graph)
    - [Molecular Graph](#molecular-graph)
    - [Graph Retrieval Augmented Generation (GraphRAG)](#graph-retrieval-augmented-generation-graphrag)
    - [Planning](#planning)
    - [Multi-Agent Systems](#multi-agent-systems)
    - [Graph Robustness](#graph-robustness)
    - [Others](#others)
  - [Resources \& Tools](#resources--tools)
  - [Contributing](#contributing)
  - [Star History](#star-history)


## Datasets, Benchmarks & Surveys
- (*NAACL'21*) Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training [[paper](https://aclanthology.org/2021.naacl-main.278/)][[code](https://github.com/google-research-datasets/KELM-corpus)]
- (*NeurIPS'23*) Can Language Models Solve Graph Problems in Natural Language? [[paper](https://arxiv.org/abs/2305.10037)][[code](https://github.com/Arthur-Heng/NLGraph)]![GitHub Repo stars](https://img.shields.io/github/stars/Arthur-Heng/NLGraph?style=social)
- (*IEEE Intelligent Systems 2023*) Integrating Graphs with Large Language Models: Methods and Prospects [[paper](https://arxiv.org/abs/2310.05499)]
- (*ICLR'24*) Talk like a Graph: Encoding Graphs for Large Language Models [[paper](https://arxiv.org/abs/2310.04560)]
- (*NeurIPS'24*) TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs [[pdf](https://arxiv.org/abs/2406.10310)][[code](https://github.com/Zhuofeng-Li/TEG-Benchmark)][[datasets](https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/tree/main)]![GitHub Repo stars](https://img.shields.io/github/stars/Zhuofeng-Li/TEG-Benchmark?style=social)
- (*NAACL'24*) Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey [[paper](https://arxiv.org/abs/2311.07914v1)]
- (*NeurIPS'24 D&B*) Can Large Language Models Analyze Graphs like Professionals? A Benchmark, Datasets and Models [[paper](https://arxiv.org/abs/2409.19667)][[code](https://github.com/BUPT-GAMMA/ProGraph)]![GitHub Repo stars](https://img.shields.io/github/stars/BUPT-GAMMA/ProGraph?style=social)
- (*NeurIPS'24 D&B*) GLBench: A Comprehensive Benchmark for Graph with Large Language Models [[paper](https://arxiv.org/abs/2407.07457)][[code](https://github.com/NineAbyss/GLBench)]![GitHub Repo stars](https://img.shields.io/github/stars/NineAbyss/GLBench?style=social)
- (*NeurIPS'24 D&B*) DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs [[paper](https://arxiv.org/abs/2406.12072)][[code](https://github.com/zjs123/DTGB)]![GitHub Repo stars](https://img.shields.io/github/stars/zjs123/DTGB?style=social)
- (*NeurIPS'24 D&B*) UKnow: A Unified Knowledge Protocol with Multimodal Knowledge Graph Datasets for Reasoning and Vision-Language Pre-Training [[paper](https://arxiv.org/abs/2302.06891)]
- (*KDD'24*) A Survey of Large Language Models for Graphs [[paper](https://arxiv.org/abs/2405.08011)][[code](https://github.com/HKUDS/Awesome-LLM4Graph-Papers)]![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/Awesome-LLM4Graph-Papers?style=social)
- (*IJCAI'24*) A Survey of Graph Meets Large Language Model: Progress and Future Directions [[paper](https://arxiv.org/abs/2311.12399)][[code](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks)]![GitHub Repo stars](https://img.shields.io/github/stars/yhLeeee/Awesome-LLMs-in-Graph-tasks?style=social)
- (*TKDE'24*) Large Language Models on Graphs: A Comprehensive Survey [[paper](https://arxiv.org/abs/2312.02783)][[code](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs)]![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Awesome-Language-Model-on-Graphs?style=social)
- (*ICLR'25*) GraphArena: Evaluating and Exploring Large Language Models on Graph Computation [[paper](https://arxiv.org/abs/2407.00379)][[code](https://github.com/squareRoot3/GraphArena)]![GitHub Repo stars](https://img.shields.io/github/stars/squareRoot3/GraphArena?style=social)
- (*ICLR'25*) How Do Large Language Models Understand Graph Patterns? A Benchmark for Graph Pattern Comprehension [[paper](https://arxiv.org/abs/2410.05298)]
- (*arxiv 2023.05*) GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking [[paper](https://arxiv.org/abs/2305.15066)]
- (*arXiv 2023.08*) Graph Meets LLMs: Towards Large Graph Models [[paper](http://arxiv.org/abs/2308.14522)]
- (*arXiv 2023.10*) Towards Graph Foundation Models: A Survey and Beyond [[paper](https://arxiv.org/abs/2310.11829v1)]
- (*arXiv 2024.02*) Towards Versatile Graph Learning Approach: from the Perspective of Large Language Models [[paper](https://arxiv.org/abs/2402.11641)]
- (*arXiv 2024.04*) Graph Machine Learning in the Era of Large Language Models (LLMs) [[paper](https://arxiv.org/abs/2404.14928)]
- (*ICLR'25*) How Do Large Language Models Understand Graph Patterns? A Benchmark for Graph Pattern Comprehension [[paper](https://arxiv.org/abs/2410.05298v1)]
- (*arXiv 2024.10*) GRS-QA - Graph Reasoning-Structured Question Answering Dataset [[paper](https://arxiv.org/abs/2411.00369)]
- (*arXiv 2024.12*) Large Language Models Meet Graph Neural Networks: A Perspective of Graph Mining [[paper](https://arxiv.org/abs/2412.19211)]
- (*arxiv 2025.01*) Graph2text or Graph2token: A Perspective of Large Language Models for Graph Learning [[paper](https://arxiv.org/abs/2501.01124)]
- (*arXiv 2025.02*) A Comprehensive Analysis on LLM-based Node Classification Algorithms [[paper](https://arxiv.org/abs/2502.00829)] [[code](https://github.com/WxxShirley/LLMNodeBed)] [[project papge](https://llmnodebed.github.io/)]![GitHub Repo stars](https://img.shields.io/github/stars/WxxShirley/LLMNodeBed?style=social)
- (*arXiv 2025.02*) Exploring Graph Tasks with Pure LLMs: A Comprehensive Benchmark and Investigation [[paper](https://arxiv.org/abs/2502.18771)][[code](https://github.com/myflashbarry/LLM-benchmarking)]![GitHub Repo stars](https://img.shields.io/github/stars/myflashbarry/LLM-benchmarking?style=social)
- (JoWS 2025) Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspective [[paper](https://www.sciencedirect.com/science/article/pii/S1570826824000301)]
- (*arXiv 2025.05*) MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations [[paper](https://arxiv.org/abs/2505.14101)] [[code](https://github.com/ernlavr/multihal)]


## Prompting
- (*EMNLP'23*) StructGPT: A General Framework for Large Language Model to Reason over Structured Data [[paper](https://arxiv.org/abs/2305.09645)][[code](https://github.com/RUCAIBox/StructGPT)]![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/StructGPT?style=social)
- (*ACL'24*) Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs [[paper](https://arxiv.org/abs/2404.07103)][[code](https://github.com/PeterGriffinJin/Graph-CoT)]![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Graph-CoT?style=social)
- (*AAAI'24*) Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[paper](https://arxiv.org/abs/2308.09687)][[code](https://github.com/spcl/graph-of-thoughts)]
- (*arXiv 2023.05*) PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs [[paper](https://arxiv.org/abs/2305.12392)][[code](https://github.com/Jiuzhouh/PiVe)]![GitHub Repo stars](https://img.shields.io/github/stars/Jiuzhouh/PiVe?style=social)
- (*arXiv 2023.08*) Boosting Logical Reasoning in Large Language Models through a New Framework: The Graph of Thought [[paper](https://arxiv.org/abs/2308.08614)]
- (*arxiv 2023.10*) Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models [[paper](https://arxiv.org/abs/2310.03965v2)]
- (*arxiv 2024.01*) Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts [[paper](https://arxiv.org/abs/2401.14295)]
- (*arXiv 2024.10*) Can Graph Descriptive Order Affect Solving Graph Problems with LLMs? [[paper](https://arxiv.org/abs/2402.07140)]

## General Graph Model
- (*ICLR'24*) One for All: Towards Training One Graph Model for All Classification Tasks [[paper](https://arxiv.org/abs/2310.00149)][[code](https://github.com/LechengKong/OneForAll)]![GitHub Repo stars](https://img.shields.io/github/stars/LechengKong/OneForAll?style=social)
- (*ICML'24*) LLaGA: Large Language and Graph Assistant [[paper](https://arxiv.org/abs/2402.08170)][[code](https://github.com/VITA-Group/LLaGA)]![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/LLaGA?style=social)
- (NeurIPS'24) LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings [[paper](https://arxiv.org/abs/2408.14512)][[code](https://github.com/W-rudder/TEA-GLM)]![GitHub Repo stars](https://img.shields.io/github/stars/W-rudder/TEA-GLM?style=social)
- (WWW'24) GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks [[paper](https://arxiv.org/abs/2402.07197)][[code](https://github.com/alibaba/GraphTranslator)]![GitHub Repo stars](https://img.shields.io/github/stars/alibaba/GraphTranslator?style=social)
- (*KDD'24*) HiGPT: Heterogeneous Graph Language Model [[paper](https://arxiv.org/abs/2402.16024)][[code](https://github.com/HKUDS/HiGPT)]![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/HiGPT?style=social)
- (*KDD'24*) ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs [[paper](https://arxiv.org/pdf/2402.11235)][[code](https://github.com/NineAbyss/ZeroG)]![GitHub Repo stars](https://img.shields.io/github/stars/NineAbyss/ZeroG?style=social)
- (*SIGIR'24*) GraphGPT: Graph Instruction Tuning for Large Language Models [[paper](https://arxiv.org/abs/2310.13023)][[code](https://github.com/HKUDS/GraphGPT)][[blog in Chinese](https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw)]![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/GraphGPT?style=social)
- (*ACL'24*) InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment [[paper](https://arxiv.org/abs/2402.08785)][[code](https://github.com/wjn1996/InstructGraph)]![GitHub Repo stars](https://img.shields.io/github/stars/wjn1996/InstructGraph?style=social)
- (*EACL'24'*) Natural Language is All a Graph Needs [[paper](https://arxiv.org/abs/2308.07134)][[code](https://github.com/agiresearch/InstructGLM)]![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/InstructGLM?style=social)
- (*KDD'25*) UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language [[paper](https://arxiv.org/abs/2402.13630)]
- (*WSDM '25*) UniGLM: Training One Unified Language Model for Text-Attributed Graph Embedding [[paper](https://arxiv.org/abs/2406.12052)][[code](https://github.com/NYUSHCS/UniGLM)]![GitHub Repo stars](https://img.shields.io/github/stars/NYUSHCS/UniGLM?style=social)
- (*ICLR'25*) GOFA: A Generative One-For-All Model for Joint Graph Language Modeling [[paper](https://arxiv.org/abs/2407.09709)][[code](https://github.com/JiaruiFeng/GOFA)]![GitHub Repo stars](https://img.shields.io/github/stars/JiaruiFeng/GOFA?style=social)
- (*arXiv 2023.10*) Graph Agent: Explicit Reasoning Agent for Graphs [[paper](https://arxiv.org/abs/2310.16421)]
- (*arXiv 2024.02*) Let Your Graph Do the Talking: Encoding Structured Data for LLMs [[paper](https://arxiv.org/abs/2402.05862)]
- (*arXiv 2024.06*) UniGLM: Training One Unified Language Model for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2406.12052)][[code](https://github.com/NYUSHCS/UniGLM)]![GitHub Repo stars](https://img.shields.io/github/stars/NYUSHCS/UniGLM?style=social)
- (*arXiv 2024.08*) AnyGraph: Graph Foundation Model in the Wild [[paper](https://arxiv.org/abs/2408.10700)][[code](https://github.com/HKUDS/AnyGraph)]![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/AnyGraph?style=social)
- (*arXiv 2024.10*) NT-LLM: A Novel Node Tokenizer for Integrating Graph Structure into Large Language Models [[paper](https://arxiv.org/abs/2410.10743)]
- (*arXiv 2025.03*) LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models [[paper](https://arxiv.org/abs/2503.03313)]



## Large Multimodal Models (LMMs)
- (*NeurIPS'23*) GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph [[paper](https://arxiv.org/abs/2309.13625)][[code](https://github.com/lixinustc/GraphAdapter)]![GitHub Repo stars](https://img.shields.io/github/stars/lixinustc/GraphAdapter?style=social)
- (*NeurIPS'24*) GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning [[paper](https://arxiv.org/abs/2402.02130)][[code](https://github.com/WEIYanbin1999/GITA)][[project](https://v-graph.github.io/)]![GitHub Repo stars](https://img.shields.io/github/stars/WEIYanbin1999/GITA?style=social)
- (*NeurIPS'24*) GraphVis: Boosting LLMs with Visual Knowledge Graph Integration [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7cb04f510593c9ba30da398f5e0a7e7b-Paper-Conference.pdf)][[code](https://github.com/yihedeng9/GraphVis)]![GitHub Repo stars](https://img.shields.io/github/stars/yihedeng9/GraphVis?style=social)
- (*ACL 2024*) Graph Language Models [[paper](https://aclanthology.org/2024.acl-long.245/)][[code](https://github.com/Heidelberg-NLP/GraphLanguageModels)]![GitHub Repo stars](https://img.shields.io/github/stars/Heidelberg-NLP/GraphLanguageModels?style=social)
- (*WWW'25*) UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs [[paper](https://arxiv.org/abs/2502.00806)]
- (*arXiv 2023.10*) Multimodal Graph Learning for Generative Tasks [[paper](https://arxiv.org/abs/2310.07478)][[code](https://github.com/minjiyoon/MMGL)]![GitHub Repo stars](https://img.shields.io/github/stars/minjiyoon/MMGL?style=social)


## Applications
### Basic Graph Reasoning
- (*KDD'24*) GraphWiz: An Instruction-Following Language Model for Graph Problems [[paper](https://arxiv.org/abs/2402.16029)][[code](https://github.com/nuochenpku/Graph-Reasoning-LLM)][[project](https://graph-wiz.github.io/)]![GitHub Repo stars](https://img.shields.io/github/stars/nuochenpku/Graph-Reasoning-LLM?style=social)
- (*ICLR'25*) GraphArena: Evaluating and Exploring Large Language Models on Graph Computation [[paper](https://openreview.net/forum?id=Y1r9yCMzeA)] [[code](https://github.com/squareRoot3/GraphArena)]![GitHub Repo stars](https://img.shields.io/github/stars/squareRoot3/GraphArena?style=social)
- (*arXiv 2023.04*) Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT [[paper](https://arxiv.org/abs/2304.11116)][[code](https://github.com/jwzhanggy/Graph_Toolformer)]![GitHub Repo stars](https://img.shields.io/github/stars/jwzhanggy/Graph_Toolformer?style=social)
- (*arXiv 2023.10*) GraphText: Graph Reasoning in Text Space [[paper](https://arxiv.org/abs/2310.01089)]
- (*arXiv 2023.10*) GraphLLM: Boosting Graph Reasoning Ability of Large Language Model [[paper](https://arxiv.org/abs/2310.05845)][[code](https://github.com/mistyreed63849/Graph-LLM)]![GitHub Repo stars](https://img.shields.io/github/stars/mistyreed63849/Graph-LLM?style=social)
- (*arXiv 2024.10*) GUNDAM: Aligning Large Language Models with Graph Understanding [[paper](https://arxiv.org/abs/2409.20053)]
- (*arXiv 2024.10*) Are Large-Language Models Graph Algorithmic Reasoners? [[paper](https://arxiv.org/abs/2410.22597)][[code](https://github.com/ataylor24/MAGMA)]![GitHub Repo stars](https://img.shields.io/github/stars/ataylor24/MAGMA?style=social)
- (*arXiv 2024.10*) GCoder: Improving Large Language Model for Generalized Graph Problem Solving [[paper](https://arxiv.org/pdf/2410.19084)] [[code](https://github.com/Bklight999/WWW25-GCoder)]![GitHub Repo stars](https://img.shields.io/github/stars/Bklight999/WWW25-GCoder?style=social)
- (*arXiv 2024.10*) GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration [[paper](https://arxiv.org/abs/2410.18032)] [[code](https://github.com/BUPT-GAMMA/GraphTeam)]![GitHub Repo stars](https://img.shields.io/github/stars/BUPT-GAMMA/GraphTeam?style=social)


### Node Classification
- (*ICLR'24*) Explanations as Features: LLM-Based Features for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2305.19523)][[code](https://github.com/XiaoxinHe/TAPE)]![GitHub Repo stars](https://img.shields.io/github/stars/XiaoxinHe/TAPE?style=social)
- (*ICLR'24*) Label-free Node Classification on Graphs with Large Language Models (LLMS) [[paper](https://arxiv.org/abs/2310.04668)]
- (*WWW'24*) Can GNN be Good Adapter for LLMs? [[paper](https://arxiv.org/html/2402.12984v1)][[code](https://github.com/zjunet/GraphAdapter)]![GitHub Repo stars](https://img.shields.io/github/stars/zjunet/GraphAdapter?style=social)
- (*CIKM'24*) Distilling Large Language Models for Text-Attributed Graph Learning [[paper](https://arxiv.org/abs/2402.12022)]
- (*EMNLP'24*) Let's Ask GNN: Empowering Large Language Model for Graph In-Context Learning [[paper](https://arxiv.org/abs/2410.07074)]
- (*TMLR'24*) Can LLMs Effectively Leverage Graph Structural Information through Prompts, and Why? [[paper](https://arxiv.org/abs/2309.16595)][[code](https://github.com/TRAIS-Lab/LLM-Structured-Data)]![GitHub Repo stars](https://img.shields.io/github/stars/TRAIS-Lab/LLM-Structured-Data?style=social)
- (*IJCAI'24*) Efficient Tuning and Inference for Large Language Models on Textual Graphs [[paper](https://arxiv.org/abs/2401.15569)][[code](https://github.com/ZhuYun97/ENGINE)]![GitHub Repo stars](https://img.shields.io/github/stars/ZhuYun97/ENGINE?style=social)
- (*CIKM'24*) Distilling Large Language Models for Text-Attributed Graph Learning [[paper](https://arxiv.org/abs/2402.12022)]
- (*AAAI'25*) Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs [[paper](https://arxiv.org/abs/2310.09872)]
- (*WSDM'25*) LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework [[paper](https://arxiv.org/abs/2405.13902)][[code](https://github.com/QiaoYRan/LOGIN)]![GitHub Repo stars](https://img.shields.io/github/stars/QiaoYRan/LOGIN?style=social)
- (*arXiv 2023.07*) Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs [[paper](https://arxiv.org/abs/2307.03393)][[code](https://github.com/CurryTang/Graph-LLM)]![GitHub Repo stars](https://img.shields.io/github/stars/CurryTang/Graph-LLM?style=social)
- (*arXiv 2023.10*) Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2310.18152)]
- (*arXiv 2023.11*) Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2311.14324)]
- (*arXiv 2024.02*) Similarity-based Neighbor Selection for Graph LLMs [[paper](https://arxiv.org/abs/2402.03720)][[code](https://github.com/ruili33/SNS)]![GitHub Repo stars](https://img.shields.io/github/stars/ruili33/SNS?style=social)
- (*arXiv 2024.02*) GraphEdit: Large Language Models for Graph Structure Learning [[paper](https://arxiv.org/abs/2402.15183)][[code](https://github.com/HKUDS/GraphEdit)]![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/GraphEdit?style=social)
- (*arXiv 2024.06*) GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models [[paper](https://arxiv.org/abs/2406.11945)][[code](https://github.com/NYUSHCS/GAugLLM)]![GitHub Repo stars](https://img.shields.io/github/stars/NYUSHCS/GAugLLM?style=social)
- (*arXiv 2024.07*) Enhancing Data-Limited Graph Neural Networks by Actively Distilling Knowledge from Large Language Models [[paper](https://arxiv.org/abs/2407.13989)]
- (*arXiv 2024.07*) All Against Some: Efficient Integration of Large Language Models for Message Passing in Graph Neural Networks [[paper](https://arxiv.org/abs/2407.14996)]
- (*arXiv 2024.10*) Let's Ask GNN: Empowering Large Language Model for Graph In-Context Learning [[paper](https://arxiv.org/abs/2410.07074)]
- (*arXiv 2024.10*) Large Language Model-based Augmentation for Imbalanced Node Classification on Text-Attributed Graphs [[paper](https://arxiv.org/abs/2410.16882)]
- (*arXiv 2024.10*) Enhance Graph Alignment for Large Language Models [[paper](https://arxiv.org/abs/2410.11370)]
- (*arXiv 2025.01*) Each Graph is a New Language: Graph Learning with LLMs [[paper](https://arxiv.org/abs/2501.11478)]
- (*arXiv 2025.02*) A Comprehensive Analysis on LLM-based Node Classification Algorithms [[paper](https://arxiv.org/abs/2502.00829)][[code](https://github.com/WxxShirley/LLMNodeBed)] [[project papge](https://llmnodebed.github.io/)]![GitHub Repo stars](https://img.shields.io/github/stars/WxxShirley/LLMNodeBed?style=social)
- (*arXiv 2025.02*) How to Make LLMs Strong Node Classifiers? [[paper](https://arxiv.org/abs/2410.02296)]


### Knowledge Graph
- (*AAAI'22*) Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21286)]
- (*EMNLP'22*) Language Models of Code are Few-Shot Commonsense Learners [[paper](https://arxiv.org/abs/2210.07128)][[code](https://github.com/reasoning-machines/CoCoGen)]![GitHub Repo stars](https://img.shields.io/github/stars/reasoning-machines/CoCoGen?style=social)
- (*EMNLP'23*) KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion [[paper]([https://aclanthology.org/2023.findings-emnlp.580.pdf)][[code](https://github.com/WEIYanbin1999/KICGPT)]![GitHub Repo stars](https://img.shields.io/github/stars/WEIYanbin1999/KICGPT?style=social)
- (*SIGIR'23*) Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction [[paper](https://arxiv.org/abs/2210.10709)][[code](https://github.com/zjunlp/RAP)]![GitHub Repo stars](https://img.shields.io/github/stars/zjunlp/RAP?style=social)
- (*TKDE‘23*) AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models [[paper](https://arxiv.org/abs/2307.11772)][[code](https://github.com/ruizhang-ai/AutoAlign)]![GitHub Repo stars](https://img.shields.io/github/stars/ruizhang-ai/AutoAlign?style=social)
- (*ICLR'24*) Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph [[paper](https://arxiv.org/abs/2307.07697)][[code](https://github.com/IDEA-FinAI/ToG)]![GitHub Repo stars](https://img.shields.io/github/stars/IDEA-FinAI/ToG?style=social)
- (*ICLR‘24*) Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning [[paper](https://arxiv.org/abs/2310.01061)][[code](https://github.com/RManLuo/reasoning-on-graphs)]![GitHub Repo stars](https://img.shields.io/github/stars/RManLuo/reasoning-on-graphs?style=social)
- (*NeurIPS'24*) Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Model [[paper](https://arxiv.org/abs/2404.03080)][[code](https://github.com/MasterAI-EAM/Material-Knowledge-Graph)]![GitHub Repo stars](https://img.shields.io/github/stars/MasterAI-EAM/Material-Knowledge-Graph?style=social)
- (*NeurIPS'24*) KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge [[paper](https://arxiv.org/abs/2405.16412)][[code](https://github.com/pat-jj/KG-FIT)]![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/KG-FIT?style=social)
- (*NeurIPS'24*) UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction [[paper](https://arxiv.org/abs/2402.06861)][[code](https://github.com/usail-hkust/UrbanKGent)]![GitHub Repo stars](https://img.shields.io/github/stars/usail-hkust/UrbanKGent?style=social)
- (*NeurIPS'24*) Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs [[paper](https://arxiv.org/abs/2410.23875)]
- (*NeurIPS'24*) Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning [[paper](https://arxiv.org/abs/2405.14170)][[code](https://github.com/jiapuwang/LLM-DA)]![GitHub Repo stars](https://img.shields.io/github/stars/jiapuwang/LLM-DA?style=social)
- (*NeurIPS'24*) KnowGPT: Knowledge Graph based PrompTing for Large Language Models [[paper](https://arxiv.org/abs/2312.06185)]
- (*AAAI'24*) Graph Neural Prompting with Large Language Models [[paper](https://arxiv.org/abs/2309.15427)][[code](https://github.com/meettyj/GNP)]![GitHub Repo stars](https://img.shields.io/github/stars/meettyj/GNP?style=social)
- (*EMNLP'24*) Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction [[paper](https://arxiv.org/abs/2404.03868)][[code](https://github.com/clear-nus/edc)]![GitHub Repo stars](https://img.shields.io/github/stars/clear-nus/edc?style=social)
- (*EMNLP'24*) LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments [[paper](https://arxiv.org/abs/2408.15903)]
- (*ACL'24*) Graph Language Models [[paper](https://aclanthology.org/2024.acl-long.245/)][[code](https://github.com/Heidelberg-NLP/GraphLanguageModels)]![GitHub Repo stars](https://img.shields.io/github/stars/Heidelberg-NLP/GraphLanguageModels?style=social)
- (*ACL'24*) Large Language Models Can Learn Temporal Reasoning [[paper](https://arxiv.org/abs/2401.06853)][[code](https://github.com/xiongsiheng/TG-LLM)]![GitHub Repo stars](https://img.shields.io/github/stars/xiongsiheng/TG-LLM?style=social)
- (*ACL'24*) Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments [[paper](https://arxiv.org/abs/2403.08593)][[code](https://github.com/microsoft/Readi)]![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/Readi?style=social)
- (*ACL'24*) MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models [[paper](https://arxiv.org/abs/2308.09729)][[code](https://github.com/wyl-willing/MindMap)]![GitHub Repo stars](https://img.shields.io/github/stars/wyl-willing/MindMap?style=social)
- (*NAACL'24*) zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models[[paper](https://arxiv.org/abs/2311.10112)]
- (*arXiv 2023.04*) CodeKGC: Code Language Model for Generative Knowledge Graph Construction [[paper](https://arxiv.org/abs/2304.09048)][[code](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC)]
- (*arXiv 2023.05*) Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs [[paper](https://arxiv.org/abs/2305.09858)]
- (*arXiv 2023.10*) Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph [[paper](https://arxiv.org/abs/2310.16452)]
- (*arXiv 2023.12*) KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn’t Know [[paper](https://arxiv.org/abs/2312.11539)]
- (*arXiv 2024.02*) Large Language Model Meets Graph Neural Network in Knowledge Distillation [[paper](https://arxiv.org/abs/2402.05894)]
- (*arXiv 2024.02*) Knowledge Graph Large Language Model (KG-LLM) for Link Prediction [[paper](https://arxiv.org/abs/2403.07311)]
- (*arXiv 2024.04*) Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs [[paper](https://arxiv.org/abs/2404.00942)][[code](https://github.com/xz-liu/GraphEval)]![GitHub Repo stars](https://img.shields.io/github/stars/xz-liu/GraphEval?style=social)
- (*arXiv 2024.05*) FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering [[paper](https://arxiv.org/abs/2405.13873)]
- (*arXiv 2024.06*) Explore then Determine: A GNN-LLM Synergy Framework for Reasoning over Knowledge Graph [[paper](https://arxiv.org/abs/2406.01145)]
- (*arXiv 2024.11*) Synergizing LLM Agents and Knowledge Graph for Socioeconomic Prediction in LBSN [[paper](https://arxiv.org/abs/2411.00028)]
- (*arXiv 2025.01*) Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph [[paper](https://arxiv.org/abs/2501.14300)][[code](https://github.com/dosonleung/FastToG)]![GitHub Repo stars](https://img.shields.io/github/stars/dosonleung/FastToG?style=social)

### Molecular Graph
- (*NeurIPS'23*) GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning [[paper](https://arxiv.org/abs/2306.13089)][[code](https://github.com/zhao-ht/GIMLET)]![GitHub Repo stars](https://img.shields.io/github/stars/zhao-ht/GIMLET?style=social)
- (*NeurIPS'24*) LLaMo: Large Language Model-based Molecular Graph Assistant [[paper](https://arxiv.org/abs/2411.00871)][[code](https://github.com/mlvlab/LLaMo)]![GitHub Repo stars](https://img.shields.io/github/stars/mlvlab/LLaMo?style=social)
- (*arXiv 2023.07*) Can Large Language Models Empower Molecular Property Prediction? [[paper](https://arxiv.org/abs/2307.07443)][[code](https://github.com/ChnQ/LLM4Mol)]![GitHub Repo stars](https://img.shields.io/github/stars/ChnQ/LLM4Mol?style=social)
- (*arXiv 2024.06*) MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction [[paper](https://arxiv.org/abs/2406.12950)][[code](https://github.com/NYUSHCS/MolecularGPT)]![GitHub Repo stars](https://img.shields.io/github/stars/NYUSHCS/MolecularGPT?style=social)
- (*arXiv 2024.06*) HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment [[paper](https://arxiv.org/abs/2406.14021)][[project](https://higraphllm.github.io/)]
- (*arXiv 2024.06*) MolX: Enhancing Large Language Models for Molecular Learning with A Multi-Modal Extension [[paper](https://arxiv.org/abs/2406.06777)]
- (*arXiv 2024.06*) LLM and GNN are Complementary: Distilling LLM for Multimodal Graph Learning [[paper](https://arxiv.org/abs/2406.01032)]
- (*arXiv 2024.10*) G2T-LLM: Graph-to-Tree Text Encoding for Molecule Generation with Fine-Tuned Large Language Models [[paper](https://arxiv.org/abs/2410.02198v1)]

### Graph Retrieval Augmented Generation (GraphRAG)
- (*NeurIPS'24*) G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering [[paper](https://arxiv.org/abs/2402.07630)][[code](https://github.com/XiaoxinHe/G-Retriever)][[blog](https://medium.com/@xxhe/graph-retrieval-augmented-generation-rag-beb19dc30424)]![GitHub Repo stars](https://img.shields.io/github/stars/XiaoxinHe/G-Retriever?style=social)
- (*NeurIPS'24*) HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models [[paper](https://arxiv.org/abs/2405.14831)][[code](https://github.com/OSU-NLP-Group/HippoRAG)]![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-Group/HippoRAG?style=social)
- (*ICLR'25*) Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2410.20724)][[code](https://github.com/Graph-COM/SubgraphRAG)]![GitHub Repo stars](https://img.shields.io/github/stars/Graph-COM/SubgraphRAG?style=social)
- (*WWW'25*) G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation [[paper](https://openreview.net/forum?id=JSSeMdhsye)]
- (*arxiv 2024.04*) From Local to Global: A Graph RAG Approach to Query-Focused Summarization [[paper](https://arxiv.org/abs/2404.16130)]
- (*arXiv 2024.05*) Don't Forget to Connect! Improving RAG with Graph-based Reranking [[paper](https://arxiv.org/abs/2405.18414)]
- (*arXiv 2024.06*) GNN-RAG: Graph Neural Retrieval for Large Language Modeling Reasoning [[paper](https://arxiv.org/abs/2405.20139)][[code](https://github.com/cmavro/GNN-RAG)]![GitHub Repo stars](https://img.shields.io/github/stars/cmavro/GNN-RAG?style=social)
- (*arXiv 2024.08*) Graph Retrieval-Augmented Generation: A Survey [[paper](https://arxiv.org/abs/2408.08921)][[code](https://github.com/pengboci/GraphRAG-Survey)]![GitHub Repo stars](https://img.shields.io/github/stars/pengboci/GraphRAG-Survey?style=social)
- (*arXiv 2024.10*) Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs [[paper](https://arxiv.org/abs/2410.11001)] [[code](https://github.com/ulab-uiuc/GoR)]![GitHub Repo stars](https://img.shields.io/github/stars/ulab-uiuc/GoR?style=social)
- (*arXiv 2025.01*) Retrieval-Augmented Generation with Graphs (GraphRAG) [[paper](https://arxiv.org/pdf/2501.00309)][[code](https://github.com/Graph-RAG/GraphRAG/)]![GitHub Repo stars](https://img.shields.io/github/stars/Graph-RAG/GraphRAG?style=social)
- (*arXiv 2025.01*) A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models [[paper](https://arxiv.org/abs/2501.13958)][[code](https://github.com/DEEP-PolyU/Awesome-GraphRAG)]![GitHub Repo stars](https://img.shields.io/github/stars/DEEP-PolyU/Awesome-GraphRAG?style=social)
- (*arXiv 2025.02*) GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation [[paper](https://arxiv.org/abs/2502.01113)] [[code](https://github.com/RManLuo/gfm-rag)]![GitHub Repo stars](https://img.shields.io/github/stars/RManLuo/gfm-rag?style=social)
- (*arXiv 2025.02*) Are Large Language Models In-Context Graph Learners? [[paper](https://arxiv.org/pdf/2502.13562)]
- (*arXiv 2025.02*) ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2502.09891)]
- (*arXiv 2025.02*) BioMaze: Benchmarking and Enhancing Large Language Models for Biological Pathway Reasoning [[paper](https://arxiv.org/abs/2502.16660)][[code](https://github.com/zhao-ht/BioMaze)]![GitHub Repo stars](https://img.shields.io/github/stars/zhao-ht/BioMaze?style=social)
- (*arXiv 2025.03*) Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning [[paper](https://arxiv.org/abs/2503.01642)]
- (*arXiv 2025.03*) In-depth Analysis of Graph-based RAG in a Unified Framework [[paper](https://www.arxiv.org/abs/2503.04338)][[code](https://github.com/JayLZhou/GraphRAG)]![GitHub Repo stars](https://img.shields.io/github/stars/JayLZhou/GraphRAG?style=social)



### Planning
- (*NeurIPS'24*) Can Graph Learning Improve Planning in LLM-based Agents? [[paper](https://arxiv.org/abs/2405.19119)][[code](https://github.com/WxxShirley/GNN4TaskPlan)]![GitHub Repo stars](https://img.shields.io/github/stars/WxxShirley/GNN4TaskPlan?style=social)
- (*ICML'24*) Graph-enhanced Large Language Models in Asynchronous Plan Reasoning [[paper](https://arxiv.org/abs/2402.02805)][[code](https://github.com/fangru-lin/graph-llm-asynchow-plan)]![GitHub Repo stars](https://img.shields.io/github/stars/fangru-lin/graph-llm-asynchow-plan?style=social)
- (*ICLR'25*) Benchmarking Agentic Workflow Generation [[paper](https://arxiv.org/abs/2410.07869)] [[code](https://github.com/zjunlp/WorFBench)]![GitHub Repo stars](https://img.shields.io/github/stars/zjunlp/WorFBench?style=social)

### Multi-Agent Systems
- (*ICML'24*) GPTSwarm: Language Agents as Optimizable Graphs [[paper](https://arxiv.org/abs/2402.16823)] [[code](https://github.com/metauto-ai/GPTSwarm)]![GitHub Repo stars](https://img.shields.io/github/stars/metauto-ai/GPTSwarm?style=social)
- (*ICLR'25*) Scaling Large-Language-Model-based Multi-Agent Collaboration [[paper](https://arxiv.org/abs/2406.07155)] [[code](https://github.com/OpenBMB/ChatDev)]![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/ChatDev?style=social)
- (*ICLR'25*) Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems [[paper](https://arxiv.org/abs/2410.02506)] [[code](https://github.com/yanweiyue/AgentPrune)]![GitHub Repo stars](https://img.shields.io/github/stars/yanweiyue/AgentPrune?style=social)
- (*arXiv 2024.10*) G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks [[paper](https://arxiv.org/abs/2410.11782)] [[code](https://github.com/yanweiyue/GDesigner)]![GitHub Repo stars](https://img.shields.io/github/stars/yanweiyue/GDesigner?style=social)
- (*arXiv 2025.02*) EvoFlow: Evolving Diverse Agentic Workflows On The Fly [[paper](https://arxiv.org/abs/2502.07373)]
- (*arXiv 2025.02*) MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications [[paper](https://arxiv.org/abs/2502.18540)][[code](https://github.com/ZIKEYUAN/MA-GTS)]![GitHub Repo stars](https://img.shields.io/github/stars/ZIKEYUAN/MA-GTS?style=social)
  

### Graph Robustness
- (*NeurIPS'24*) Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level [[paper](https://arxiv.org/abs/2405.16405)]
- (*KDD'25*) Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks? [[paper](https://arxiv.org/pdf/2408.08685)][[code](https://github.com/zhongjian-zhang/LLM4RGNN)]![GitHub Repo stars](https://img.shields.io/github/stars/zhongjian-zhang/LLM4RGNN?style=social)
- (*arXiv 2024.07*) Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness [[paper](https://arxiv.org/abs/2407.12068)][[code](https://github.com/KaiGuo20/GraphLLM_Robustness)]![GitHub Repo stars](https://img.shields.io/github/stars/KaiGuo20/GraphLLM_Robustness?style=social)

### Others
- (*NeurIPS'24*) Microstructures and Accuracy of Graph Recall by Large Language Models [[paper](https://arxiv.org/abs/2402.11821)][[code](https://github.com/Abel0828/llm-graph-recall)]![GitHub Repo stars](https://img.shields.io/github/stars/Abel0828/llm-graph-recall?style=social)
- (*WSDM'24*) LLMRec: Large Language Models with Graph Augmentation for Recommendation [[paper](https://arxiv.org/abs/2311.00423)][[code](https://github.com/HKUDS/LLMRec)][[blog in Chinese](https://mp.weixin.qq.com/s/aU-uzLWH6xfIuoon-Zq8Cg)]![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/LLMRec?style=social)
- (*KDD'24*) LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs? [[paper](https://arxiv.org/abs/2310.17110)][[code](https://github.com/wondergo2017/LLM4DyG)]![GitHub Repo stars](https://img.shields.io/github/stars/wondergo2017/LLM4DyG?style=social)
- (*Complex Networks 2024*) LLMs hallucinate graphs too: a structural perspective [[paper](https://arxiv.org/abs/2409.00159)]
- (AAAI'25) Bootstrapping Heterogeneous Graph Representation Learning via Large Language Models: A Generalized Approach [[paper](https://arxiv.org/abs/2412.08038)]
- (*ICLR'25*) RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph [[paper](https://arxiv.org/abs/2410.14684)] [[code](https://github.com/ozyyshr/RepoGraph)]![GitHub Repo stars](https://img.shields.io/github/stars/ozyyshr/RepoGraph?style=social)
- (*ICLR'25*) GraphRouter: A Graph-based Router for LLM Selections [[paper](https://arxiv.org/abs/2410.03834)][[code](https://github.com/ulab-uiuc/GraphRouter)]![GitHub Repo stars](https://img.shields.io/github/stars/ulab-uiuc/GraphRouter?style=social)
- (*arXiv 2023.03*) Ask and You Shall Receive (a Graph Drawing): Testing ChatGPT’s Potential to Apply Graph Layout Algorithms [[paper](https://arxiv.org/abs/2303.08819)]
- (*arXiv 2023.05*) Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding [[paper](https://arxiv.org/abs/2305.14449)]
- (*arXiv 2023.05*) ChatGPT Informed Graph Neural Network for Stock Movement Prediction [[paper](https://arxiv.org/abs/2306.03763)][[code](https://github.com/ZihanChen1995/ChatGPT-GNN-StockPredict)]![GitHub Repo stars](https://img.shields.io/github/stars/ZihanChen1995/ChatGPT-GNN-StockPredict?style=social)
- (*arXiv 2023.10*) Graph Neural Architecture Search with GPT-4 [[paper](https://arxiv.org/abs/2310.01436)]
- (*arXiv 2023.11*) Biomedical knowledge graph-enhanced prompt generation for large language models [[paper](https://arxiv.org/abs/2311.17330)][[code](https://github.com/BaranziniLab/KG_RAG)]![GitHub Repo stars](https://img.shields.io/github/stars/BaranziniLab/KG_RAG?style=social)
- (*arXiv 2023.11*) Graph-Guided Reasoning for Multi-Hop Question Answering in Large Language Models [[paper](https://arxiv.org/abs/2311.09762)]
- (*arXiv 2024.02*) Causal Graph Discovery with Retrieval-Augmented Generation based Large Language Models [[paper](https://arxiv.org/abs/2402.15301)]
- (*arXiv 2024.02*) Efficient Causal Graph Discovery Using Large Language Models [[paper](https://arxiv.org/abs/2402.01207)]
- (*arXiv 2024.03*) Exploring the Potential of Large Language Models in Graph Generation [[paper](https://arxiv.org/abs/2403.14358)]
- (*arXiv 2024.07*) LLMExplainer: Large Language Model based Bayesian Inference for Graph Explanation Generation [[paper](https://arxiv.org/abs/2407.15351)]
- (*arXiv 2024.08*) CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases [[paper](https://arxiv.org/abs/2408.03910)][[code](https://github.com/modelscope/modelscope-agent/tree/master/apps/codexgraph_agent)][[project](https://laptype.github.io/CodexGraph-page/)]
- (*arXiv 2024.10*) Graph Linearization Methods for Reasoning on Graphs with Large Language Models [[paper](https://arxiv.org/abs/2410.19494)]

## Resources & Tools
- [GraphGPT: Extrapolating knowledge graphs from unstructured text using GPT-3](https://github.com/varunshenoy/GraphGPT)
- [GraphML: Graph markup language](https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/graphml.pdf). An XML-based file format for graphs.
- [GML: Graph modelling language](https://networkx.org/documentation/stable/reference/readwrite/gml.html). Read graphs in GML format.
- [PyG: GNNs + LLMs](https://github.com/pyg-team/pytorch_geometric/tree/master/examples/llm): Examples for Co-training LLMs and GNNs

## Contributing
👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- (*conference|journal*) paper_name [[pdf](link)][[code](link)]
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XiaoxinHe/Awesome-Graph-LLM&type=Date)](https://star-history.com/#XiaoxinHe/Awesome-Graph-LLM&Date)
