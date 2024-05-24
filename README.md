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
    - [Graph Classification/Regression](#graph-classificationregression)
    - [Knowledge Graph](#knowledge-graph)
    - [Others](#others)
  - [Resources \& Tools](#resources--tools)
  - [Contributing](#contributing)
  - [Star History](#star-history)

## Datasets, Benchmarks & Surveys
- (*NAACL'21*) Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training [[paper](https://aclanthology.org/2021.naacl-main.278/)][[code](https://github.com/google-research-datasets/KELM-corpus)]
- (*NeurIPS'23*) Can Language Models Solve Graph Problems in Natural Language? [[paper](https://arxiv.org/abs/2305.10037)][[code](https://github.com/Arthur-Heng/NLGraph)]
- (*IEEE Intelligent Systems 2023*) Integrating Graphs with Large Language Models: Methods and Prospects [[paper](https://arxiv.org/abs/2310.05499)]
- (*ICLR'24*) Talk like a Graph: Encoding Graphs for Large Language Models [[paper](https://arxiv.org/abs/2310.04560)]
- (*arXiv 2023.05*) GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking [[paper](https://arxiv.org/abs/2305.15066)][[code](https://github.com/SpaceLearner/Graph-GPT)]
- (*arXiv 2023.08*) Graph Meets LLMs: Towards Large Graph Models [[paper](http://arxiv.org/abs/2308.14522)]
- (*arXiv 2023.10*) Towards Graph Foundation Models: A Survey and Beyond [[paper](https://arxiv.org/abs/2310.11829v1)]
- (*arXiv 2023.10*) LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs? [[paper](https://arxiv.org/abs/2310.17110)]
- (*arXiv 2023.11*) Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey [[paper](https://arxiv.org/abs/2311.07914v1)]
- (*arXiv 2023.11*) A Survey of Graph Meets Large Language Model: Progress and Future Directions [[paper](https://arxiv.org/abs/2311.12399)][[code](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks)]
- (*arXiv 2023.12*) Large Language Models on Graphs: A Comprehensive Survey [[paper](https://arxiv.org/abs/2312.02783)][[code](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs)]
- (*arXiv 2024.02*) Towards Versatile Graph Learning Approach: from the Perspective of Large Language Models [[paper](https://arxiv.org/abs/2402.11641)]
- (*arXiv 2024.04*) Graph Machine Learning in the Era of Large Language Models (LLMs) [[paper](https://arxiv.org/abs/2404.14928)]
- (*arXiv 2024.05*) A Survey of Large Language Models for Graphs [[paper](https://arxiv.org/abs/2405.08011)][[code](https://github.com/HKUDS/Awesome-LLM4Graph-Papers)]
  
## Prompting
- (*EMNLP'23*) StructGPT: A General Framework for Large Language Model to Reason over Structured Data [[paper](https://arxiv.org/abs/2305.09645)][[code](https://github.com/RUCAIBox/StructGPT)]
- (*AAAI'24*) Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[paper](https://arxiv.org/abs/2308.09687)][[code](https://github.com/spcl/graph-of-thoughts)]
- (*arXiv 2023.05*) PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs [[paper](https://arxiv.org/abs/2305.12392)][[code](https://github.com/Jiuzhouh/PiVe)]
- (*arXiv 2023.08*) Boosting Logical Reasoning in Large Language Models through a New Framework: The Graph of Thought [[paper](https://arxiv.org/abs/2308.08614)]
- (*arxiv 2023.10*) Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models [[paper](https://arxiv.org/abs/2310.03965v2)]
- (*arxiv 2024.01*) Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts [[paper](https://arxiv.org/abs/2401.14295)]
- (*arxiv 2024.04*) Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs [[paper](https://arxiv.org/abs/2404.07103))


## General Graph Model
- (*ICLR'24*) One for All: Towards Training One Graph Model for All Classification Tasks [[paper](https://arxiv.org/abs/2310.00149)][[code](https://github.com/LechengKong/OneForAll)]
- (*arXiv 2023.08*) Natural Language is All a Graph Needs [[paper](https://arxiv.org/abs/2308.07134)][[code](https://github.com/agiresearch/InstructGLM)]
- (*arXiv 2023.10*) GraphGPT: Graph Instruction Tuning for Large Language Models [[paper](https://arxiv.org/abs/2310.13023)][[code](https://github.com/HKUDS/GraphGPT)][[blog in Chinese](https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw)]
- (*arXiv 2023.10*) Graph Agent: Explicit Reasoning Agent for Graphs [[paper](https://arxiv.org/abs/2310.16421)]
- (*arXiv 2024.02*) Let Your Graph Do the Talking: Encoding Structured Data for LLMs [[paper](https://arxiv.org/abs/2402.05862)]
- (*arXiv 2024.02*) G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering [[paper](https://arxiv.org/abs/2402.07630)][[code](https://github.com/XiaoxinHe/G-Retriever)][[blog](https://medium.com/@xxhe/graph-retrieval-augmented-generation-rag-beb19dc30424)]
- (*arXiv 2024.02*) InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment [[paper](https://arxiv.org/abs/2402.08785)][[code](https://github.com/wjn1996/InstructGraph)]
- (*arXiv 2024.02*) LLaGA: Large Language and Graph Assistant [[paper](https://arxiv.org/abs/2402.08170)][[code](https://github.com/VITA-Group/LLaGA)]
- (*arXiv 2024.02*) HiGPT: Heterogeneous Graph Language Model [[paper](https://arxiv.org/abs/2402.16024)][[code](https://github.com/HKUDS/HiGPT)]



## Large Multimodal Models (LMMs)
- (*NeurIPS'23*) GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph [[paper](https://arxiv.org/abs/2309.13625)][[code](https://github.com/lixinustc/GraphAdapter)]
- (*arXiv 2023.10*) Multimodal Graph Learning for Generative Tasks [[paper](https://arxiv.org/abs/2310.07478)][[code](https://github.com/minjiyoon/MMGL)]
- (*arXiv 2024.02*) Rendering Graphs for Graph Reasoning in Multimodal Large Language Models [[paper](https://arxiv.org/abs/2402.02130)]


## Applications
### Basic Graph Reasoning
- (*arXiv 2023.04*) Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT [[paper](https://arxiv.org/abs/2304.11116)][[code](https://github.com/jwzhanggy/Graph_Toolformer)]
- (*arXiv 2023.10*) GraphText: Graph Reasoning in Text Space [[paper](https://arxiv.org/abs/2310.01089)]
- (*arXiv 2023.10*) GraphLLM: Boosting Graph Reasoning Ability of Large Language Model [[paper](https://arxiv.org/abs/2310.05845)][[code](https://github.com/mistyreed63849/Graph-LLM)]



### Node Classification
- (*ICLR'24*) Explanations as Features: LLM-Based Features for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2305.19523)][[code](https://github.com/XiaoxinHe/TAPE)]
- (*ICLR'24*) Label-free Node Classification on Graphs with Large Language Models (LLMS) [[paper](https://arxiv.org/abs/2310.04668)]
- (*WWW'24*) Can GNN be Good Adapter for LLMs? [[paper](https://arxiv.org/html/2402.12984v1)][[code](https://github.com/zjunet/GraphAdapter)]
- (*arXiv 2023.07*) Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs [[paper](https://arxiv.org/abs/2307.03393)][[code](https://github.com/CurryTang/Graph-LLM)]
- (*arXiv 2023.09*) Can LLMs Effectively Leverage Structural Information for Graph Learning: When and Why [[paper](https://arxiv.org/abs/2309.16595)][[code](https://github.com/TRAIS-Lab/LLM-Structured-Data)]
- (*arXiv 2023.10*) Empower Text-Attributed Graphs Learning with Large Language Models (LLMs) [[paper](https://arxiv.org/abs/2310.09872)]
- (*arXiv 2023.10*) Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2310.18152)]
- (*arXiv 2023.11*) Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2311.14324)]
- (*arXiv 2024.01*) Efficient Tuning and Inference for Large Language Models on Textual Graphs [[paper](https://arxiv.org/abs/2401.15569)][[code](https://github.com/ZhuYun97/ENGINE)]
- (*arXiv 2024.02*) Similarity-based Neighbor Selection for Graph LLMs [[paper](https://arxiv.org/abs/2402.03720)] [[code](https://github.com/ruili33/SNS)]
- (*arXiv 2024.02*) Distilling Large Language Models for Text-Attributed Graph Learning [[paper](https://arxiv.org/abs/2402.12022)]
- (*arXiv 2024.02*) GraphEdit: Large Language Models for Graph Structure Learning [[paper](https://arxiv.org/abs/2402.15183)][[code](https://github.com/HKUDS/GraphEdit?tab=readme-ov-file)]


### Graph Classification/Regression
- (*arXiv 2023.06*) GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning [[paper](https://arxiv.org/abs/2306.13089)][[code](https://github.com/zhao-ht/GIMLET)]
- (*arXiv 2023.07*) Can Large Language Models Empower Molecular Property Prediction? [[paper](https://arxiv.org/abs/2307.07443)][[code](https://github.com/ChnQ/LLM4Mol)]


### Knowledge Graph
- (*AAAI'22*) Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21286)]
- (*EMNLP'22*) Language Models of Code are Few-Shot Commonsense Learners [[paper](https://arxiv.org/abs/2210.07128)][[code](https://github.com/reasoning-machines/CoCoGen)]
- (*SIGIR'23*) Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction [[paper](https://arxiv.org/abs/2210.10709)][[code](https://github.com/zjunlp/RAP)]
- (*TKDE‘23*) AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models [[paper](https://arxiv.org/abs/2307.11772)][[code](https://github.com/ruizhang-ai/AutoAlign)]
- (*AAAI'24*) Graph Neural Prompting with Large Language Models [[paper](https://arxiv.org/abs/2309.15427)][[code](https://github.com/meettyj/GNP)]
- (*NAACL'24*) zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models [[paper](https://arxiv.org/abs/2311.10112)]
- (*ICLR'24*) Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph [[paper](https://arxiv.org/abs/2307.07697)][[code](https://github.com/IDEA-FinAI/ToG)]
- (*arXiv 2023.04*) CodeKGC: Code Language Model for Generative Knowledge Graph Construction [[paper](https://arxiv.org/abs/2304.09048)][[code](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC)]
- (*arXiv 2023.05*) Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs [[paper](https://arxiv.org/abs/2305.09858)]
- (*arXiv 2023.08*) MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models [[paper](https://arxiv.org/abs/2308.09729)][[code](https://github.com/wyl-willing/MindMap)]
- (*arXiv 2023.10*) Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph [[paper](https://arxiv.org/abs/2310.16452)]
- (*arXiv 2023.10*) Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning [[paper](https://arxiv.org/abs/2310.01061)][[code](https://github.com/RManLuo/reasoning-on-graphs)]
- (*arXiv 2023.11*) Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models [[paper](https://arxiv.org/abs/2311.10112)]
- (*arXiv 2023.12*) KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn’t Know [[paper](https://arxiv.org/abs/2312.11539)]
- (*arXiv 2024.02*) Large Language Model Meets Graph Neural Network in Knowledge Distillation [[paper](https://arxiv.org/abs/2402.05894)]
- (*arXiv 2024.02*) Large Language Models Can Learn Temporal Reasoning [[paper](https://arxiv.org/pdf/2401.06853v2.pdf)][[code](https://github.com/xiongsiheng/TG-LLM)]
- (*arXiv 2024.03*) Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments [[paper](https://arxiv.org/abs/2403.08593)]
- (*arXiv 2024.04*) Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs [[paper](https://arxiv.org/abs/2404.00942)][[code](https://github.com/xz-liu/GraphEval)]
- (*arXiv 2024.04*) Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction [[paper](https://arxiv.org/abs/2404.03868)]




### Others
- (*WSDM'24*) LLMRec: Large Language Models with Graph Augmentation for Recommendation [[paper](https://arxiv.org/abs/2311.00423)][[code](https://github.com/HKUDS/LLMRec)][[blog in Chinese](https://mp.weixin.qq.com/s/aU-uzLWH6xfIuoon-Zq8Cg)].
- (*arXiv 2023.03*) Ask and You Shall Receive (a Graph Drawing): Testing ChatGPT’s Potential to Apply Graph Layout Algorithms [[paper](https://arxiv.org/abs/2303.08819)]
- (*arXiv 2023.05*) Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding [[paper](https://arxiv.org/abs/2305.14449)]
- (*arXiv 2023.05*) ChatGPT Informed Graph Neural Network for Stock Movement Prediction [[paper](https://arxiv.org/abs/2306.03763)][[code](https://github.com/ZihanChen1995/ChatGPT-GNN-StockPredict)]
- (*arXiv 2023.10*) Graph Neural Architecture Search with GPT-4 [[paper](https://arxiv.org/abs/2310.01436)]
- (*arXiv 2023.11*) Biomedical knowledge graph-enhanced prompt generation for large language models [[paper](https://arxiv.org/abs/2311.17330)][[code](https://github.com/BaranziniLab/KG_RAG)]
- (*arXiv 2023.11*) Graph-Guided Reasoning for Multi-Hop Question Answering in Large Language Models [[paper](https://arxiv.org/abs/2311.09762)]
- (*arXiv 2024.02*) Microstructures and Accuracy of Graph Recall by Large Language Models [[paper](https://arxiv.org/abs/2402.11821)]
- (*arXiv 2024.02*) Causal Graph Discovery with Retrieval-Augmented Generation based Large Language Models [[paper](https://arxiv.org/abs/2402.15301)]
- (*arXiv 2024.02*) Graph-enhanced Large Language Models in Asynchronous Plan Reasoning [[paper](https://arxiv.org/abs/2402.02805)][[code](https://github.com/fangru-lin/graph-llm-asynchow-plan)]
- (*arXiv 2024.03*) Exploring the Potential of Large Language Models in Graph Generation [[paper](https://arxiv.org/abs/2403.08593)]


## Resources & Tools
- [GraphGPT: Extrapolating knowledge graphs from unstructured text using GPT-3](https://github.com/varunshenoy/GraphGPT)
- [GraphML: Graph markup language](https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/graphml.pdf). An XML-based file format for graphs.
- [GML: Graph modelling language](https://networkx.org/documentation/stable/reference/readwrite/gml.html). Read graphs in GML format.

## Contributing
👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- (*conference|journal*) paper_name [[pdf](link)][[code](link)]
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XiaoxinHe/Awesome-Graph-LLM&type=Date)](https://star-history.com/#XiaoxinHe/Awesome-Graph-LLM&Date)
