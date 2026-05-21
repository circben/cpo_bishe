# 第八章 总结与展望

---

## 8.1 工作总结

本课题围绕链式偏好优化（CPO）范式，设计并实现了一套完整的推理增强系统工程方案。具体工作总结如下：

**（1）五阶段自动化管道架构**。设计了Stage 1数据准备→Stage 2 ToT搜索与偏好构建→Stage 3模型训练→Stage 4多基线评估→Stage 5 Web可视化的完整管道，各阶段通过JSON/JSONL解耦通信，配置通过YAML集中管理。管道支持任意阶段独立运行，各模块代码以`src/cpo/`为核心按功能分层组织。

**（2）SC-ToT增强搜索算法**。针对标准ToT的评分机制单一、搜索策略固定、终端缺乏验证三个局限，提出了三项增强：双向评分综合步骤质量与状态预期、全局束选择避免局部归一化偏差、终端自洽性投票抑制随机误差。增强搜索为偏好数据构建提供了更高质量的信号源。

**（3）步骤级偏好数据自动构建与质量过滤**。从搜索树回溯成功路径，提取步骤级偏好对，通过评分阈值和语义相似度过滤进行质量控制，有效剔除低区分度和冗余样本。

**（4）DPO/TS-SFT双轨训练方案**。CPO路线利用步骤级偏好数据使模型学会在每个推理分叉点选择更优方向，TS-SFT路线作为对比基线验证偏好信号的独特价值。两条路线共享基座模型和LoRA配置，确保对比公平。

**（5）多基线统一评估与消融实验框架**。在GSM8K和StrategyQA上系统对比SC-ToT、CoT、TS-SFT和CPO四种方法的准确率和推理延迟。设计并执行了SC-ToT组件消融、超参数敏感度分析和训练数据规模消融等多维度消融实验。

**（6）交互式Web可视化平台**。基于FastAPI和D3.js构建了四大可视化模块：思维树浏览器、性能对比仪表盘、统计报告页面和消融分析视图，为实验结果分析提供了直观的交互式工具。

## 8.2 核心结论

通过本课题的研究，可得出以下核心结论：

（1）CPO范式有效实现了搜索质量与推理效率的兼得——以离线训练增加计算开销换取在线推理时保持单次CoT解码的高效率，推理延迟仅为搜索方法的百分之一而准确率接近。

（2）步骤级的偏好信号优于路径级的监督信号。DPO的逐步骤偏好建模使得模型在每个推理决策点都能学到精细的"选优去劣"能力，而整条路径的SFT则无法传递这种步骤级指导。

（3）SC-ToT的三项增强贡献显著且互补，其中终端自洽性投票贡献最大，验证了LLM终端决策的随机性问题是推理准确率的瓶颈之一。

（4）偏好数据的质量（区分度）比数量更重要。评分差距大、语义差异明显的偏好对提供了更强的训练信号，低质量的偏好对不仅无益，甚至可能引入噪声。

## 8.3 展望

在现有工作基础上，以下方向值得进一步探索：

**（1）更高效的在线搜索策略**。当前SC-ToT每个问题的135次LLM调用在计算成本和API费用上仍较高。引入A*搜索使用启发式函数替代全量状态评估、基于小模型的快速预筛、或者学习式剪枝策略，有望在保持搜索质量的同时大幅降低LLM调用次数。

**（2）迭代式CPO训练**。当前CPO训练为单轮次——使用初始模型进行搜索和偏好提取后训练一次。若采用多轮迭代：第一轮训练后的模型→用新模型进行ToT搜索→提取新的、更高质量的偏好数据→第二轮训练……这一"自举式"过程有望形成正向增强循环。

**（3）跨推理域的泛化验证**。将CPO框架扩展到数学推理和常识推理之外的任务类型——代码生成、法律推理、医学诊断等——验证偏好优化范式在不同推理场景中的有效性和通用性。

**（4）结合外部验证器的评分机制**。当前评分完全依赖LLM自身，存在自我偏袒的风险。将代码执行器（如Python解释器）引入数学推理步骤的评分——对数值步骤执行实际计算并与推理结果对照——可提供更客观的评分信号。

**（5）偏好信号的可解释性研究**。通过分析CPO训练前后模型在特定决策点的注意力分布和隐状态变化，理解偏好优化在模型内部表征层面的作用机制，为后续算法改进提供理论指导。

---

## Abstract

This thesis presents a comprehensive system for enhancing large language model (LLM) reasoning capabilities through the Chain of Preference Optimization (CPO) paradigm. The core idea is to transfer the computational cost of tree-search-based reasoning (Tree of Thoughts, ToT) from online inference to offline training: during training, an enhanced SC-ToT search algorithm explores the reasoning space, extracts step-level preference pairs from successful reasoning paths, and fine-tunes the base model via Direct Preference Optimization (DPO) on a per-step basis. At inference time, the fine-tuned model performs a single standard Chain-of-Thought (CoT) decoding, achieving near-ToT accuracy with CoT-level latency.

The system encapsulates a five-stage automated pipeline: (1) data preparation and standardization, (2) enhanced SC-ToT search with bidirectional scoring, global beam selection, and terminal self-consistency voting, (3) dual-track training with CPO and TS-SFT baselines, (4) unified multi-baseline evaluation on GSM8K and StrategyQA with systematic ablation studies, and (5) an interactive web-based visualization platform built with FastAPI and D3.js for exploring search trees, comparing performance, and analyzing training dynamics.

The results validate the CPO paradigm: step-level preference signals outperform path-level supervised signals, the three SC-ToT enhancements are complementary and each contributes significantly, and the fine-tuned model achieves efficient, high-quality reasoning without online search overhead. This work provides a complete engineering reference implementation for the CPO framework and empirical evidence for key design choices through extensive ablation experiments.

**Keywords**: Large Language Models, Chain-of-Thought Reasoning, Tree of Thoughts, Preference Optimization, Direct Preference Optimization, LLM Reasoning Enhancement

---

> **第八章合计约1,700字（中文）+ 英文摘要约200词。**  
> **论文全文累计：2,600 + 2,800 + 2,300 + 2,900 + 2,300 + 2,700 + 2,300 + 1,700 = 19,600字。**
