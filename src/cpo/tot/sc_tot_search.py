"""
ToT (Tree of Thoughts) 搜索核心实现

严格遵循论文标准：
1. Thought Generation: 每步生成 k=10 个候选节点
2. State Evaluation: likely=10, impossible=1, uncertain=5
3. Beam Search: 只保留前 b=5 个高分节点
4. Self-Consistency: temperature=0.7, 5次采样投票
5. Terminal Node: 深度达标/等式结果/纯数字/结论词+数字
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING ,Optional

from cpo.llm.generation import GenerationRuntime, complete_text

if TYPE_CHECKING:
    from cpo.tot.tree import ToTTree


@dataclass
class ToTSearchConfig:
    """ToT 搜索配置（论文标准超参数）"""

    # 搜索控制
    max_depth: int = 5  # 最大推理深度
    candidates_per_step: int = 10  # 每步生成候选数 (k=10)
    beam_width: int = 5  # Beam Search 保留数 (b=5)

    # 模型控制
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    strict_llm: bool = True  # 是否严格使用 LLM API

    # 生成控制
    max_new_tokens_step: int = 128  # 步骤生成最大 token 数
    max_new_tokens_score: int = 8  # 评分生成最大 token 数
    max_new_tokens_sc: int = 16  # SC 投票最大 token 数

    # SC 投票控制
    sc_votes: int = 5  # Self-Consistency 投票次数
    sc_temperature: float = 0.7  # SC 采样温度

    # 打分控制
    score_likely: float = 10.0  # likely 分数
    score_impossible: float = 1.0  # impossible 分数
    score_uncertain: float = 5.0  # 模糊回答分数


@dataclass
class ToTNodeResult:
    """单个节点结果"""

    node_id: str
    depth: int
    state: str  # 合并后的完整推理路径
    thought: str  # 当前步骤
    score: float  # 步骤评分 (likely/impossible)
    is_terminal: bool  # 是否为终止节点
    sc_score: float = 0.0  # SC 投票分数
    sc_answer: str = ""  # SC 投票答案
    sc_vote_map: dict = field(default_factory=dict)  # 投票分布
    sc_valid_votes: int = 0  # 有效票数
    parent_id: Optional[str] = None


@dataclass
class ToTSearchResult:
    """ToT 搜索完整结果"""

    nodes: list[ToTNodeResult]  # 所有节点
    terminal_nodes: list[ToTNodeResult]  # 终止节点列表
    frontier_nodes: list[ToTNodeResult]  # Beam 保留节点
    best_node: ToTNodeResult  # 最终选择的最佳节点
    total_nodes: int  # 总节点数
    max_depth_reached: int  # 实际达到的最大深度


# ============================================================================
# 提示词构建
# ============================================================================


def _clean_reasoning_history(state: str) -> str:
    """清理推理历史，过滤无效文本"""
    if not state.strip():
        return ""
    lines = [x.strip() for x in state.splitlines() if x.strip()]
    # 过滤指令式文本
    kept = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ["do not", "please", "output", "cannot", "unable"]):
            continue
        kept.append(line)
    return "\n".join(kept)


def _build_step_prompt(question: str, state: str) -> str:
    """
    推理步骤生成 Prompt（论文标准格式）

    格式：
    Question: [QUESTION]
    Reasoning: [HIST]
    Next step:
    """
    cleaned_state = _clean_reasoning_history(state)
    return (
        f"Question: {question}\n"
        f"Reasoning: {cleaned_state if cleaned_state else '(start)'}\n"
        "Next step:"
    )


def _build_score_prompt(question: str, state: str, candidate_step: str) -> str:
    """
    评分评估 Prompt（论文最关键部分）

    规则：
    1. 不需要完美正确
    2. 合理且有用 → likely
    3. 无用 → impossible
    4. 只回复一个单词
    """
    return (
        "Please assess whether this candidate step helps move the solution forward.\n"
        f"Question: {question}\n"
        f"Current reasoning: {state if state.strip() else '(start)'}\n"
        f"Candidate step: {candidate_step}\n"
        "It does not need to be perfectly correct. "
        "If it is reasonable and helpful, reply: likely. "
        "If it does not help, reply: impossible.\n"
        "Reply with one word only."
    )


def _build_sc_answer_prompt(task: str, question: str, path_state: str) -> str:
    """
    SC 投票 Prompt - 从完整推理路径提取最终答案

    规则：
    - 只输出答案，不输出解释
    - 不重新计算、不重新推理
    """
    if task == "gsm8k":
        return (
            "Extract the final numeric answer from the given completed reasoning path.\n"
            f"Question: {question}\n"
            f"Completed reasoning path:\n{path_state}\n"
            "Output only the final number. Do not explain."
        )
    else:  # strategyqa
        return (
            "Extract the final answer (yes or no) from the given completed reasoning path.\n"
            f"Question: {question}\n"
            f"Completed reasoning path:\n{path_state}\n"
            "Output only yes or no. Do not explain."
        )


# ============================================================================
# 文本处理
# ============================================================================


def _normalize_step(text: str) -> str:
    """标准化推理步骤：取第一行并压缩空白"""
    if not text.strip():
        return ""
    first_line = text.strip().splitlines()[0].strip()
    return " ".join(first_line.split())


def _is_instructional_text(text: str) -> bool:
    """判断是否为指令式文本（应过滤）"""
    s = text.strip().lower()
    if not s:
        return True
    blocked = ["do not", "please", "output", "cannot", "unable to", "should"]
    return any(k in s for k in blocked)


def _is_terminal_state(state: str, depth: int, max_depth: int) -> bool:
    """
    判定是否为终止节点

    满足任一条件即终止：
    1. 推理深度达到最大深度
    2. 最后一行是等式结果（如 = 64）
    3. 最后一行是纯数字
    4. 最后一行包含结论词 + 数字
    """
    if depth >= max_depth:
        return True

    lines = [x.strip() for x in state.splitlines() if x.strip()]
    if not lines:
        return False
    last = lines[-1]

    # 条件1: 明确的最终答案等式（必须包含final/total/answer等关键词）
    if re.search(r"(final|total|answer|result).*=\s*[-+]?\d+(?:\.\d+)?\s*$", last):
        return True

    # 条件2: 纯数字仅在最大深度前1层判定为终止
    if depth >= max_depth - 1 and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", last.strip()):
        return True

    # 条件3: 结论词 + 数字
    conclusion_words = r"\b(result|total|answer|final answer|answer is)\b"
    if re.search(conclusion_words, last.lower()) and re.search(r"[-+]?\d+(?:\.\d+)?", last):
        return True

    return False


def _extract_sc_answer(task: str, text: str) -> str:
    """从 SC 投票响应中提取答案"""
    raw = str(text).strip()
    if not raw:
        return ""

    if task == "gsm8k":
        lowered = raw.lower()
        # 优先抽取显式最终答案短语后的数字
        m = re.findall(r"so the final answer is\s*[:\-]?\s*([-+]?\d+(?:\.\d+)?)", lowered)
        if m:
            return m[-1]
        # 回退：在整段文本中抽最后一个数字
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", lowered)
        if nums:
            return nums[-1]
        return ""
    else:
        # StrategyQA: 只接受 yes/no
        s = raw.lower()
        if s in {"yes", "no"}:
            return s
        if "yes" in s:
            return "yes"
        if "no" in s:
            return "no"
        return ""


def _runtime_with_temperature(runtime: GenerationRuntime, temperature: float) -> GenerationRuntime:
    """Create a runtime copy with a different temperature.

    complete_text does not accept a direct temperature kwarg in this project.
    """
    return GenerationRuntime(
        model_name=runtime.model_name,
        device=runtime.device,
        max_new_tokens=runtime.max_new_tokens,
        temperature=temperature,
        top_p=runtime.top_p,
        allow_fallback=runtime.allow_fallback,
        provider=runtime.provider,
        api_key=runtime.api_key,
        api_base_url=runtime.api_base_url,
        api_timeout=runtime.api_timeout,
        api_enable_thinking=runtime.api_enable_thinking,
        api_max_retries=runtime.api_max_retries,
        api_retry_backoff_sec=runtime.api_retry_backoff_sec,
    )


def _answer_support_score(state: str, answer: str) -> int:
    """Score how strongly the path text supports the extracted final answer.

    Only used as a tie-breaker after SC score and step score are equal.
    """
    if not answer:
        return 0
    s = state.lower()
    a = re.escape(str(answer).lower())
    score = 0
    if re.search(rf"so the final answer is\s*[:\-]?\s*{a}\b", s):
        score += 3
    if re.search(rf"(therefore|thus|hence|answer is)[^\n]*\b{a}\b", s):
        score += 2
    if re.search(rf"=\s*{a}\b", s):
        score += 1
    if re.search(rf"\b{a}\b", s):
        score += 1
    return score


# ============================================================================
# 核心搜索函数
# ============================================================================


def _generate_candidates(
    *,
    question: str,
    state: str,
    runtime: GenerationRuntime,
    config: ToTSearchConfig,
) -> list[str]:
    """
    生成候选推理步骤

    论文规范：每步生成 k 个候选，过滤无效文本和重复
    """
    candidates: list[str] = []
    seen: set[str] = set()
    prompt = _build_step_prompt(question=question, state=state)

    gen_runtime = _runtime_with_temperature(runtime, 0.8)
    for _ in range(config.candidates_per_step):
        # 使用 do_sample=True 生成多样化的候选
        out = complete_text(
            prompt=prompt,
            runtime=gen_runtime,
            max_new_tokens=config.max_new_tokens_step,
            do_sample=True,
        )
        step = _normalize_step(out)
        if not step:
            continue
        if _is_instructional_text(step):
            continue
        if step in seen:
            continue
        seen.add(step)
        candidates.append(step)

    return candidates


def _score_candidate(
    *,
    question: str,
    state: str,
    candidate_step: str,
    runtime: GenerationRuntime,
    config: ToTSearchConfig,
) -> tuple[float, str]:
    """
    评估候选步骤

    论文规范：
    - likely = 10 分
    - impossible = 1 分
    - 模糊回答 = 5 分
    """
    prompt = _build_score_prompt(question=question, state=state, candidate_step=candidate_step)
    out = complete_text(
        prompt=prompt,
        runtime=runtime,
        max_new_tokens=config.max_new_tokens_score,
        do_sample=False,  # 评分用确定性生成
    )

    # 解析评分
    text = out.strip().lower()
    if "likely" in text:
        return config.score_likely, "likely"
    if "impossible" in text:
        return config.score_impossible, "impossible"
    return config.score_uncertain, "uncertain"


def _score_path_with_sc(
    *,
    task: str,
    question: str,
    path_state: str,
    runtime: GenerationRuntime,
    config: ToTSearchConfig,
) -> tuple[float, str, int, dict[str, int]]:
    """
    Self-Consistency 投票

    论文规范：
    1. 固定推理路径，不修改、不重写
    2. 使用带采样 (temperature=0.7) 生成 votes 次
    3. 统计频次，取众数
    4. 众数得票率 = SC 分数
    """
    counter: dict[str, int] = {}
    prompt = _build_sc_answer_prompt(task=task, question=question, path_state=path_state)

    sc_runtime = _runtime_with_temperature(runtime, config.sc_temperature)
    for _ in range(config.sc_votes):
        out = complete_text(
            prompt=prompt,
            runtime=sc_runtime,
            max_new_tokens=config.max_new_tokens_sc,
            do_sample=True,
        )
        ans = _extract_sc_answer(task, out)
        if not ans:
            continue
        counter[ans] = counter.get(ans, 0) + 1

    valid_votes = sum(counter.values())
    if valid_votes <= 0:
        return 0.0, "", 0, {}

    # 取众数
    majority_answer = max(counter.items(), key=lambda kv: kv[1])[0]
    majority_count = counter[majority_answer]
    score = float(majority_count / valid_votes)

    return score, majority_answer, valid_votes, counter


# ============================================================================
# 主搜索流程
# ============================================================================


def run_tot_search(
    *,
    question: str,
    task: str,
    config: ToTSearchConfig,
) -> ToTSearchResult:
    """
    ToT 主搜索流程

    执行步骤：
    1. 从根节点开始
    2. 每步生成 k 个候选节点
    3. 每个节点打分
    4. 剪枝保留前 b 个高分节点
    5. 继续扩展直到触发终止
    """
    # 初始化
    runtime = GenerationRuntime(
        model_name=config.model_name,
        allow_fallback=not config.strict_llm,
    )

    # 节点存储
    all_nodes: list[ToTNodeResult] = []
    frontier: list[ToTNodeResult] = []
    node_counter = 0

    # 初始化根节点
    root = ToTNodeResult(
        node_id="root",
        depth=0,
        state="",
        thought="",
        score=0.0,
        is_terminal=False,
    )
    all_nodes.append(root)
    frontier.append(root)

    # Beam Search 循环
    max_depth_reached = 0

    for depth in range(1, config.max_depth + 1):
        all_candidates = []  # 全局候选列表
        for parent in frontier:
            # 终止节点不再扩展（同步修复第二个缺陷）
            if parent.is_terminal:
                continue
            # 每条独立路径单独生成、单独剪枝
            candidates = _generate_candidates(
                question=question,
                state=parent.state,
                runtime=runtime,
                config=config,
            )
            parent_level_nodes = []
            for step in candidates:
                node_counter += 1
                merged_state = step if not parent.state else f"{parent.state}\n{step}"
                score, label = _score_candidate(
                    question=question,
                    state=parent.state,
                    candidate_step=step,
                    runtime=runtime,
                    config=config,
                )
                is_terminal = _is_terminal_state(merged_state, depth, config.max_depth)
                node = ToTNodeResult(
                    node_id=f"n{node_counter}",
                    depth=depth,
                    state=merged_state,
                    thought=step,
                    score=score,
                    is_terminal=is_terminal,
                    parent_id=parent.node_id, # 同步修复第四个缺陷
                )
                parent_level_nodes.append(node)
            all_candidates.extend(parent_level_nodes)  # 合并到全局列表
        # 所有父节点处理完后，再全局排序+更新frontier
        if not all_candidates:
            break  # 无候选可扩展，终止    
        all_candidates.sort(key=lambda n: (n.score, int(n.is_terminal)), reverse=True)    
        frontier = all_candidates[:config.beam_width]  # 全局择优
        all_nodes.extend(all_candidates)
        max_depth_reached = depth

        # 如果有终止节点，停止扩展
        if all(n.is_terminal for n in frontier):
            break

    # 获取终止节点
    terminal_nodes = [n for n in all_nodes if n.node_id != "root" and n.is_terminal]
    if not terminal_nodes:
        # 如果没有终止节点，使用 frontier
        terminal_nodes = frontier.copy()

    # SC 投票：对每个终止节点执行投票
    candidate_for_sc = terminal_nodes if terminal_nodes else frontier

    best_node = frontier[0] if frontier else root
    best_node.sc_score = -1.0  # 初始化为最低

    for node in candidate_for_sc:
        sc_score, sc_answer, sc_valid_votes, sc_vote_map = _score_path_with_sc(
            task=task,
            question=question,
            path_state=node.state,
            runtime=runtime,
            config=config,
        )
        node.sc_score = sc_score
        node.sc_answer = sc_answer
        node.sc_valid_votes = sc_valid_votes
        node.sc_vote_map = sc_vote_map

        # 更新最佳节点：SC 分数 > 节点分数 > 是否终止；其后用答案支撑度打破平局
        if (
            sc_score,
            node.score,
            int(node.is_terminal),
            _answer_support_score(node.state, node.sc_answer),
        ) > (
            best_node.sc_score,
            best_node.score,
            int(best_node.is_terminal),
            _answer_support_score(best_node.state, best_node.sc_answer),
        ):
            best_node = node

    return ToTSearchResult(
        nodes=all_nodes,
        terminal_nodes=terminal_nodes,
        frontier_nodes=frontier,
        best_node=best_node,
        total_nodes=len(all_nodes),
        max_depth_reached=max_depth_reached,
    )


# ============================================================================
# 辅助函数
# ============================================================================


def backtrack_path(nodes: list[ToTNodeResult], target_id: str) -> list[ToTNodeResult]:
    """
    回溯从根节点到目标节点的路径(借助parent_id)

    """
    node_map = {n.node_id: n for n in nodes}
    if target_id not in node_map:
        return []
    # 从目标节点回溯到根
    path_reversed = []
    current = node_map[target_id]
    while current is not None:
        path_reversed.append(current)
        current = node_map.get(current.parent_id) if current.parent_id else None
    # 反转得到根到目标的完整路径
    path_reversed.reverse()
    return path_reversed


def extract_final_answer(task: str, node: ToTNodeResult) -> str:
    """
    从最佳节点提取最终答案

    优先级：
    1. SC 投票答案（如果有效）
    2. 从 state 中提取数字
    """
    if node.sc_answer:
        return node.sc_answer

    # 从 state 中提取
    lines = [x.strip() for x in node.state.splitlines() if x.strip()]
    if not lines:
        return ""

    if task == "gsm8k":
        # 优先从等式提取
        for line in reversed(lines):
            m = re.search(r"=\s*([-+]?\d+(?:\.\d+)?)\s*$", line)
            if m:
                return m.group(1)
        # 回退：提取最后一个数字
        for line in reversed(lines):
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
            if nums:
                return nums[-1]
    else:
        # StrategyQA
        for line in reversed(lines):
            s = line.lower()
            if "yes" in s or s.strip() == "true":
                return "yes"
            if "no" in s or s.strip() == "false":
                return "no"

    return ""
