# AI Agent Pipeline 详细解析

本文档将详细解释 `agent.py` 脚本的工作流程。该脚本设计了一个基于大型语言模型（LLM）的自动化系统，旨在通过多轮生成、验证和修正的循环，来解决国际数学奥林匹克（IMO）级别的高难度数学问题。

下方的流程图直观地展示了整个工作流。随后的章节将详细阐述每个步骤及其在代码中的具体实现。

![Agent Pipeline Flowchart](images/pipeline.png)

---

### 总览：核心思想

该系统的核心思想是模拟一位顶尖数学家的解题过程：

1.  **提出初步解法**：首先，快速生成一个完整的解题思路。
2.  **自我审视和改进**：对自己提出的解法进行反思，修正明显的漏洞。
3.  **同行评审**：将解法交给另一位（虚拟的）专家进行严格的、挑剔的验证。
4.  **根据反馈修正**：如果评审发现问题，则根据具体的反馈报告进行有针对性的修正。
5.  **重复验证**：将修正后的解法再次提交评审，直到它无懈可击。
6.  **建立信心**：为了防止偶然的成功，一个解法需要连续通过多次严格的评审，才能被最终接受。

---

### 步骤 1: 初始解题 (Initial Solution Generation)

**流程图描述:** 这是整个流程的起点。AI 接收问题陈述，并生成一个初步的解决方案。

**代码实现:** 此步骤在 `init_explorations` 函数的开始部分完成。

1.  函数接收 `problem_statement`（问题陈述）和可选的 `other_prompts`（附加提示）作为输入。
2.  通过 `build_request_payload` 函数构建对 AI 的第一个请求。请求的核心是 `step1_prompt`，它是一个精心设计的系统提示，包含了对 AI 的核心指令：要求它以极高的严谨性（Rigor is Paramount）来生成一个逻辑完整、步骤清晰的解答，并且必须使用指定的格式（Summary, Detailed Solution）。
3.  调用 `send_api_request` 将请求发送给 AI 模型，获取初步的解决方案 `output1`。

```python
# 位于 init_explorations 函数
def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    p1  = build_request_payload(
            system_prompt=step1_prompt,
            question_prompt=problem_statement,
            other_prompts=other_prompts
        )
    # ...
    response1 = send_api_request(get_api_key(), p1)
    output1 = extract_text_from_response(response1)
    # ...
```

---

### 步骤 2: 自我改进 (Self-Improvement)

**流程图描述:** 在生成初步解答后，AI 会立即对其进行一次自我审视和改进，这是第一层修正机制。

**代码实现:** 此步骤紧接着步骤 1，同样在 `init_explorations` 函数中执行。

1.  将步骤 1 中生成的初始解答 `output1` 作为模型的历史回答（`"role": "model"`）添加到对话中。
2.  将 `self_improvement_prompt`（一个简洁的自我改进提示）作为新的用户指令（`"role": "user"`）追加到请求中。该提示引导 AI 重新审视自己的答案，修正潜在的错误并填补逻辑上的空白。
3.  再次调用 `send_api_request`，获得一个经过改进的、更完善的解答版本 `solution`。这个 `solution` 将是后续验证和修正的基础。

```python
# 位于 init_explorations 函数
def init_explorations(...):
    # ... (步骤 1 代码)

    p1["contents"].append(
        {"role": "model", "parts": [{"text": output1}]}
    )
    p1["contents"].append(
        {"role": "user", "parts": [{"text": self_improvement_prompt}]}
    )

    response2 = send_api_request(get_api_key(), p1)
    solution = extract_text_from_response(response2)
    # ...
```

---

### 步骤 3: 完整性检查 (Completeness Check)

**流程图描述:** 这是一个关键的“守门”环节。在投入资源进行严格验证之前，系统会先检查 AI 生成的解答是否**声称**是完整的。如果 AI 自己承认这只是一个部分解，那么流程将提前终止，避免浪费时间去验证一个注定不合格的答案。

**代码实现:** 由 `check_if_solution_claimed_complete` 函数实现。

1.  该函数接收 `solution` 文本，并构建一个简单的提示，询问 AI 该文本是否在声称一个完整的解。
2.  AI 会以简单的 "yes" 或 "no" 回答。
3.  此检查在 `init_explorations` 函数末尾被首次调用，并在 `agent` 主循环的每次修正后再次调用。如果检查结果为 "no"，则判定为失败，当前轮次的尝试结束。

```python
# check_if_solution_claimed_complete 函数的核心逻辑
def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
...
{solution}
...
Response in exactly "yes" or "no". No other words.
    """
    # ... 发送 API 请求并解析 "yes" 或 "no" ...
    return "yes" in o.lower()

# 在 agent 函数中调用
# ...
is_complete = check_if_solution_claimed_complete(solution)
if not is_complete:
    print(f">>>>>>> Solution is not complete. Failed.")
    return None
```

---

### 步骤 4: 验证 (Verification)

**流程图描述:** 这是流程的核心决策点。通过完整性检查的解答会在这里被一个独立的、扮演“批评家”角色的 AI 进行严格验证。

**代码实现:** 该功能由 `verify_solution` 函数实现。这是一个精巧的**两阶段验证**过程：

1.  **阶段 A: 生成验证报告**
    *   `verify_solution` 函数会构建一个特殊的 API 请求。其核心是 `verification_system_prompt`，这个提示会要求另一个 AI 实例扮演一个苛刻的 IMO 竞赛评分员。它的唯一任务是找出解答中的所有逻辑错误和不严谨之处，并生成详细的**验证报告 (bug report)**，而不是自己去解题。
2.  **阶段 B: 解析验证结论**
    *   评分员生成的报告是自然语言文本。为了让程序能理解其最终结论，代码会再次调用 AI，用一个非常简单的问题 (`check_correctness`) 来判断该报告的总体倾向是“正面”的（如 "The solution is correct"）还是“负面”的（如 "The solution contains a Critical Error"）。
    *   这个最终的判断结果，一个简单的 "yes" 或 "no"，被存储在 `good_verify` 变量中，它将直接决定 `agent` 函数主循环的走向。

```python
# 位于 verify_solution 函数
def verify_solution(problem_statement, solution, verbose=True):
    # ... 构建带有 verification_system_prompt 的请求 ...
    res = send_api_request(get_api_key(), p2)
    out = extract_text_from_response(res)  # 'out' 是完整的验证报告

    # ... 构建 check_correctness 请求来判断 'out' 的结论 ...
    r = send_api_request(get_api_key(), prompt)
    o = extract_text_from_response(r)  # 'o' 就是 agent() 中的 'good_verify'

    bug_report = ""
    if("yes" not in o.lower()):
        # 如果结论是“不正确”，则提取详细的错误报告
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

    return bug_report, o
```

---

### 步骤 5 & 6: 审阅错误报告并修正 (Bug Report Review & Correction)

**流程图描述:** 如果验证失败（`failed`），流程将进入核心的修正循环。解题 AI 会接收到来自批评家 AI 的错误报告，并被要求根据该报告生成一个修正版的解答。

**代码实现:** 这个循环由 `agent` 函数的主循环中 `if("yes" not in good_verify.lower())` 条件块负责。

1.  **进入条件:** 当 `verify_solution` 返回的 `good_verify` 不包含 "yes" 时，代表验证失败。
2.  **计数器更新:** 连续正确次数 `correct_count` 被清零，而连续错误次数 `error_count` 则加一，用于追踪失败的“僵局”。
3.  **构建修正请求:** 这是实现修正的关键。代码会构建一个信息丰富的 API 请求，其中包含了：
    *   原始的 `problem_statement`。
    *   先前被证明有误的 `solution`。
    *   `correction_prompt`：一个明确的指令，告诉 AI “根据下面的错误报告来修正你的答案”。
    *   `verify`：从验证环节得到的完整错误报告。
4.  **循环:** 新生成的解答 `solution` 将在下一次循环中被重新送回**步骤 3 (完整性检查)**，然后是**步骤 4 (验证)**，形成一个完整的“验证-修正”闭环。

```python
# 位于 agent 函数的主循环中
for i in range(30):
    # ...
    if("yes" not in good_verify.lower()):
        # 验证失败，进入修正流程
        correct_count = 0
        error_count += 1

        # ... 构建修正请求 (同时完成步骤 5 & 6) ...
        p1["contents"].append(
            {"role": "model", "parts": [{"text": solution}]}
        )
        p1["contents"].append(
            {"role": "user", "parts": [{"text": correction_prompt}, {"text": verify}]}
        )

        response2 = send_api_request(get_api_key(), p1)
        solution = extract_text_from_response(response2) # 获得修正后的解答
        
        # ... 别忘了对新解进行完整性检查 ...
        is_complete = check_if_solution_claimed_complete(solution)
        if not is_complete: # ...
    # ...
```

---

### 步骤 7: 接受 (Accept)

**流程图描述:** 当一个解答连续通过 5 次验证后，它将被视为最终的正确答案而被接受。设置这个阈值是为了确保解答的正确性不是偶然的，而是稳定和可靠的。

**代码实现:** 通过 `agent` 函数中的 `correct_count` 变量实现。

1.  每当验证结果 `good_verify` 为 "yes"，`correct_count` 就会加一，同时 `error_count` 被重置为 0。
2.  循环会检查 `correct_count` 是否达到了 5。如果达到，则认为已经找到了一个高度可靠的解答，程序成功退出并返回该解答。

```python
# 位于 agent 函数的主循环中
if("yes" in good_verify.lower()):
    print(">>>>>>> Solution is good, verifying again ...")
    correct_count += 1
    error_count = 0 # 重置错误计数

if(correct_count >= 5):
    print(">>>>>>> Correct solution found.")
    return solution
```

---

### 步骤 8: 拒绝 (Reject)

**流程图描述:** 如果解答连续 10 次都未能通过验证，系统将判定无法找到有效解，流程终止。这是一种熔断机制，用于防止 AI 陷入一个无法修复的逻辑错误中，从而避免无限循环和资源浪费。

**代码实现:** 通过 `agent` 函数中的 `error_count` 变量实现。

1.  每当验证失败，`error_count` 就会加一。
2.  循环会检查 `error_count` 是否达到了 10。如果达到，则认为 AI 已陷入困境，程序将终止执行并返回 `None`，表示本次运行失败。

```python
# 位于 agent 函数的主循环中
if("yes" not in good_verify.lower()):
    # ...
    error_count += 1

# ...

elif(error_count >= 10):
    print(">>>>>>> Failed in finding a correct solution.")
    return None
``` 