"""
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from pickle import FALSE
import sys
import json
from textwrap import indent
import argparse
import logging
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
model_id = "openai/gpt-oss-120b"

# Global variables for logging
_log_file = None
original_print = print

# Global model and tokenizer
_model = None
_tokenizer = None

def log_print(*args, **kwargs):
    """
    Custom print function that writes to both stdout and log file.
    """
    # Convert all arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Add timestamp to lines starting with ">>>>>"
    if message.startswith('>>>>>'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] {message}"
    
    # Print to stdout
    original_print(message)
    
    # Also write to log file if specified
    if _log_file is not None:
        _log_file.write(message + '\n')
        _log_file.flush()  # Ensure immediate writing

# Replace the built-in print function
print = log_print

def set_log_file(log_file_path):
    """Set the log file for output."""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

def cleanup_memory():
    """Clean up GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_gpt_oss_model():
    """Load the GPT OSS model and tokenizer."""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        print("Loading GPT OSS model...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="cuda:0",
            )
            print("GPT OSS model loaded successfully.")
        except Exception as e:
            print(f"Error loading GPT OSS model: {e}")
            sys.exit(1)
    
    return _model, _tokenizer

def save_memory(memory_file, problem_statement, other_prompts, current_iteration, max_runs, solution=None, verify=None):
    """
    Save the current state to a memory file.
    """
    memory = {
        "problem_statement": problem_statement,
        "other_prompts": other_prompts,
        "current_iteration": current_iteration,
        "max_runs": max_runs,
        "solution": solution,
        "verify": verify,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"Memory saved to {memory_file}")
        return True
    except Exception as e:
        print(f"Error saving memory to {memory_file}: {e}")
        return False

def load_memory(memory_file):
    """
    Load the state from a memory file.
    """
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        print(f"Memory loaded from {memory_file}")
        return memory
    except Exception as e:
        print(f"Error loading memory from {memory_file}: {e}")
        return None

step1_prompt = """
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

check_verification_prompt = """
Can you carefully review each item in your list of findings? Are they valid or overly strict? An expert grader must be able to distinguish between a genuine flaw and a concise argument that is nonetheless sound, and to correct their own assessment when necessary.

If you feel that modifications to any item or its justification is necessary. Please produce a new list. In your final output, please directly start with **Summary** (no need to justify the new list).
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

*   **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
*   **Location:** "By interchanging the limit and the integral, we get..."
    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.

"""


verification_remider = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

def read_file_content(filepath):
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def build_messages(system_prompt, question_prompt, other_prompts=None):
    """
    Builds the message list for the GPT OSS model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_prompt}
    ]
    
    if other_prompts:
        for prompt in other_prompts:
            messages.append({"role": "user", "content": prompt})
    
    return messages

def send_gpt_oss_request(messages):
    """
    Sends a request to the local GPT OSS model and returns the response.
    """
    model, tokenizer = load_gpt_oss_model()
    
    try:
        # Apply chat template with high reasoning effort
        inputs = tokenizer.apply_chat_template(
            messages, 
            reasoning_effort="low",
            add_generation_prompt=True, 
            return_tensors="pt", 
            return_dict=True
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=128000, 
                temperature=0.1, 
                top_p=1.0,
                # do_sample=True,
                # pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        full_text = tokenizer.decode(output[0])#, skip_special_tokens=False)
        
        # Clean up memory after generation
        del output
        cleanup_memory()
        
        return full_text
    
    except Exception as e:
        print(f"Error during GPT OSS request: {e}")
        raise e

def extract_text_from_response(response_text):
    """
    Extracts the generated text from the GPT OSS response.
    """
    try:
        # Extract final answer from GPT OSS channel markers
        if "<|channel|>final<|message|>" in response_text:
            answer = response_text.split("<|channel|>final<|message|>")[-1].split("<|end|>")[0].strip()
            if answer.endswith("<|return|>"):
                answer = answer[:-len("<|return|>")].rstrip()
            return answer
        # Extract from analysis channel if no final channel
        elif "<|channel|>analysis<|message|>" in response_text:
            analysis = response_text.split("<|channel|>analysis<|message|>")[-1].split("<|end|>")[0].strip()
            if analysis.endswith("<|return|>"):
                analysis = analysis[:-len("<|return|>")].rstrip()
            return analysis
        else:
            # Fallback: extract content after the last user message
            # Look for assistant response after the last user input
            parts = response_text.split("<|im_start|>assistant")
            if len(parts) > 1:
                assistant_response = parts[-1].split("<|im_end|>")[0].strip()
                return assistant_response
            else:
                # If no clear assistant response, return the full text
                return response_text
    except Exception as e:
        print(f"Error extracting text from response: {e}")
        print(f"Full response: {response_text}")
        return response_text

def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if(after):
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):

    dsol = extract_detailed_solution(solution)

    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_remider}
"""
    if(verbose):
        print(">>>>>>> Start verification.")
    
    messages = build_messages(
        system_prompt=verification_system_prompt, 
        question_prompt=newst
    )
    
    if(verbose):
        print(">>>>>>> Verification messages:")
        print(json.dumps(messages, indent=4))

    response_text = send_gpt_oss_request(messages)
    out = extract_text_from_response(response_text)

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4))

    check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
            + "\n\n" + out 
    
    check_messages = build_messages(system_prompt="", question_prompt=check_correctness)
    check_response = send_gpt_oss_request(check_messages)
    o = extract_text_from_response(check_response)

    if(verbose):
        print(">>>>>>> Is verification good?")
        print(json.dumps(o, indent=4))
        
    bug_report = ""

    if("yes" not in o.lower()):
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

    if(verbose):
        print(">>>>>>>Bug report:")
        print(json.dumps(bug_report, indent=4))
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """

    messages = build_messages(system_prompt="", question_prompt=check_complete_prompt)
    response_text = send_gpt_oss_request(messages)
    o = extract_text_from_response(response_text)

    print(o)
    return "yes" in o.lower()


def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    messages = build_messages(
        system_prompt=step1_prompt,
        question_prompt=problem_statement,
        other_prompts=other_prompts
    )

    print(f">>>>>> Initial messages.")
    print(json.dumps(messages, indent=4))

    response_text = send_gpt_oss_request(messages)
    output1 = extract_text_from_response(response_text)

    print(f">>>>>>> First solution: ") 
    print(json.dumps(output1, indent=4))

    print(f">>>>>>> Self improvement start:")
    
    # Add the model's response and improvement prompt
    messages.append({"role": "assistant", "content": output1})
    messages.append({"role": "user", "content": self_improvement_prompt})

    response_text2 = send_gpt_oss_request(messages)
    solution = extract_text_from_response(response_text2)
    print(f">>>>>>> Corrected solution: ")
    print(json.dumps(solution, indent=4))
    
    print(f">>>>>>> Vefify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print(f">>>>>>> Initial verification: ")
    print(json.dumps(verify, indent=4))
    print(f">>>>>>> verify results: {good_verify}")
    
    return messages, solution, verify, good_verify

def agent(problem_statement, other_prompts=[], memory_file=None, resume_from_memory=False):
    if resume_from_memory and memory_file:
        # Load memory and resume from previous state
        memory = load_memory(memory_file)
        if memory:
            problem_statement = memory.get("problem_statement", problem_statement)
            other_prompts = memory.get("other_prompts", other_prompts)
            current_iteration = memory.get("current_iteration", 0)
            solution = memory.get("solution", None)
            verify = memory.get("verify", None)
            print(f"Resuming from iteration {current_iteration}")
        else:
            print("Failed to load memory, starting fresh")
            current_iteration = 0
            solution = None
            verify = None
    else:
        # Start fresh
        current_iteration = 0
        solution = None
        verify = None
    
    if solution is None:
        messages, solution, verify, good_verify = init_explorations(problem_statement, True, other_prompts)
        if(solution is None):
            print(">>>>>>> Failed in finding a complete solution.")
            return None
    else:
        # We have a solution from memory, need to get good_verify
        _, good_verify = verify_solution(problem_statement, solution)

    error_count = 0
    correct_count = 1
    success = False
    for i in range(current_iteration, 30):
        print(f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")

        if("yes" not in good_verify.lower()):
            # clear
            correct_count = 0
            error_count += 1

            #self improvement
            print(">>>>>>> Verification does not pass, correcting ...")
            # establish a new prompt that contains the solution and the verification

            messages = build_messages(
                system_prompt=step1_prompt,
                question_prompt=problem_statement,
                other_prompts=other_prompts
            )

            messages.append({"role": "assistant", "content": solution})
            messages.append({"role": "user", "content": correction_prompt + "\n\n" + verify})

            print(">>>>>>> New messages:")
            print(json.dumps(messages, indent=4))
            
            response_text = send_gpt_oss_request(messages)
            solution = extract_text_from_response(response_text)

            print(">>>>>>> Corrected solution:")
            print(json.dumps(solution, indent=4))

        print(f">>>>>>> Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)

        if("yes" in good_verify.lower()):
            print(">>>>>>> Solution is good, verifying again ...")
            correct_count += 1
            error_count = 0


        # Save memory every iteration
        if memory_file:
            save_memory(memory_file, problem_statement, other_prompts, i, 30, solution, verify)
        
        if(correct_count >= 5):
            print(">>>>>>> Correct solution found.")
            print(json.dumps(solution, indent=4))
            return solution

        elif(error_count >= 10):
            print(">>>>>>> Failed in finding a correct solution.")
            # Save final state before returning
            if memory_file:
                save_memory(memory_file, problem_statement, other_prompts, i, 30, solution, verify)
            return None

    if(not success):
        print(">>>>>>> Failed in finding a correct solution.")
        # Save final state before returning
        if memory_file:
            save_memory(memory_file, problem_statement, other_prompts, 30, 30, solution, verify)
        return None
        
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent using GPT OSS')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt', 
                       help='Path to the problem statement file (default: problem_statement.txt)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    parser.add_argument('--memory', '-mem', type=str, help='Path to memory file for saving/loading state (optional)')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from memory file if provided')
    
    args = parser.parse_args()

    max_runs = args.max_runs
    memory_file = args.memory
    resume_from_memory = args.resume
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)
    
    if memory_file:
        print(f"Memory file: {memory_file}")
        if resume_from_memory:
            print("Resume mode: Will attempt to load from memory file")

    # Set up logging if log file is specified
    if args.log:
        if not set_log_file(args.log):
            sys.exit(1)
        print(f"Logging to file: {args.log}")
    
    problem_statement = read_file_content(args.problem_file)

    for i in range(max_runs):
        print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts, memory_file, resume_from_memory)
            if(sol is not None):
                print(f">>>>>>> Found a correct solution in run {i}.")
                print(json.dumps(sol, indent=4))
                break
        except Exception as e:
            print(f">>>>>>> Error in run {i}: {e}")
            continue
    
    # Close log file if it was opened
    close_log_file()