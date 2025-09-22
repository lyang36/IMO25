# IMO 2025 Problem Solver
An AI agent system for solving International Mathematical Olympiad (IMO) problems using Google's Gemini, OpenAI, and XAI APIs.

```
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
```

## Overview

This project consists of the following components:
- `code/agent.py`: A single AI agent that attempts to solve IMO problems with default base model: Google Gemini 2.5 Pro
- `code/agent_oai.py`: A single AI agent that uses OpenAI GPT-5 model (same CLI/usage as `agent.py`)
- `code/agent_xai.py`: A single AI agent that uses XAI Grok-4-0709 models (same CLI/usage as `agent.py`)
- `code/run_parallel.py`: A parallel execution system that runs multiple agents simultaneously
- `code/res2md.py`: A small utility to parse a result file that contains JSON (e.g., JSONL) and print the last JSON object

These agents have successfully solved IMO 2025 problems 1–5 in internal runs (logs attached), indicative of gold-medal performance.

## Run logs

- `run_logs/`: initial runs using Google Gemini 2.5 Pro as the base model
- `run_logs_gpt5/`: runs using OpenAI GPT-5 as the base model
- `run_logs_grok4/`: runs using XAI Grok-4 as the base model

These folders contain example successful run logs demonstrating end-to-end solutions produced by the respective base models.

## Prerequisites

1. **Python 3.7+** installed on your system
2. **API key(s) for your chosen provider(s)**:
   - Google Gemini: https://aistudio.google.com/app/apikey
   - OpenAI: https://platform.openai.com/api-keys
   - XAI: Refer to your XAI account portal for API key issuance
3. **Required Python packages**:
   ```bash
   pip install requests
   ```

## Setup

1. **Clone or download the project files**
2. **Set up your API key(s)**:
   - Set environment variables in your shell for the providers you plan to use, for example:
     - `export GOOGLE_API_KEY=your_google_api_key`
     - `export OPENAI_API_KEY=your_openai_api_key`
     - `export XAI_API_KEY=your_xai_api_key`

## Usage

### Single Agent (`agent.py`, `agent_oai.py`, `agent_xai.py`)

Run a single agent to solve an IMO problem (usage and flags are the same for all three agents):

```bash
python agent.py problem.txt [options]
```

**Arguments:**
- `problem.txt`: Path to the problem statement file (required); imo2025 problems are in `problems`

**Options:**
- `--log LOG_FILE`: Specify a log file for output (default: prints to console)
- `--other_prompts PROMPTS`: Additional prompts separated by commas

**Example:**
```bash
python agent.py imo2025_p1.txt --log agent_output.log
```

To run with OpenAI or XAI instead, simply invoke the corresponding script with the same options:

```bash
python agent_oai.py imo2025_p1.txt --log agent_output_oai.log
python agent_xai.py imo2025_p1.txt --log agent_output_xai.log
```

### Parallel Execution (`code/run_parallel.py`)

Run multiple agents in parallel to increase the chance of finding a solution:

```bash
python IMO25/code/run_parallel.py <problem_file> [options]
```

**Arguments:**
- `problem.txt`: Path to the problem statement file (required). Use an absolute path or ensure the path is valid from within `IMO25/code/` (the script runs the agent with its working directory set to `IMO25/code/`).

**Options:**
- `--num-agents N` or `-n N`: Number of parallel agents (default: 10)
- `--log-dir DIR` or `-d DIR`: Directory for log files (default: logs)
- `--timeout SECONDS` or `-t SECONDS`: Timeout per agent in seconds (default: no timeout)
- `--max-workers N` or `-w N`: Maximum worker processes (default: number of agents)
- `--other_prompts PROMPTS` or `-o PROMPTS`: Additional prompts separated by commas
- `--agent-file PATH` or `-a PATH`: Path to the agent file to run (default: `agent.py` inside `IMO25/code/`)
- `--exit-immediately` or `-e`: Exit the whole run as soon as any agent finds a correct solution (otherwise, all agents run to completion)

**Examples:**
```bash
# Run 20 agents with 5-minute timeout each
python IMO25/code/run_parallel.py problems/imo2025_p1.txt -n 20 -t 300

# Run 5 agents with custom log directory and exit immediately on first success
python IMO25/code/run_parallel.py problems/imo2025_p1.txt -n 5 -d logs/p1_run -e

# Run with additional prompts and a custom agent file
python IMO25/code/run_parallel.py problems/imo2025_p1.txt -n 15 -o "focus_on_geometry,use_induction" -a agent.py

# Run OpenAI/XAI variants by pointing to the agent file
python IMO25/code/run_parallel.py problems/imo2025_p1.txt -n 10 -a agent_oai.py
python IMO25/code/run_parallel.py problems/imo2025_p1.txt -n 10 -a agent_xai.py
```

### Result extractor (`code/res2md.py`)

Parse a result file that contains JSON (for example, a `.jsonl` file where each line is a JSON object), and print the last JSON object in the file. Useful for quickly extracting the final structured result produced by some runs.

```bash
python IMO25/code/res2md.py <result_file>
```

**Example:**
```bash
python IMO25/code/res2md.py logs/results.jsonl
```

## Problem File Format
See the `problems` folder.

## Output and Logging

### Single Agent
- Output is printed to console by default
- Use `--log` to save output to a file
- The agent will indicate if a complete solution was found

### Parallel Execution
- Each agent creates a separate log file in the specified directory
- Progress is shown in real-time
- Final summary shows:
  - Total execution time
  - Number of successful/failed agents
  - Success rate
  - Which agent found a solution (if any)
  - Location of log files
- Each log entry includes a timestamp

## Understanding the Output

### Solution Detection
The system looks for the phrase "Found a correct solution in run" to identify successful solutions.

### Agent Behavior
- Agents can use Google's Gemini 2.5 Pro, OpenAI, or XAI models depending on the chosen script
- Each agent follows a structured approach with multiple attempts
- Solutions are verified for completeness and correctness
- Agents can provide partial solutions if complete solutions aren't found

## Tips for Best Results

1. **Problem Formatting**: Ensure your problem file is clear and well-formatted
2. **Parallel Execution**: Use more agents for harder problems (10-20 agents recommended)
3. **Timeout Settings**: Set reasonable timeouts (you may set no timeout)
4. **API Limits**: Be aware of Google/OpenAI/XAI API rate limits and costs
5. **Log Analysis**: Check individual agent logs for detailed reasoning

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure the relevant API key(s) are properly set (Google/OpenAI/XAI)
2. **Timeout Issues**: Increase timeout or reduce number of agents
3. **Memory Issues**: Reduce max-workers if running out of memory
4. **No Solutions Found**: Try running more agents or check problem clarity

### Debug Mode
Add verbose logging by modifying the agent code or check individual log files for detailed output.

## Changelog 

### 08/18/2025

- Added `code/agent_oai.py` and `code/agent_xai.py` (usage identical to `agent.py`).
- Logs now record timestamps.
- Parallel runs no longer exit by default when a solution is found; use `--exit-immediately` to stop at the first complete solution.
- Adding running logs from grok-4 and gpt-5

### 07/24/2025 

- Initial code release

## License

MIT License - Copyright (c) 2025 Lin Yang, Yichen Huang

This software is provided as-is. Users are free to copy, modify, and distribute the code with proper attribution.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

### Community Codes

Community contributions are located in `code/community_codes/`. These have not been thoroughly tested, so please use them at your own risk. 

## Disclaimer

This tool is for educational and research purposes. 

## Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2025gemini,
  title={Gemini 2.5 Pro Capable of Winning Gold at IMO 2025},
  author={Huang, Yichen and Yang, Lin F},
  journal={arXiv preprint arXiv:2507.15855},
  year={2025}
}
``` 
