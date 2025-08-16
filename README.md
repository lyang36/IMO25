# IMO 2025 Problem Solver
A parallel AI agent system for solving International Mathematical Olympiad (IMO) problems using Google's Gemini API.

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
- `code/agent.py`: A single AI agent that attempts to solve IMO problems
- `code/run_parallel.py`: A parallel execution system that runs multiple agents simultaneously
- `code/res2md.py`: A small utility to parse a result file that contains JSON (e.g., JSONL) and print the last JSON object

## Prerequisites

1. **Python 3.10+** installed on your system (tested with Python 3.11)
2. **Google Gemini API key** - Get one from [Google AI Studio](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro)
3. **Required Python packages**:
   ```bash
   pip install requests python-dotenv
   ```

## Setup

1. **Clone or download the project files**
2. **Set up your API key**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   # Edit .env and add your Gemini API key
   # GEMINI_API_KEY=your_api_key_here
   ```
   - The system also supports `GOOGLE_API_KEY` for backwards compatibility
3. **Optional: Use direnv for automatic environment loading**:
   - The project includes a `.envrc` file for direnv users
   - Run `direnv allow` to automatically load environment variables

## Usage

### Quick Start with Make

```bash
# Set up environment and create .env file
make env

# Run agent on the first IMO problem
make run-imo01

# Run agent on a specific problem
make run-agent PROBLEM=problems/imo02.txt
```

### Single Agent (`code/agent.py`)

Run a single agent to solve an IMO problem:

```bash
python code/agent.py problems/imo01.txt [options]
```

**Arguments:**
- `problem_file`: Path to the problem statement file (required)

**Options:**
- `--log LOG_FILE` or `-l`: Specify a log file for output (default: prints to console)
- `--other_prompts PROMPTS` or `-o`: Additional prompts separated by commas
- `--max_runs N` or `-m`: Maximum number of runs (default: 10)

**Examples:**
```bash
# Run agent on problem 1 with single attempt
python code/agent.py problems/imo01.txt --max_runs 1

# Run with logging
python code/agent.py problems/imo01.txt --log output.log

# Run with additional prompts
python code/agent.py problems/imo01.txt -o "use analytic geometry,consider edge cases"
```

### Parallel Execution (`code/run_parallel.py`)

Run multiple agents in parallel to increase the chance of finding a solution:

```bash
python code/run_parallel.py problems/imo01.txt [options]
```

**Arguments:**
- `problem_file`: Path to the problem statement file (required)

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
python code/run_parallel.py problems/imo01.txt -n 20 -t 300

# Run 5 agents with custom log directory and exit immediately on first success
python code/run_parallel.py problems/imo01.txt -n 5 -d logs/p1_run -e

# Run with additional prompts
python code/run_parallel.py problems/imo01.txt -n 15 -o "focus_on_geometry,use_induction"
```

### Result Extractor (`code/res2md.py`)

Parse a result file that contains JSON (for example, a `.jsonl` file where each line is a JSON object), and print the last JSON object in the file. Useful for quickly extracting the final structured result produced by some runs.

```bash
python code/res2md.py <result_file>
```

**Example:**
```bash
python code/res2md.py logs/results.jsonl
```

## Problem File Format

Problem files should be plain text files containing the problem statement. See the `problems/` folder for IMO 2025 problems (imo01.txt through imo06.txt).

## Project Structure

```
IMO25-solution/
├── code/
│   ├── agent.py           # Main agent implementation
│   ├── run_parallel.py    # Parallel execution script
│   └── res2md.py          # Result extraction utility
├── prompts/               # Extracted prompt templates
│   ├── step1_prompt.txt
│   ├── verification_system_prompt.txt
│   └── ...
├── problems/              # IMO 2025 problems
│   ├── imo01.txt
│   ├── imo02.txt
│   └── ...
├── .env.example           # Example environment configuration
├── .envrc                 # Direnv configuration
├── Makefile              # Build and run targets
└── README.md             # This file
```

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

## Understanding the Output

### Solution Detection
The system looks for the phrase "Found a correct solution in run" to identify successful solutions.

### Agent Behavior
- Agents use Google's Gemini 2.5 Pro model
- Each agent follows a structured approach with:
  - Initial solution generation
  - Self-improvement step
  - Verification and correction loop (up to 30 iterations)
- Solutions are verified for completeness and correctness
- Agents can provide partial solutions if complete solutions aren't found
- The system requires 5 consecutive successful verifications before accepting a solution

## Tips for Best Results

1. **Problem Formatting**: Ensure your problem file is clear and well-formatted
2. **Parallel Execution**: Use more agents for harder problems (10-20 agents recommended)
3. **Timeout Settings**: Set reasonable timeouts (you may set no timeout)
4. **API Limits**: Be aware of Google API rate limits and costs
5. **Log Analysis**: Check individual agent logs for detailed reasoning

## Troubleshooting

### Common Issues

1. **API Key Error**: 
   - Ensure your `GEMINI_API_KEY` is properly set in `.env`
   - Verify the key starts with "AIza" using: `echo $GEMINI_API_KEY | cut -c 1-4`
   - Get a valid key from: https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro

2. **Timeout Issues**: 
   - The thinking budget feature may cause long response times
   - Increase timeout or reduce number of agents
   - Consider running with `--max_runs 1` for testing

3. **Memory Issues**: Reduce max-workers if running out of memory

4. **No Solutions Found**: Try running more agents or check problem clarity

### Debug Mode
Add verbose logging by modifying the agent code or check individual log files for detailed output.

## License

MIT License - Copyright (c) 2025 Lin Yang, Yichen Huang

This software is provided as-is. Users are free to copy, modify, and distribute the code with proper attribution.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

### Community Codes

Community contributions are located in `code/community_codes/`. These have not been thoroughly tested, so please use them at your own risk. 

## Disclaimer

This tool is for educational and research purposes. Success in solving IMO problems depends on problem difficulty and AI model capabilities. Not all problems may be solvable by the current system.

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
