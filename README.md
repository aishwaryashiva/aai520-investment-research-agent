# Investment Research Agent

AAI-520 Group 6:

- **Sandeep Bhargava** owned the news and sentiment specialist, including the headline chain and the routing of earnings and macro stories.
- **Ashish Kumar** owned the earnings and market specialists, plus the rubric that scores the brief and sends thin drafts back for revision.
- **Aishwarya Pareek** owned the orchestrator, cross-run memory, and the notebook walkthrough that ties the desk together.

We research one ticker with an orchestrator and three specialist agents that
work from separate packets.

- **Research Orchestrator** plans the tools, routes each piece of content,
  and writes the final brief.
- **News Sentiment Specialist** runs the headline chain (ingest, preprocess,
  classify, extract, summarize) and writes the tone note.
- **Earnings Fundamental Specialist** reads ratios, the income statement, and
  earnings headlines.
- **Market Technical Specialist** reads the chart: return, volatility,
  drawdown, the 20-day and 50-day moving averages, and RSI(14).

The orchestrator does not do that reading itself. It hands each agent a
packet and keeps their notes. After the brief is drafted, a rubric scores it
and a revision pass fills whatever the score marked as thin. The run then
stores a short memory note. The next run for the same ticker reads that note,
which is why the second Microsoft pass pulls the income statement.

Yahoo Finance is the data source (`yfinance`): prices, fundamentals, analyst
targets, statements, and headlines.

## Layout

```text
aai520_research/agent.py      orchestrator
aai520_research/agents.py     the three specialists
aai520_research/prompt_chain.py
aai520_research/router.py
aai520_research/evaluator.py
aai520_research/memory.py
notebooks/Investment_Research_Agent.ipynb
```

## Setup

```bash
python -m pip install -r requirements.txt
```

With no API key the desk uses the local handlers in `heuristic.py`, which
follow the same prompts as a hosted model. To use a hosted model, copy
`.env.example` to `.env` and set `OPENAI_API_KEY`. `OPENAI_BASE_URL` can
point at any OpenAI-compatible endpoint.

## Run

```bash
python -m aai520_research MSFT
```

GitHub: https://github.com/aishwaryashiva/aai520-investment-research-agent

The walkthrough with saved output is
`notebooks/Investment_Research_Agent.ipynb`.

```bash
python -m pytest
python -m flake8 aai520_research tests
```

We kept the line length at 99, which is the limit we agreed on in `setup.cfg`.

Memory for the command line agent is `data/memory/research_memory.json`.
The notebook uses `data/memory/notebook_demo.json`. Neither file is committed.

This is a class project. The brief is not investment advice.
