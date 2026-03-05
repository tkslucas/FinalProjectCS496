# Poker Agent in a Simulated Environment

CS 496 Agentic AI Final Project -- Northwestern University

**Team**: Abhi Vinnakota, Caleb Weldon, Eric Fei, Frank Xin, Lucas Takayasu

## Overview

A tool-augmented poker agent that plays No-Limit Texas Hold'em in a simulated environment with 2-6 players. The agent uses **GPT-4.1** via the **OpenAI Agents SDK**, augmented with three decision-support tools:

- **Equity Calculator** -- Monte Carlo simulation to estimate win probability against opponents
- **Legal Action Validator** -- queries the game engine for valid moves (fold, check, call, raise) with bet sizing constraints
- **RAG Similar-Hand Retriever** -- finds historically similar hands via feature-based similarity for strategic context

Each decision produces a reasoning chain (tool calls, inputs/outputs, final action) that is logged to JSON for post-game analysis.

## Architecture

```
                  +------------------+
                  |   GPT-4.1 Agent  |
                  |  (OpenAI Agents  |
                  |      SDK)        |
                  +--------+---------+
                           |
              +------------+------------+
              |            |            |
     +--------v--+  +------v------+  +--v-----------+
     |  Equity   |  |   Legal     |  |  RAG Hand    |
     | Calculator|  |  Action     |  |  Retriever   |
     | (Monte    |  |  Validator  |  |  (feature    |
     |  Carlo)   |  |             |  |  similarity) |
     +-----------+  +-------------+  +--------------+
              |            |            |
              +------------+------------+
                           |
                  +--------v---------+
                  |    PokerKit      |
                  |  (NLHE Engine)   |
                  +------------------+
                           |
                  +--------v---------+
                  |  Reasoning Logger|
                  |  (JSON export)   |
                  +------------------+
```

## Setup

```bash
pip install -r requirements.txt
```

Dependencies:
- `pokerkit>=0.7.0` -- poker simulation engine
- `openai-agents>=0.1.0` -- OpenAI Agents SDK for GPT-4.1

## Usage

### Baseline simulation (no API key needed)

```bash
# 100 hands, 6 players: heuristic vs random/call-station
python -m poker_agent.main -n 100 -p 6 --agents heuristic random call random heuristic random -v
```

### LLM agent simulation (requires OPENAI_API_KEY)

```bash
export OPENAI_API_KEY=your_key_here
python -m poker_agent.main -n 50 -p 4 --agents llm random call heuristic -v
```

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of hands | 100 |
| `-p` | Number of players (2-6) | 6 |
| `-s` | Starting stack (chips) | 200 |
| `--small-blind` | Small blind | 1 |
| `--big-blind` | Big blind | 2 |
| `--agents` | Agent type per seat | heuristic + random |
| `-v` | Verbose output | off |

Agent types: `llm`, `heuristic`, `random`, `call`

## Evaluation

Performance is measured by **bb/100** (big blinds won per 100 hands). Example output:

```
============================================================
Session Summary: 100 hands played
============================================================
Agent                  bb/100   Profit   Win%  Hands
------------------------------------------------------------
heuristic_0            251.00    502.0   2.0%    100
random_1               315.00    630.0  32.0%    100
call_2                2444.50   4889.0  49.0%    100
random_3             -1287.00  -2574.0  14.0%    100
============================================================
```

Reasoning logs are exported to `logs/` as JSON after each session for post-game analysis.

## Project Structure

```
poker_agent/
  environment.py          PokerKit wrapper (game creation, dealing, actions)
  game_state.py           Structured game state representation
  agent.py                GPT-4.1 agent with tool integration
  tools/
    equity.py             Monte Carlo equity calculator
    validator.py          Legal action validator
    rag.py                RAG hand history retriever
  baselines/
    random_agent.py       Random action baseline
    call_agent.py         Always check/call baseline
    heuristic_agent.py    Equity-threshold strategy
  evaluation/
    metrics.py            bb/100, session tracking, leaderboard
  reasoning/
    logger.py             Per-decision reasoning chain logger
  main.py                 CLI entry point
```
