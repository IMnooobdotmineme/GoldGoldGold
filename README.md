# AI Gold Market Risk Monitoring Agent

## Project Overview

This project is an **AI-powered Gold Market Risk Monitoring Agent** that automatically analyzes market conditions, predicts gold market volatility, uses a Large Language Model for reasoning, and decides whether an urgent Telegram alert should be sent.

The main goal is to help users monitor gold market risk more efficiently by combining real market data, machine learning, LLM reasoning, and automated notifications.

---

## Problem Statement

Gold prices can change quickly because of market volatility, the US dollar, interest rates, and other economic factors.

Manually monitoring these indicators takes time and may cause users to miss important market changes.

This project solves this problem by creating an AI agent that can:

- Collect real market data automatically
- Predict future gold volatility
- Analyze market conditions using an LLM
- Determine the level of market risk
- Decide whether an urgent alert is needed
- Send a Telegram notification automatically

---

## AI Agent Solution

The system combines several AI and automation components.

It uses:

- **Yahoo Finance** for real market data
- **GRU Neural Network** for volatility prediction
- **Groq LLM** for market reasoning and decision-making
- **Telegram Bot API** for automatic alerts

The agent does not simply generate an answer. It analyzes the situation, makes decisions, and changes its workflow depending on the market condition.

---

## Workflow

The AI agent follows this workflow:

```text
Step 1: Fetch market data from Yahoo Finance
                ↓
Step 2: Prepare recent market features
                ↓
Step 3: GRU model predicts next-day gold volatility
                ↓
Step 4: LLM analyzes the market situation
                ↓
Step 5: AI Decision Point 1
        Determine Risk Level
        LOW / ELEVATED / HIGH
                ↓
Step 6: AI Decision Point 2
        Should an urgent alert be sent?
                ↓
          YES         NO
           ↓           ↓
 Send Telegram     No urgent alert
     Alert
           ↓
Step 7: Generate Final AI Agent Report
