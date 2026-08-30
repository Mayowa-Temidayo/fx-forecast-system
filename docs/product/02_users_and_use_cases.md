# Users and Use Cases

## 1. Primary User

The primary modeled user is a treasury or risk professional responsible for managing foreign-currency exposure.

The user needs to understand potential FX risk before a foreign-currency obligation reaches its settlement date.

The system supports the user's analysis but does not replace professional judgment.

---

## 2. Secondary Users

### 2.1 Finance Manager

A finance manager may use the system to understand the potential financial impact of foreign-exchange movements on upcoming obligations.

### 2.2 Risk Analyst

A risk analyst may use the system to compare historical performance of different hedging strategies and investigate model assumptions.

### 2.3 ML Engineer

An ML engineer may use the system to inspect forecasts, model performance, data quality, model versions, and system health.

---

## 3. Primary Use Case — Evaluate an FX Exposure

A treasury user provides:

- Currency
- Exposure amount
- Settlement date
- Current hedge percentage
- Risk tolerance

The system evaluates the exposure against the relevant FX forecast and historical scenarios.

### Expected outcome

The user receives:

- Forecast information
- Forecast uncertainty
- Exposure risk
- Candidate hedge strategies
- Historical strategy performance
- Recommended strategy
- Supporting evidence
- Decision assumptions

---

## 4. Use Case — Forecast FX Risk

The user requests a forecast for a supported currency and horizon.

The system should provide:

- Point forecast where appropriate
- Prediction interval
- Forecast horizon
- Model version
- Forecast timestamp

The forecast must be accompanied by appropriate uncertainty information.

---

## 5. Use Case — Compare Hedging Strategies

The user evaluates alternative hedge ratios for an exposure.

The system compares strategies such as:

- 0% hedge
- 25% hedge
- 50% hedge
- 75% hedge
- 100% hedge

The system reports the simulated cost, downside protection, and other relevant risk measures for each strategy.

---

## 6. Use Case — Receive a Recommendation

The user provides an exposure and risk configuration.

The decision engine evaluates available strategies using the configured assumptions and produces a recommendation.

The recommendation must include:

- Recommended hedge ratio
- Risk assessment
- Relevant forecast information
- Key assumptions
- Supporting evidence

The recommendation must be reproducible from its recorded inputs and configuration.

---

## 7. Use Case — Investigate Market Intelligence

The user asks what recent developments may be relevant to the FX exposure.

The system retrieves relevant news and market information and summarizes the evidence.

The system should distinguish between:

- Retrieved facts
- Model-generated interpretation
- Uncertainty
- Unsupported or unavailable information

---

## 8. Use Case — Understand a Recommendation

The user wants to know why a strategy was recommended.

The system provides an explanation connecting the recommendation to:

1. The FX forecast
2. Forecast uncertainty
3. Exposure characteristics
4. Historical strategy performance
5. Configured risk assumptions
6. Relevant market evidence

The language model may generate the explanation, but it must not override the deterministic decision engine.

---

## 9. Use Case — Review Historical Performance

The user investigates how the forecasting and hedging strategies performed historically.

The system provides:

- Forecast errors
- Baseline comparisons
- Strategy performance
- Downside outcomes
- Relevant evaluation periods
- Known limitations

---

## 10. Use Case — Audit a Recommendation

An authorized user should be able to determine:

- When the recommendation was generated
- Which exposure was evaluated
- Which forecast was used
- Which model version was used
- Which assumptions were applied
- Which strategy was selected
- Which evidence supported the explanation

This creates an auditable decision trail.

---

## 11. Non-Goals

The initial system will not:

- Execute trades
- Automatically enter hedge contracts
- Guarantee profits
- Guarantee loss avoidance
- Replace professional treasury judgment
- Provide personalized financial advice
- Allow an LLM to independently determine financial actions

---

## 12. Core User Journey

The primary user journey is:

Exposure
→ Forecast
→ Risk Analysis
→ Strategy Simulation
→ Decision
→ Evidence
→ Explanation
→ Audit Record

This sequence defines the core interaction model for the system.