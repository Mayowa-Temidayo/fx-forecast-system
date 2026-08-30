# Business Problem Definition

## 1. Problem Statement

Businesses with foreign-currency obligations are exposed to uncertainty in the future NGN cost of those obligations.

For a Nigerian business that must settle a USD or EUR obligation at a future date, an adverse exchange-rate movement can materially increase the naira cost of settlement.

The business therefore needs to understand:

1. How the exchange rate may evolve over the exposure horizon.
2. How uncertain that forecast is.
3. How much financial risk the exposure creates.
4. How different hedging strategies would have performed under historical market conditions.
5. Which strategy provides an acceptable balance between downside protection and hedging cost.

FX Treasury Copilot is designed to support this decision-making process.

---

## 2. Core Business Problem

The system addresses the following problem:

> Given a foreign-currency exposure, settlement horizon, forecast distribution, market conditions, and defined risk assumptions, evaluate alternative hedging strategies and provide an evidence-backed decision recommendation.

The system is a decision-support tool. It does not guarantee exchange-rate predictions, profits, or loss avoidance.

---

## 3. Primary User

The primary modeled user is a treasury or risk professional at a Nigerian business with foreign-currency exposure.

The user may need to evaluate exposures such as:

- Supplier payments
- Foreign-denominated operating expenses
- International service obligations
- Other contractual USD or EUR liabilities

The project does not assume that the system is connected to an actual financial institution or customer.

---

## 4. Primary Use Case

The core workflow is:

1. User provides an FX exposure.
2. The system identifies the relevant forecast horizon.
3. The forecasting engine estimates the future exchange-rate distribution.
4. The risk engine evaluates the exposure under different market scenarios.
5. The hedging simulator compares alternative hedge ratios.
6. The decision engine evaluates the available strategies.
7. News intelligence provides relevant supporting market information.
8. The system produces an auditable recommendation and explanation.

---

## 5. Example Exposure

A representative scenario may contain:

| Field              | Example  |
| ------------------ | -------- |
| Currency           | USD      |
| Exposure           | $500,000 |
| Settlement horizon | 14 days  |
| Risk tolerance     | Moderate |
| Current hedge      | 0%       |

The example is illustrative only and does not represent an actual customer.

---

## 6. Business Objective

The primary objective is to build a decision-support platform that combines:

- FX forecasting
- Forecast uncertainty
- Exposure analysis
- Historical hedge simulation
- Market-news intelligence
- Deterministic risk logic
- Evidence-backed explanations

The system should help users understand the potential consequences of different hedging strategies rather than simply provide a point exchange-rate prediction.

---

## 7. Success Measures

Success will be evaluated at two levels.

### 7.1 Forecasting Performance

The forecasting system will be evaluated using appropriate time-series validation and metrics such as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy
- Forecast Bias
- Prediction Interval Coverage
- Prediction Interval Width

### 7.2 Risk and Decision Performance

The decision system will evaluate:

- Simulated unhedged cost
- Simulated hedged cost
- Downside exposure
- Risk reduction
- Hedging cost
- Opportunity cost
- Strategy performance across historical scenarios

No business-performance result will be claimed until it has been produced by an explicit experiment or backtest.

---

## 8. System Positioning

FX Treasury Copilot is a decision-support and scenario-analysis system.

It is not:

- An autonomous trading system
- A guaranteed FX prediction system
- A guaranteed profit-generation system
- A substitute for professional treasury judgment
- A source of personalized financial advice

Final financial decisions remain with the user.

---

## 9. Core Design Principle

The system separates prediction, risk evaluation, and explanation.

The forecasting engine estimates possible future FX outcomes.

The risk engine evaluates those outcomes against the user's exposure.

The decision engine applies deterministic and configurable risk logic.

The language model, where used, provides evidence retrieval, synthesis, and explanation.

The language model does not independently determine the hedge ratio.

---

## 10. Project Boundary

The initial system will focus on:

- USD/NGN
- EUR/NGN
- Short- to medium-term FX exposure
- Historical scenario analysis
- Forecasting
- Hedge strategy evaluation
- Market-news intelligence
- Decision support

The initial system will not execute real financial transactions.

---

## 11. Expected Outcome

The completed system should allow a user to move from:

> "I have a foreign-currency obligation and I am uncertain about the FX risk."

to:

> "Here is the forecast, here is the uncertainty, here is how alternative hedging strategies performed under comparable historical conditions, here is the strategy selected under the configured risk assumptions, and here is the evidence supporting the recommendation."

This represents the central business purpose of FX Treasury Copilot.