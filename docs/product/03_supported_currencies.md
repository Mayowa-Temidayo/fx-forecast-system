# Supported Currencies

## 1. Scope

FX Treasury Copilot initially supports two Nigerian-naira currency pairs:

* USD/NGN
* EUR/NGN

USD/NGN is the primary implementation target.

EUR/NGN is the secondary target and will be introduced after the core forecasting and risk pipeline has been validated with USD/NGN.

## 2. Canonical Representation

Currency pairs are represented using:

`BASE/QUOTE`

Examples:

* USD/NGN
* EUR/NGN

The quote represents the amount of NGN required to purchase one unit of the base currency.

## 3. Primary Pair

### USD/NGN

USD/NGN is the primary currency pair because it represents the main FX exposure considered by the initial system.

The complete forecasting, risk, hedging simulation, and decision workflow must work reliably for USD/NGN before EUR/NGN becomes a required implementation target.

## 4. Secondary Pair

### EUR/NGN

EUR/NGN extends the system beyond a single currency pair and tests whether the architecture can generalize across supported instruments.

EUR/NGN must use the same domain interfaces as USD/NGN.

Pair-specific behavior should only exist where the underlying market or data source genuinely requires it.

## 5. Domain Symbols vs Provider Symbols

The system distinguishes between canonical domain pairs and external data-source identifiers.

For example:

`USD/NGN`

is a domain-level currency pair.

A provider may represent the same instrument using a provider-specific identifier such as:

`USDNGN=X`

The provider identifier must not become the canonical representation used throughout the application.

This separation allows additional data sources to be introduced without changing the domain model.

## 6. Implementation Sequence

The implementation sequence is:

1. USD/NGN data ingestion
2. USD/NGN data validation
3. USD/NGN forecasting
4. USD/NGN uncertainty evaluation
5. USD/NGN hedging simulation
6. USD/NGN decision workflow
7. Generalize the validated interfaces
8. Add EUR/NGN
9. Validate both pairs

## 7. Out of Scope

The initial system does not target:

* EUR/USD
* USD/JPY
* GBP/USD
* Other international FX pairs
* Cryptocurrency pairs

Additional pairs may be introduced in a future version without changing the core domain architecture.

## 8. Design Principle

The system should be currency-pair aware without being currency-pair hardcoded.

Business logic should operate on canonical currency pairs.

Data-source-specific identifiers should remain inside the relevant ingestion adapter.

This separation allows the forecasting, risk, decision, API, and monitoring layers to operate consistently across supported currency pairs.
