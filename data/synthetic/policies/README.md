# Synthetic Internal Policy Corpus

**These documents are entirely fictional.** They describe a fictional bank
("Meridian Small Finance Bank Ltd.") and its internal compliance policies,
written for the purpose of demonstrating regulatory-change impact analysis
against **real** RBI regulatory documents. No real organization, real
policy, or real internal document is represented here.

## Why synthetic policies, and why these six

RegDelta's core capability is mapping a real regulatory change to the
internal policies it affects. We don't have access to any real
organization's compliance library, so per the project's design, we
constructed a small, realistic one instead.

Rather than covering every RBI regulatory domain, these six policies are
deliberately scoped to what's **actually present** in the real fetched RBI
corpus (see `notebooks/01-raw-rbi-data-eda.ipynb` and the fetcher output in
`data/raw/`) — dominated by interest-rate-on-deposits amendments and CRR/SLR
(Cash Reserve Ratio / Statutory Liquidity Ratio) amendments across several
bank categories. Testing against real matches (and real non-matches) is more
useful than testing against a large but hollow policy set with nothing real
to compare against.

| # | Policy | Expected relevance to current RBI corpus |
|---|---|---|
| POL-001 | Deposit Interest Rate Policy | **Relevant** — directly affected by the Interest Rate on Deposits Third Amendment Directions |
| POL-002 | Reserve Requirements & Liquidity Management Policy | **Relevant** — directly affected by the CRR/SLR Fourth Amendment Directions |
| POL-003 | NRE/NRO/FCNR(B) Deposit Management Policy | **Relevant** — narrower, clause-level relevance to the same amendments (the specific relaxation-deadline provision) |
| POL-004 | KYC and Customer Due Diligence Policy | Control — no current regulatory match expected |
| POL-005 | Fraud Risk Management Policy | Control — no current regulatory match expected (mirrors the exact negative eval case in `src/evaluation/eval_cases.py`) |
| POL-006 | Priority Sector Lending Policy | Control — no current regulatory match expected |

## A deliberate, realistic gap

POL-001, POL-002, and POL-003 are written citing the **pre-amendment**
relaxation deadline of **September 30, 2026** — the exact date the real RBI
notifications (e.g. `RBI/2026-27/243`) just amended to **August 31, 2026**.
This isn't an error; it's the realistic scenario this whole project exists
to catch: a policy that hasn't caught up with a just-issued regulatory
change. Future gap-analysis logic (M8+) should be able to detect this
specific, concrete discrepancy.

## Fictional organization profile

**Meridian Small Finance Bank Ltd.** — a fictional RBI-regulated small
finance bank, for the purposes of this demonstration only. Any resemblance
to a real institution is coincidental and unintended.
