# Data

`raw/fraud_oracle.csv` — Vehicle Insurance Claim Fraud Detection

| | |
|---|---|
| Rows / columns | 15,420 / 33 |
| Target | `FraudFound_P` (1 = fraud found), 923 positives (5.99%) |
| Period | 1994–1996 |
| Original distribution | Kaggle: `shivamb/vehicle-claim-fraud-detection` |
| Licence | CC0: Public Domain. Not read from the Kaggle page directly; taken from a peer-reviewed article that cites the dataset under this licence (https://www.mdpi.com/2571-5577/6/1/20) |
| Downloaded from | `https://raw.githubusercontent.com/rezamosavi8740/Vehicle-Insurance-Claim-Fraud-Detection/main/data.csv` |
| SHA-256 | `914d6c63a0d5e19b81f3ac0f2e2a45c8284b0a75c87233c3e5a7dcbeb806ac23` |

The file is stored unmodified. All cleaning happens in code (`src/data.py`).

**Provenance.** The data is widely used in the academic fraud-detection literature and is
reported to have been collected with the Angoss KnowledgeSEEKER software between January 1994
and December 1996. The original publisher's release could not be traced, so the dataset should
be treated as a teaching and benchmarking resource, not as a description of any current
insurance portfolio.
