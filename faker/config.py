"""
config.py  --  STAGE 0: all design decisions live here.

Everything you might want to tweak (how many employees, how much each factor
matters, how much mess to inject) is centralized in this file so the rest of
the code just *reads* these choices. This is where you play "the universe" and
decide how your fake company behaves.
"""

import os

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# A fixed seed means every run produces the SAME dataset. Turn this into a
# different number (or None) if you want a fresh draw each time.
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Size of the dataset
# --------------------------------------------------------------------------
N_EMPLOYEES = 10_000

# --------------------------------------------------------------------------
# Risk-score WEIGHTS  (Activity A: chosen by us, once, up front)
# --------------------------------------------------------------------------
# How much each driver pushes someone toward leaving. These are DELIBERATE
# beliefs about attrition, not random numbers. They sum to 1.0 purely for
# interpretability ("satisfaction is 35% of the risk story"). Only the RATIOS
# between them actually matter mathematically.
WEIGHTS = {
    "low_satisfaction": 0.35,   # unhappiness is the strongest driver
    "underpaid":        0.30,   # pay relative to peers
    "no_promotion":     0.20,   # feeling stuck / career stagnation
    "long_commute":     0.10,   # annoying but tolerable
    "high_overtime":    0.05,   # burnout, smaller marginal push
}

# --------------------------------------------------------------------------
# Label calibration
# --------------------------------------------------------------------------
# Target share of employees who leave. Real attrition is imbalanced (~10-15%).
TARGET_ATTRITION_RATE = 0.13

# How sharply the risk score separates leavers from stayers in logit space.
# Higher = stronger signal (drivers matter more). The bias/intercept is found
# automatically to hit TARGET_ATTRITION_RATE, so you rarely touch this.
RISK_SCALE = 6.0

# Size of the random "life is unpredictable" nudge added to each risk score.
# This is what guarantees the model can never hit 100% accuracy.
NOISE_STD = 0.05

# --------------------------------------------------------------------------
# Messiness rates  (STAGE 4: fraction of rows affected by each injector)
# --------------------------------------------------------------------------
# Kept modest so cleaning is a real task but the data isn't destroyed.
MESS = {
    "date_format_variants": 0.30,   # reformat some hire dates
    "category_typos":       0.08,   # Sales -> sales / SALES / Salez
    "missing_mcar":         0.03,   # random blanks, no pattern
    "missing_mnar_salary":  0.25,   # high earners hide salary (informative!)
    "duplicate_rows":       0.01,   # fraction of rows duplicated
    "whitespace_nulls":     0.05,   # stray spaces, "N/A"/""/unknown, "50k"
    "impossible_values":    0.005,  # age=250, hire after termination, etc.
}

# --------------------------------------------------------------------------
# Paths  (computed relative to THIS file so it works from any working dir)
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))

# The dirty file we deliver (what the ML pipeline will consume).
DIRTY_CSV = os.path.join(DATA_DIR, "Employee_data.csv")
# The pristine "answer key" we keep aside to check our cleaning later.
CLEAN_CSV = os.path.join(DATA_DIR, "Employee_data_clean.csv")
