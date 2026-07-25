"""
generate.py  --  the staged pipeline that builds the dataset.

Stages
------
1. build_clean_features   -> realistic, inter-correlated raw features
2. add_probabilistic_label -> normalize drivers -> weighted risk -> sigmoid -> SAMPLE
3. (save clean truth is done in run.py)
4. inject_messiness       -> deliberately break a COPY, features only, never the label
5. (save dirty file is done in run.py)
6. sanity_checks          -> prove the data behaves as designed before we build on it
"""

import numpy as np
import pandas as pd
from faker import Faker

import config


# ==========================================================================
# small helpers
# ==========================================================================
def _minmax(x):
    """Squeeze an array onto a 0-1 scale so weights control influence,
    not the raw number ranges. (Normalization.)"""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo == 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _sigmoid(x):
    """Squish any number into a 0-1 probability."""
    return 1.0 / (1.0 + np.exp(-x))


# ==========================================================================
# STAGE 1 -- clean, plausible, inter-correlated features
# ==========================================================================
def build_clean_features(rng, faker):
    n = config.N_EMPLOYEES

    # --- job level drives a lot, so draw it first (more juniors than execs) ---
    job_level = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.30, 0.30, 0.22, 0.13, 0.05])

    # --- age: independent-ish but bounded to sensible working range ---
    age = np.clip(rng.normal(38, 9, n).round().astype(int), 22, 60)

    # --- income depends on job level (+ mild age bump) with lognormal spread ---
    base_by_level = {1: 3200, 2: 5000, 3: 7500, 4: 11000, 5: 16000}
    base = np.array([base_by_level[lv] for lv in job_level])
    income = (base * (1 + (age - 38) * 0.004) * rng.lognormal(0, 0.15, n))
    monthly_income = income.round(0).astype(int)

    # --- tenure cannot exceed working years; capped by age ---
    max_tenure = np.clip(age - 22, 0, 40)
    years_at_company = np.floor(rng.uniform(0, 1, n) ** 1.5 * (max_tenure + 1)).astype(int)
    years_at_company = np.minimum(years_at_company, max_tenure)

    # --- years since last promotion: bounded by tenure ---
    years_since_last_promotion = np.floor(
        rng.uniform(0, 1, n) * (years_at_company + 1)
    ).astype(int)

    # --- commute distance (km): most people close, a tail of long commutes ---
    distance_from_home_km = np.clip(rng.exponential(9, n).round(1), 1, 60)

    # --- overtime flag (~28% work overtime) ---
    overtime = np.where(rng.uniform(0, 1, n) < 0.28, "Yes", "No")

    # --- satisfaction 1-5, nudged DOWN a bit by heavy overtime + long commute
    #     (adds realism: miserable conditions correlate with low satisfaction) ---
    sat_base = rng.normal(3.4, 1.0, n)
    sat_base -= (overtime == "Yes") * 0.4
    sat_base -= _minmax(distance_from_home_km) * 0.5
    satisfaction_score = np.clip(np.round(sat_base), 1, 5).astype(int)

    # --- other contextual / cosmetic features ---
    department = rng.choice(
        ["Sales", "Research & Development", "Human Resources", "Engineering", "Marketing"],
        size=n, p=[0.30, 0.30, 0.10, 0.20, 0.10],
    )
    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n, p=[0.20, 0.45, 0.28, 0.07],
    )
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    marital_status = rng.choice(["Single", "Married", "Divorced"], size=n, p=[0.35, 0.50, 0.15])
    num_projects = np.clip(rng.poisson(3, n), 1, 10)
    avg_monthly_hours = np.clip(rng.normal(170, 25, n) + (overtime == "Yes") * 30, 90, 320).round().astype(int)

    # --- identity / cosmetic (Faker) ---
    employee_id = np.arange(1, n + 1)
    names = [faker.name() for _ in range(n)]
    emails = [
        f"{nm.split()[0].lower()}.{nm.split()[-1].lower()}@examplecorp.com" for nm in names
    ]
    city = [faker.city() for _ in range(n)]
    # hire date consistent with tenure (roughly today minus years_at_company)
    hire_date = [
        faker.date_between(start_date=f"-{int(t)*365 + 30}d", end_date=f"-{int(t)*365}d")
        for t in years_at_company
    ]

    # --- PURE NOISE columns: deliberately NOT in the risk formula.
    #     SHAP should later rank these near zero. They are our honesty check. ---
    badge_id = [faker.bothify("BDG-####-????") for _ in range(n)]
    survey_response_id = rng.integers(100000, 999999, n)

    df = pd.DataFrame(
        {
            "employee_id": employee_id,
            "name": names,
            "email": emails,
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "department": department,
            "job_level": job_level,
            "education": education,
            "monthly_income": monthly_income,
            "years_at_company": years_at_company,
            "years_since_last_promotion": years_since_last_promotion,
            "distance_from_home_km": distance_from_home_km,
            "overtime": overtime,
            "num_projects": num_projects,
            "avg_monthly_hours": avg_monthly_hours,
            "satisfaction_score": satisfaction_score,
            "city": city,
            "hire_date": hire_date,
            "badge_id": badge_id,             # noise
            "survey_response_id": survey_response_id,  # noise
        }
    )
    return df


# ==========================================================================
# STAGE 2 -- probabilistic label (the part that must NOT leak a rule)
# ==========================================================================
def add_probabilistic_label(df, rng):
    # ---- build each normalized driver, oriented so HIGHER = MORE risk ----

    # 1. low satisfaction: invert the 1-5 score
    low_satisfaction = _minmax(5 - df["satisfaction_score"])

    # 2. underpaid RELATIVE TO PEERS at the same job level (not an absolute cutoff)
    peer_median = df.groupby("job_level")["monthly_income"].transform("median")
    underpaid_raw = (peer_median - df["monthly_income"]) / peer_median
    underpaid = _minmax(underpaid_raw.clip(lower=0))  # only being BELOW peers counts

    # 3. no recent promotion
    no_promotion = _minmax(df["years_since_last_promotion"])

    # 4. long commute
    long_commute = _minmax(df["distance_from_home_km"])

    # 5. high overtime
    high_overtime = (df["overtime"] == "Yes").astype(float).values

    drivers = {
        "low_satisfaction": low_satisfaction,
        "underpaid": underpaid,
        "no_promotion": no_promotion,
        "long_commute": long_commute,
        "high_overtime": high_overtime,
    }

    # ---- weighted sum + noise = hidden risk score ----
    risk = np.zeros(len(df))
    for name, w in config.WEIGHTS.items():
        risk = risk + w * np.asarray(drivers[name], dtype=float)
    risk = risk + rng.normal(0, config.NOISE_STD, len(df))

    # ---- move to logit space, then AUTO-CALIBRATE the bias to hit target rate ----
    centered = config.RISK_SCALE * (risk - risk.mean())
    bias = _calibrate_bias(centered, config.TARGET_ATTRITION_RATE)
    prob_leave = _sigmoid(centered + bias)

    # ---- SAMPLE the label (do NOT threshold) ----
    attrition = np.where(rng.uniform(0, 1, len(df)) < prob_leave, "Yes", "No")
    df = df.copy()
    df["attrition"] = attrition
    return df


def _calibrate_bias(centered, target_rate, iters=60):
    """Find the intercept so the AVERAGE leave-probability ~= target rate.
    Simple bisection: this is the 'master dial' for the overall quit rate."""
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = _sigmoid(centered + mid).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ==========================================================================
# STAGE 4 -- messiness injectors (operate on a COPY, features only)
# ==========================================================================
def inject_messiness(clean_df, rng):
    df = clean_df.copy()
    n = len(df)

    def pick(frac):
        """random row indices covering the given fraction of rows"""
        k = int(n * frac)
        return rng.choice(n, size=k, replace=False)

    # 1. inconsistent date formats -------------------------------------------
    df["hire_date"] = df["hire_date"].astype(str)
    idx = pick(config.MESS["date_format_variants"])
    for i in idx:
        d = pd.to_datetime(df.at[i, "hire_date"])
        fmt = rng.choice(["%m/%d/%Y", "%d-%m-%Y", "%B %d %Y", "%d %b %Y"])
        df.at[i, "hire_date"] = d.strftime(fmt)

    # 2. categorical typos / casing (department) -----------------------------
    variants = {
        "Sales": ["sales", "SALES", "Salez", " Sales"],
        "Human Resources": ["human resources", "HR", "Human resources"],
        "Marketing": ["marketing", "MARKETING", "Marketng"],
    }
    idx = pick(config.MESS["category_typos"])
    for i in idx:
        dep = df.at[i, "department"]
        if dep in variants:
            df.at[i, "department"] = rng.choice(variants[dep])

    # 3a. missing completely at random (MCAR) -- scattered, no pattern -------
    for col in ["education", "num_projects", "distance_from_home_km", "marital_status"]:
        idx = pick(config.MESS["missing_mcar"])
        df.loc[idx, col] = np.nan

    # 3b. missing NOT at random (MNAR) -- high earners hide their salary -----
    high_earners = df.index[df["monthly_income"] > df["monthly_income"].quantile(0.75)]
    k = int(len(high_earners) * config.MESS["missing_mnar_salary"])
    if k > 0:
        idx = rng.choice(high_earners.to_numpy(), size=k, replace=False)
        df.loc[idx, "monthly_income"] = np.nan

    # 4. duplicate records (some exact, some with a tiny variation) ----------
    k = int(n * config.MESS["duplicate_rows"])
    if k > 0:
        dup_idx = rng.choice(n, size=k, replace=False)
        dups = df.loc[dup_idx].copy()
        # tweak the name slightly on half of them (fuzzy duplicates)
        for j in dups.index[: k // 2]:
            dups.at[j, "name"] = str(dups.at[j, "name"]) + " "
        df = pd.concat([df, dups], ignore_index=True)

    # 5. whitespace / fake nulls / mixed units -------------------------------
    # allow mixed types (e.g. "5k") to sit alongside numbers in these columns
    df["monthly_income"] = df["monthly_income"].astype(object)
    df["marital_status"] = df["marital_status"].astype(object)
    df["department"] = df["department"].astype(object)
    idx = pick(config.MESS["whitespace_nulls"])
    for i in idx:
        r = rng.uniform(0, 1)
        if r < 0.34:
            df.at[i, "department"] = f"  {df.at[i, 'department']}  "  # stray whitespace
        elif r < 0.67:
            df.at[i, "marital_status"] = rng.choice(["N/A", "unknown", ""])  # fake nulls
        else:
            inc = df.at[i, "monthly_income"]
            if pd.notna(inc):
                df.at[i, "monthly_income"] = f"{int(int(inc)/1000)}k"  # mixed units "5k"

    # 6. impossible values ---------------------------------------------------
    idx = pick(config.MESS["impossible_values"])
    for i in idx:
        c = rng.choice(["age", "distance", "promo"])
        if c == "age":
            df.at[i, "age"] = int(rng.choice([250, -5, 0]))
        elif c == "distance":
            df.at[i, "distance_from_home_km"] = -abs(df.at[i, "distance_from_home_km"])
        else:
            df.at[i, "years_since_last_promotion"] = int(df.at[i, "age"]) + 5  # > possible tenure

    # shuffle so duplicates aren't glued to the bottom
    df = df.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    return df


# ==========================================================================
# STAGE 6 -- sanity checks (prove the data before building on it)
# ==========================================================================
def sanity_checks(clean_df):
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check 1: overall attrition rate in target band?
    rate = (clean_df["attrition"] == "Yes").mean()
    flag = "OK" if 0.10 <= rate <= 0.15 else "!! out of 10-15% band"
    print(f"[1] Attrition rate: {rate:.1%}   ({flag})")

    # Check 2: do the intended drivers actually separate leavers vs stayers?
    print("\n[2] Driver means (leavers should look WORSE):")
    grp = clean_df.groupby("attrition")
    for col in ["satisfaction_score", "monthly_income",
                "years_since_last_promotion", "distance_from_home_km"]:
        stay = grp.get_group("No")[col].mean()
        leave = grp.get_group("Yes")[col].mean()
        print(f"    {col:30s} stay={stay:9.2f}  leave={leave:9.2f}")
    ot_leave = (clean_df[clean_df.overtime == "Yes"]["attrition"] == "Yes").mean()
    ot_no = (clean_df[clean_df.overtime == "No"]["attrition"] == "Yes").mean()
    print(f"    overtime=Yes attrition={ot_leave:.1%} vs overtime=No {ot_no:.1%}")

    # Check 3: do the NOISE columns stay quiet? (correlation with leaving ~ 0)
    print("\n[3] Noise-column correlation with leaving (should be ~0):")
    y = (clean_df["attrition"] == "Yes").astype(int)
    for col in ["survey_response_id"]:
        corr = np.corrcoef(clean_df[col].astype(float), y)[0, 1]
        print(f"    corr({col}, attrition) = {corr:+.4f}")
    print("=" * 60 + "\n")
