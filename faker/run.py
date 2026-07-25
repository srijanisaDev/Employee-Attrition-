"""
run.py  --  entry point. Runs every stage in order and writes both CSVs.

    python run.py        (run from inside the faker/ folder)

Outputs
-------
    data/Employee_data_clean.csv   the pristine "answer key" (clean truth)
    data/Employee_data.csv         the dirty, delivered file for the pipeline
"""

import os
import numpy as np
from faker import Faker

import config
import generate


def main():
    # one shared random generator + seed = fully reproducible dataset
    rng = np.random.default_rng(config.RANDOM_SEED)
    faker = Faker()
    Faker.seed(config.RANDOM_SEED)

    os.makedirs(config.DATA_DIR, exist_ok=True)

    # STAGE 1: clean, plausible features
    print("Stage 1: generating clean features...")
    clean = generate.build_clean_features(rng, faker)

    # STAGE 2: probabilistic label (normalize -> weight -> sigmoid -> sample)
    print("Stage 2: computing probabilistic labels...")
    clean = generate.add_probabilistic_label(clean, rng)

    # STAGE 3: save the clean truth (answer key) BEFORE dirtying anything
    print(f"Stage 3: saving clean truth -> {config.CLEAN_CSV}")
    clean.to_csv(config.CLEAN_CSV, index=False)

    # STAGE 4: inject messiness onto a COPY (features only, never the label)
    print("Stage 4: injecting messiness...")
    dirty = generate.inject_messiness(clean, rng)

    # STAGE 5: save the delivered dirty file
    print(f"Stage 5: saving dirty file -> {config.DIRTY_CSV}")
    dirty.to_csv(config.DIRTY_CSV, index=False)

    # STAGE 6: sanity checks (run on the CLEAN data, where the label is trustworthy)
    generate.sanity_checks(clean)

    print(f"Done. {len(clean):,} employees generated "
          f"({len(dirty) - len(clean)} duplicate rows added to dirty file).")


if __name__ == "__main__":
    main()
