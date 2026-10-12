"""
Module 1 Master Runner: Digital Phenotyping Atlas
Runs all steps in sequence.
"""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("module1_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    t0 = time.time()

    logger.info("=" * 80)
    logger.info("MODULE 1: DIGITAL PHENOTYPING ATLAS - FULL PIPELINE")
    logger.info("=" * 80)


    # Step 01: Feature Engineering
    logger.info("\n>>> STEP 01: Feature Engineering <<<")
    from s01_feature_extraction import run_feature_extraction
    run_feature_extraction()

    # Step 02: Feature Engineering
    logger.info("\n>>> STEP 02: Feature Aggregation <<<")
    from s02_feature_aggregation import run_aggregation
    run_aggregation()
    
    # Step 03: Effect Size Computation
    logger.info("\n>>> STEP 03: Effect Sizes <<<")
    from s03_effect_sizes import run_effect_sizes
    run_effect_sizes()

    # Step 04: Discrimination Analysis
    logger.info("\n>>> STEP 04: Discrimination Analysis <<<")
    from s04_discrimination import run_discrimination
    run_discrimination()

    # Step 05: Factor Analysis
    logger.info("\n>>> STEP 05: Factor Analysis <<<")
    from s05_factor_analysis import run_factor_analysis
    run_factor_analysis()

    # Step 06: Sensitivity Analysis
    logger.info("\n>>> STEP 06: Sensitivity Analysis <<<")
    from s06_sensitivity import run_sensitivity
    run_sensitivity()


    # Step 07: Visualization
    logger.info("\n>>> STEP 07: Visualization <<<")
    from s07_visualization import run_visualization
    run_visualization()
    
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 80}")
    logger.info(f"MODULE 1 COMPLETE. Total time: {elapsed/60:.1f} minutes")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
