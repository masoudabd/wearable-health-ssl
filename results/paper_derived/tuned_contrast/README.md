# Post hoc contrast: tuned foundation-model probe against the clinical features' default probe

Command (2026-10-12, paper session, HEAD cafc52d, CPU):

    .venv/bin/python scripts/paired_tests.py results/rq12_tuned_ft_seed0 results/rq12_tuned_ft_seed1 \
      results/rq12_tuned_ft_seed2 results/rq12_tuned_ft_seed3 results/rq12_tuned_ft_seed4 \
      --out results/paper_derived/tuned_contrast --name paired_tests --comparisons harnet5_tc:feat60_fi

Why: in the equal-effort runs, tuning the probe's C on validation patients raised the foundation model's AUROC but
lowered the clinical features' (feat60_fi_tc minus feat60_fi below zero at most budgets). This contrast asks whether
the tuned foundation model's lead over the tuned features remains against the features' default (C = 1) probe.
Same patient resamples as every other paired test. Holm over its own 8 tests; the paper reports it as post hoc.
