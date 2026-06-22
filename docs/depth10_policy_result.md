# Brogue Depth-10 Policy Search Result

This scripted-policy configuration reaches depth 10 on seed 5996:

```sh
uv run python scripts/search_brogue_depth_policy.py \
  --start-seed 5996 \
  --episodes 1 \
  --parallel-envs 1 \
  --max-steps 6000 \
  --max-no-change-steps 768 \
  --target-depth 10 \
  --progress-interval 0 \
  --panic-items \
  --panic-min-depth 6 \
  --panic-scrolls \
  --panic-scrolls-min-depth 8 \
  --panic-zaps \
  --panic-zaps-min-depth 8 \
  --panic-near-radius 8 \
  --avoid-monsters-min-depth 6 \
  --adjacent-attack-min-hp-frac 0.9 \
  --verbose \
  --trace-dir /tmp/brogue-depth-search/verify-current-seed5996 \
  --stop-on-success
```

Verified result:

```text
seed=5996 max_depth=10 final_depth=10 score=886 gold=886 turn=1020 steps=407 terminated=False
```

The exact 407-action trace is saved in `docs/seed-5996-depth-10.json`.
