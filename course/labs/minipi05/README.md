# `minipi05` — Capstone skeleton

Assemble a minimal, clean-room π₀.₅ here from your Module 01–07 lab solutions. See
[`../../08_capstone_minimal_pi05.md`](../../08_capstone_minimal_pi05.md) for goals,
success criteria, and build order.

This directory ships only stubs + the file layout. **You** write the implementations —
that's the capstone. Suggested files:

```
config.py     MiniConfig (dims, horizons, pi05 flag)
masks.py      make_attn_mask           (from lab03)
transformer.py two-expert Block + scan (from lab03)
vit.py        mini-SigLIP              (from lab02)
tokenizer.py  prompt + discrete state  (from lab06)
model.py      MiniPi05: embed_prefix/embed_suffix/compute_loss/sample_actions
data.py       toy dataset + quantile norm (from lab01)
train.py      optimizer + train_step + EMA + loop (from lab07)
test_against_openpi.py   cross-checks vs the real openpi classes
```

Build incrementally and run `test_against_openpi.py` as you go.
