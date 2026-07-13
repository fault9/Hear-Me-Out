# Hear-Me-Out fork changes

This fork tracks PersonaPlex / Hear-Me-Out integration work that supports the X-VC
accent-conversion experiments.

## 2026-07-07: X-VC serving and setup notes

### Current branch

Local fork branch:

```text
pp-vc-quality
```

Current local changes touch:

- `infra/run_all.sh`
- `infra/setup.sh`

### X-VC checkpoint picker in `infra/run_all.sh`

When `VC_ENGINE=xvc`, `run_all.sh` now resolves which X-VC checkpoint should be served
on port `5002`.

Behaviour:

- `XVC_CKPT=/path/to/checkpoint.pt` skips the menu and serves that checkpoint;
- otherwise, any `*.pt` under `$XVC_DIR/ckpts/` is offered interactively;
- `xvc.pt` is labelled as the original released model;
- fine-tuned checkpoints default `XVC_EMA_LOAD=0`, because the fine-tune configs save
  raw generator weights rather than EMA weights.

This makes it possible to compare:

- stock X-VC (`ckpts/xvc.pt`);
- accent fine-tunes pulled into `$XVC_DIR/ckpts/`;
- future condition-specific checkpoints without editing the launcher.

### X-VC fork/ref override in `infra/setup.sh`

`infra/setup.sh --xvc` no longer hard-codes only upstream X-VC. The X-VC source can be
overridden with environment variables:

```bash
XVC_URL=https://github.com/fault9/X-VC-accent-finetuning.git \
XVC_REF=latent-alignment \
bash infra/setup.sh --xvc
```

Default behaviour remains upstream-compatible:

```text
XVC_URL=https://github.com/Jerrister/X-VC.git
XVC_REF=49df8c591eafc48b096e466d96f9839f9c0dd739
```

### Operational caution

For training, keep X-VC separate from Hear-Me-Out setup:

```text
~/X-VC          -> fine-tuning branch / training runs
~/Hear-Me-Out  -> PersonaPlex/HMO app and serving scripts
```

Do not run `infra/setup.sh --xvc` blindly when the goal is to fine-tune, because the
setup script is for serving/installing an X-VC engine inside the HMO workspace. For the
current experiment, restore/train the X-VC fork directly and only use HMO once a
checkpoint should be served.

### Intended evaluation direction

The live study direction is native participant speech -> assigned target accent.
Therefore HMO/X-VC serving and evaluation should focus on:

- unseen native-English source speech;
- assigned target accent reference voice;
- stock X-VC vs fine-tuned X-VC;
- original native audio as a no-conversion control;
- real accented recordings only as positive-reference controls.
