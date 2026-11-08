# A1 and A2 validation

The packaged local training workflow and recovery of the original adapter have been verified. Strategic learning remains unverified.

| Milestone | Verified outcome | Remaining work |
| --- | --- | --- |
| A1 | Full packaged GPU run; 10 environment tests; CPU/GPU LoRA preflight; two FSDP cycles; model/optimizer saving; revision and patch capture | Clean installation remains a separate reproducibility task |
| A2 | Recovery of 336 finite tensors; exact saved-tensor equality; PEFT reload; zero logit difference on two fixed prompts | Automatic RAGEN export, upstream base revision, and independent Colab loading |

## Training rerun

Run train-one-QOwTZfVo completed one fresh iteration on November 8, 2026. RAGEN revision was 9d021fc782a4a9e8e047f13c1cd706b32f16d8f2 with local compatibility patches captured in the evidence directory. Dependency conflicts remain; this is not a clean-install lock.

All four episodes lost. Mean terminal return was -1, advantages and policy-gradient loss were zero, and the iteration took approximately 414.7 seconds. Actor gradient norm was approximately 0.009006. This verifies the execution path, not strategic improvement.

The two completed runs are independent one-iteration runs. Do not plot them as a continuous two-iteration learning curve.

## Adapter recovery

A2 used the earlier train-one-F5JCCw3d checkpoint. Its empty standalone export was bypassed by extracting LoRA tensors from the full checkpoint. The recovered adapter contains 336 tensors, including 168 nonzero B tensors; the safetensors file is 17,640,136 bytes.

Every tensor matched after saving and reloading. PEFT loading succeeded. Using the checkpoint's frozen base weights and CPU float32 SDPA, both fixed maze prompts had maximum next-token logit differences of 0.0.

The generation smoke test returned:

> Please provide the coordinates for A and B so I can assist you further.

Generation executes, but this response does not follow the requested action format. The fixtures are not the complete RAGEN evaluation protocol. No game success or policy improvement is claimed.

The base model is Qwen/Qwen2.5-0.5B-Instruct; its exact upstream revision is unknown. A checkpoint-based reload does not verify that downloading the current upstream model recreates identical frozen weights.

## Reproduction and next steps

See [the local workflow](local-training.md). The exporter has seven guard tests covering key normalization and source isolation, missing tensors, NaNs, incorrect shapes, multiple ranks, nondefault adapter names, and unsupported DoRA.

Compact evidence is in [a1-a2-2026-11-08.json](a1-a2-2026-11-08.json). Original logs are under results/a1-a2-ZgMHXXWL/. Raw results and weights remain excluded from Git.

Next is A3: diagnose invalid actions and obtain informative within-group reward variation. Continuous training curves and before/after game evaluation remain pending.
