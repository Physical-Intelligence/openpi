# Simple Client

A minimal client that sends observations to a policy server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
uv run examples/simple_client/main.py --help
```

## CPU-only dry run

If you want to test the websocket path before downloading a checkpoint or connecting a robot, start the fake policy server in one terminal:

```bash
uv run examples/simple_client/fake_policy_server.py --action-horizon 10 --action-dim 8
```

Then send random observations from another terminal:

```bash
uv run examples/simple_client/main.py --env DROID --num-steps 3 --show-action-chunk
```

This verifies that the client can connect, send an observation dictionary, receive an `actions` array, and print the action chunk shape. The fake server does not load model weights.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
uv run examples/simple_client/main.py --env DROID
```

Terminal window 2:

```bash
uv run scripts/serve_policy.py --env DROID
```

For a trained or fine-tuned checkpoint, start the real policy server with the training config and checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=/path/to/checkpoint
```

Then keep the same client command, choosing the `--env` whose random observation fields match the policy input mapping:

```bash
uv run examples/simple_client/main.py --env DROID --num-steps 10 --show-action-chunk
```
