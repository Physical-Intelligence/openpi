# Simple Modal Client

A client running inference remotely on an H100 container on Modal.

### Recommended for production: deploy a Modal app for inference

Deploy the app. You only need to run this once - once deployed, inference can be
invoked from any context. Read more on the pattern
[here](https://modal.com/docs/guide/trigger-deployed-functions#function-lookup-and-invocation-basics).

```bash
uv run --with quic-portal[modal]==0.1.6 modal deploy examples/simple_modal_client/modal_policy.py
```

Then, run inference.

```bash
uv run --with quic-portal[modal]==0.1.6 examples/simple_modal_client/main.py
```

### Development mode: use an ephemeral Modal app

You can specify which runtime environment to use using the `--env` flag. You can
see the available options by running:

```bash
uv run examples/simple_modal_client/main.py --ephemeral
```
