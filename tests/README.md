# Tests

Before running any tests, make sure you have `uv` installed (and ideally run `make sync` after).

## Running tests

```
make tests
```

`make tests` runs the shard-safe suite in parallel and then runs tests marked `serial`
in a separate serial pass.

Optional live CLI runtime checks are disabled by default. To run them in an environment
where `codex`, `gemini`, or `copilot` are installed and already authenticated:

```bash
AGENTS_RUN_CLI_LIVE_TESTS=1 uv run pytest tests/models/test_cli_live.py -v

ACP live coverage exercises both `gemini` and `copilot`. Some local Gemini CLI environments never
complete the initial ACP handshake; the Gemini ACP live test will skip in that case instead of
hanging indefinitely.
```

## Snapshots

We use [inline-snapshots](https://15r10nk.github.io/inline-snapshot/latest/) for some tests. If your code adds new snapshot tests or breaks existing ones, you can fix/create them. After fixing/creating snapshots, run `make tests` again to verify the tests pass.

### Fixing snapshots

```
make snapshots-fix
```

### Creating snapshots

```
make snapshots-create
```
