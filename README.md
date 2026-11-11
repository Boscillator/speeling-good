# speeling-good

A minimal Python and PyTorch project with CUDA support.

On x86-64 Linux with Nix, direnv, and a compatible NVIDIA driver installed:

```sh
git add flake.nix flake.lock
direnv allow
uv sync
nvidia-smi
uv run python main.py
```

Alternatively, enter the environment with `nix develop` instead of direnv. The
shell includes the CUDA toolkit and uses the host NVIDIA driver; on WSL it also
exposes `/usr/lib/wsl/lib`. PyTorch uses CUDA 12.8 wheels, which require a
compatible host driver. GPU access must be enabled in the host or WSL session.
The `nvidia-smi` wrapper expects the WSL driver utility at
`/usr/lib/wsl/lib/nvidia-smi`.

On NixOS, uv-installed executables such as Ruff and ty also require `nix-ld`.
Enable `programs.nix-ld.enable = true;` in the NixOS configuration and rebuild
the system, then reload the development shell. The shell sets `NIX_LD` and the
library path for these executables.

## LLM spelling annotations

`LlmAnnotationProvider` accepts a synchronous OpenAI client and can be shared by
multiple Python threads. Configure the client's API base URL, including `/v1` for
LM Studio. The caller owns and closes the client.

```python
import os

from openai import OpenAI

from speeling_good.preprocessing.llm_annotation_provider import LlmAnnotationProvider
from speeling_good.preprocessing.types import RawPassage

with OpenAI(
    base_url=os.environ["MODEL_ENDPOINT"].rstrip("/") + "/v1",
    api_key=os.environ["MODEL_TOKEN"],
) as client:
    provider = LlmAnnotationProvider(client=client, model="qwen/3.8-27b")
    result = provider.annotate(RawPassage("I recieved the package."))
    print(result.annotations)
```

This example assumes `MODEL_ENDPOINT` is the server root; omit the appended `/v1`
if your endpoint already includes it. Each annotation has the original text,
corrected spelling, and half-open Python string offsets into the unchanged passage.
Exact surrounding context must locate a unique span; ambiguous, missing,
duplicate, and overlapping corrections invalidate the entire response.

Defaults are four active requests per provider, three total attempts, and a
120-second timeout per request. Invalid output retries with validation feedback.
Connection errors, timeouts, HTTP 408/409/429, and 5xx responses retry with jittered
exponential backoff starting at one second and capped at eight seconds;
`Retry-After` takes precedence and is capped at 30 seconds. Other HTTP errors fail
immediately. SDK retries are disabled on a client copy. Exhausted invalid output
raises `AnnotationError`; exhausted transport failures preserve the SDK exception.
Backoff does not occupy a request slot. There is no caching or async interface.
