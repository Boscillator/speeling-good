# Speeling Good
Machine learning based spell checker research project.

Use `uv` for all python commands. You should already be inside the nix dev shell, but
if not you can use `nix develop` to activate the enviroment.

The name of the project, speeling-good, is deliberately misspelled. Other
mistakes are accidental and should be fixed without comment.

Use `ruff` for formatting and linting. Use `ty` for typechecking. Use `pytest`
for unit tests.

Unit tests should not depend on external files or detailed internal behaviour.
Never right a unit test that asserts trivial behaviour. If you need to test
something that depends on a file or external service, you can and should use
a repl. You may need to leave some features without tests in the repository.
This is preferable to brittle tests.

Use loguru for logging. Logs should be informative and helpful for debugging errors
and keeping track of progress without spamming the console during non-error conditions.

Use the readme for quick descriptions of the overall project and commmand references.
Most sessions should require zero changes to the README.

MODEL_ENDPOINT and MODEL_TOKEN enviroment variables include the information you need to
test against a live LLM server (either a local LM Studio with qwen/qwen3.8-27b or
OpenRouter depending on my current needs)

To run code that uses the GPU (ml training runs, etc.), you must request unsandboxed execution. 
This should be approved automatically, and otherwise I will approve it.

