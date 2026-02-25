# Unit testing
mod test 'tests/justfile'

# List all commands
default:
    just --list

# Reformat and lint
format:
    uvx ruff@latest format .
    uvx ruff@latest check . --fix
    uv run cargo fmt
    uv run cargo clippy --fix --allow-dirty --all-targets --all-features

# Run type checking
check *ARGS:
    uvx ty@latest check --output-format concise {{ARGS}}
    uv run cargo check
