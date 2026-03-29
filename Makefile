.PHONY: help build run test check clean

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@echo '  build       Build the Rust proxy binary'
	@echo '  run         Start the proxy server'
	@echo '  test        Run the test suite'
	@echo '  check       Run clippy + test'
	@echo '  clean       Remove build artifacts'

build:
	cargo build --release

run:
	cargo run --release

test:
	cargo test

check:
	cargo clippy -- -D warnings
	cargo test

clean:
	cargo clean
