### Building

- Most important Makefile targets: `make clean`, `make -j 32 [all]`, `make src_lib/FILE.o`, and `make src_pybind11/FILE.o`.
- CRITICAL: Always build multiple targets with 32 threads: `make -j 32`.

### Style/coding Guidelines

- Prefer private (or static) functions over anonymous scopes `{ ... }`.
- Use `//` comments, not `/* */`
- Use `long` for sizes and indices, not `int` or `size_t`
- Use spaces, not tabs.
- All functions should error-check function arguments thoroughly and throw an exception if anything is wrong, unless the function is in a "hot loop". 
- In contexts where it's not important to customize exception text, use the exception-throwing macros `xassert()`, `xassert_eq()`, `xassert_le()`, `xassert_divisible()` from `include/ksgpu/xassert.hpp`.
- Please write comments to explain big-picture functionality or non-obvious issues, but avoid superficial comments. 
- Please ask me questions in the chat if my instructions are incomplete or unclear.