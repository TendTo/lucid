# Testing

Lucid includes unit tests and integration tests.
All tests use the [Google Test framework](https://github.com/google/googletest) and are run using [Bazel](https://bazel.build/).

## Running the tests

To run all the tests, use the following command:

```bash
bazel test //tests/...
```

To only run a subset of the available tests, there are multiple options:

```bash
# Run all the unit tests
bazel test //tests/unit/...
# Run all the integration tests
bazel test //tests/integration/...
# Run a specific test (the logging test in this case)
bazel test //tests/unit/util:test_logging
```

> [!NOTE]  
> Integration tests and bindings tests use Gurobi, and therefore need a valid Gurobi license to run.
> Just copy the licence `gurobi.lic` in the `tests/integration` and `tests/bindings` directories respectively.
