# Bindings

Lucid is a C++ library, but it also provides bindings for

- [Python](#python)

## Python

Lucid provides a thin wrapper around the C++ library, called **pylucid**, allowing you to use the main features of Lucid from Python.
There are multiple ways to use it.
You can either run it withing the Bazel environment, or you can install it in your Python environment.

### Running with Bazel

Write your script somewhere in the `lucid` directory.
In the same directory, create or update the file called `BUILD` with the following content:

```bzl

py_binary(
    name = "my_script",
    srcs = ["main.py"],
    main = "main.py",
    python_version = "PY3",
    deps = [
        "@lucid//bindings/pylucid:pylucid_lib",
    ],
)
```

Then, run your script using the following command:

```bash
bazel run //path/to/your/script:your_script
```

### Installing in your Python environment

You can install the bindings in your Python environment using the following command:

```bash
pip install .
```

This will install the bindings in your Python environment, allowing you to use them directly from Python.
After installing, you can run the following command to check if everything is working correctly:

```bash
python -c "import pylucid; print(pylucid.__version__)"
```

### Troubleshooting

<details>
<summary>

#### ImportError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory

</summary>

This error occurs when the expected Python shared library is not found in the expected location on the system
To fix this, you need to set the `LD_LIBRARY_PATH` environment variable to include the path to the Python library.
You can do this by running the following command:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/python/lib
```

For example, if you are using Python 3.12 via `conda`

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(conda info --base)/envs/your_env/lib
```

This change will only last for the current session.

</details>
