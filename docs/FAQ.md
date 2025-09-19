# FAQ

## General Questions

#### What is Lucid?

Lucid (Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems) is a verification engine for certifying safety of black-box stochastic dynamical systems from a finite dataset of random state transitions.
Lucid employs a data-driven methodology rooted in control barrier certificates, which are learned directly from system transition data, to ensure formal safety guarantees.

#### What is Pylucid?

Pylucid is a Python wrapper for the Lucid library, providing an easy-to-use command-line and graphical interface to interact with the Lucid engine, as well as the ability to use it in any Python script.

## Troubleshooting

#### ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.35' not found

This error occurs if the `libc` version on your Linux system is too old.
Updating is challenging and error prone.
We recommend moving to a supported Linux distribution, such as

- Ubuntu 22.04 or later
- Debian 12 or later
- Fedora 36 or later
- RHEL 10 or later
- Arch Linux (rolling release)

#### ImportError: libgurobi120: cannot open shared object file: No such file or directory

This error occurs when the Gurobi library is not found in the expected location.
Ensure that you have installed Gurobi and that the `GUROBI_HOME` environment variable is set correctly.
Moreover, on Linux, you may need to set the `LD_LIBRARY_PATH` environment variable to include the path to the Gurobi library.

```bash
# For example, if Gurobi is installed in /opt/gurobi1201/linux64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/gurobi1201/linux64/lib"
```

Note that a Gurobi license is only mandatory if you plan to use the Gurobi solver.
Otherwise, the freely available installation of Gurobi, which you can download from [the official website](https://www.gurobi.com/downloads/) (a login may be required), is sufficient.

#### ImportError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory

This error occurs when the expected version of the Python shared library is not found in the expected location on the system
To fix this, you need to set the `LD_LIBRARY_PATH` environment variable to include the path to the Python library.
You can do this by running the following command:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/python/lib"
```

For example, if you are using Python 3.12 via `conda`

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(conda info --base)/envs/your_env/lib"
```

This change will only last for the current session.

#### LucidNotSupportedException: ...the following dependency was not included during compilation: 'X'

This error indicates that some optional dependencies were not included during the compilation of the bindings.
As a result, some features may not be available.
To resolve this, you must recompile the bindings from source, ensuring that the correct [flags are set](Installation.md#build-options) and the requirements are met.

For Gurobi support, see [Building with Gurobi](Installation.md#gurobi-requirements).

#### The process does not stop after pressing `Ctrl+C`

Pylucid is a thin wrapper around the C++ Lucid library, which means that when you press `Ctrl+C`, the Python interpreter sends a signal to the C++ executable, which may not handle it immediately.  
Long running operations may not notice the signal until they complete, which can cause the process to appear unresponsive.
In most cases the process will stop soon, (e.g., after an iteration of the optimiser has completed), but if a immediate termination is required, the safest way to stop the process is to close the terminal.

#### Address already in use. Port X is in use by another program

When running the GUI, you may encounter this error if the default port (3661) is already in use by another program.
To resolve this, you can either stop the program using that port or specify a different port when launching the GUI:

```bash
pylucid-gui --port 5001
```

Note that this will change the port the backend will listen on, but the frontend will still try to connect to the default port (3661).
To change the frontend port, you need to update the `vite.config.ts` file in the `frontend` directory, serve it with `pnpm dev`, and then access the GUI from the indicated url.

#### GUI is stuck

This can happen for a variety of reasons, usually related to the web server being unable to conclude a long operation in a timely manner.
The simplest solution is to restart the process.
To stop the server, you can press `Ctrl+C` in the terminal where the GUI was launched or simply kill the terminal session.

#### Bazel server stuck

Sometimes, especially on windows, bazel can get stuck compiling without a clear reason why.
If this happens, the bazel server may become unavailable and unable to run any other command.
To solve the situation, you can kill the bazel server and restart it.

```bash
# Get the bazel server PID.
# If it is stuck, it will return something like:
# Another command (pid=7760) is running. Waiting for it to complete on the server (server_pid=8032)...
bazel info server_pid
```

```bash
# Kill the bazel server (Windows)
taskkill /F /PID <server_pid>
# Kill the bazel server (Linux)
kill -9 <server_pid>
```

Any other command will restart the bazel server automatically.
