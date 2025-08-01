# Integration benchmarks

This directory contains integration benchmarks for LUCID, which are designed to test the functionality and performance of the LUCID framework in various scenarios.
The benchmarks are structured to evaluate different aspects of the system, including problem solving, optimization, and data handling.

> [!NOTE]  
> We use [MLflow 3.x](https://mlflow.org/) to track the results of the benchmarks.
> Only Python 3.9 and later versions are supported.

To run the benchmarks, make sure `pylucid` and `mlflow` are installed in your Python environment.
Then, execute the desired script.

```bash
python benchmarks/integration/Linear.py
```

To view the results, you can use the MLflow UI. Start the MLflow server with:

```bash
mlflow server
```

Then, navigate to [`http://localhost:5000`](http://localhost:5000) in your web browser to see the tracked runs and their results.
