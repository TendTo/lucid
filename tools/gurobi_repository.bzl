load("//tools:local_repository.bzl", _UNSET = "UNSET", _local_repository_impl = "local_repository_impl")

def _gurobi_repository_impl(rctx):
    """A rule to be called in the MODULE.bazel file to set up the Gurobi repository.

    It uses the GUROBI_HOME environment variable to find the Gurobi installation path.
    Alternatively, the user can specify the path using the --repo_env=GUROBI_HOME=<gurobi/installation/path> flag when building.
    If such a variable is not set, it fails with an error message.

    Args:
        rctx: The context object for the repository rule.
    """
    _local_repository_impl(rctx, env_var_name = "GUROBI_HOME", default_pah = rctx.attr.default_gurobi_home)

gurobi_repository = repository_rule(
    implementation = _gurobi_repository_impl,
    local = True,
    attrs = {
        "build_file": attr.label(
            doc = "A file to use as a BUILD file for this repo.\n\nExactly one of `build_file` and " +
                  "`build_file_content` must be specified.\n\nThe file addressed by this label " +
                  "does not need to be named BUILD, but can be. Something like " +
                  "`BUILD.new-repo-name` may work well to distinguish it from actual BUILD files.",
        ),
        "build_file_content": attr.string(
            doc = "The content of the BUILD file to be created for this repo.\n\nExactly one of `build_file` and `build_file_content` must be specified.",
            default = _UNSET,
        ),
        "default_gurobi_home": attr.string(
            doc = "The default Gurobi installation path.\n\nThis is used if the GUROBI_HOME environment variable is not set.",
            default = _UNSET,
        ),
    },
)
