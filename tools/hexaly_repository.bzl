load("//tools:local_repository.bzl", _UNSET = "UNSET", _local_repository_impl = "local_repository_impl")

def _hexaly_repository_impl(rctx):
    """A rule to be called in the MODULE.bazel file to set up the Hexaly repository.

    It uses the HEXALY_HOME environment variable to find the Hexaly installation path.
    Alternatively, the user can specify the path using the --repo_env=HEXALY_HOME=<hexaly/installation/path> flag when building.
    If such a variable is not set, it fails with an error message.

    Args:
        rctx: The context object for the repository rule.
    """
    _local_repository_impl(rctx, env_var_name = "HEXALY_HOME", default_pah = rctx.attr.default_hexaly_home)

hexaly_repository = repository_rule(
    implementation = _hexaly_repository_impl,
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
        "default_hexaly_home": attr.string(
            doc = "The default Hexaly installation path.\n\nThis is used if the HEXALY_HOME environment variable is not set.",
            default = _UNSET,
        ),
    },
)
