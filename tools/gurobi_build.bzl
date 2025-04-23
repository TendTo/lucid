# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# WARNING:
# https://github.com/bazelbuild/bazel/issues/17713
# .bzl files in this package (tools/build_defs/repo) are evaluated
# in a Starlark environment without "@_builtins" injection, and must not refer
# to symbols associated with build/workspace .bzl files

_UNSET = "_UNSET"

def _gurobi_repository_impl(rctx):
    """A rule to be called in the MODULE.bazel file to set up the Gurobi repository.

    It uses the GUROBI_HOME environment variable to find the Gurobi installation path.
    Alternatively, the user can specify the path using the --repo_env=GUROBI_HOME=<gurobi/installation/path> flag when building.
    If such a variable is not set, it fails with an error message.

    Args:
        rctx: The context object for the repository rule.
    """
    gurobi_home = rctx.getenv("GUROBI_HOME") or rctx.attr.default_gurobi_home
    if gurobi_home == _UNSET:
        fail("The environment variable GUROBI_HOME is not set. " +
             "Please set it to the path of your Gurobi installation," +
             "add the flag --repo_env=GUROBI_HOME=<gurobi/installation/path>" +
             "or use the 'default_gurobi_home' param in the repository rule")

    if (rctx.attr.build_file == None) == (rctx.attr.build_file_content == _UNSET):
        fail("exactly one of `build_file` and `build_file_content` must be specified")

    path = rctx.workspace_root.get_child(gurobi_home)
    if not path.is_dir:
        fail(
            ("The repository's path is \"%s\" (absolute: \"%s\") but it does not exist or is not " +
             "a directory.") % (gurobi_home, path),
        )

    children = path.readdir()
    for child in children:
        rctx.symlink(child, child.basename)

        # On Windows, `rctx.symlink` actually does a copy for files (for directories, it uses
        # junctions which basically behave like symlinks as far as we're concerned). So we need to
        # watch the symlink target as well.
        if rctx.os.name.startswith("windows") and not child.is_dir:
            rctx.watch(child)

    if rctx.attr.build_file != None:
        # Remove any existing BUILD.bazel in the repository to ensure
        # the symlink to the defined build_file doesn't fail.
        rctx.delete("BUILD.bazel")
        rctx.symlink(rctx.attr.build_file, "BUILD.bazel")
        if rctx.os.name.startswith("windows"):
            rctx.watch(rctx.attr.build_file)  # same reason as above
    else:
        rctx.file("BUILD.bazel", rctx.attr.build_file_content)

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
