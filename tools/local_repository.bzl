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

UNSET = "_UNSET"

def local_repository_impl(rctx, env_var_name = "", default_pah = UNSET):
    """A rule to be called in the MODULE.bazel file to set up the Hexaly repository.

    It uses the 'env_var_name' environment variable to find the repository installation path.
    Alternatively, the user can specify the path using the --repo_env=<env_var_name>=<hexaly/installation/path> flag when building.
    If such a variable is not set, it fails with an error message.

    Args:
        rctx: The context object for the repository rule.
        env_var_name: The name of the environment variable to use to find the repository installation path.
        default_pah: The default installation path to use if the environment variable is not set.
    """
    local_repository_home = rctx.getenv(env_var_name) or default_pah
    if local_repository_home == UNSET:
        fail(("The environment variable %s is not set. " % env_var_name) +
             "Please set it to the path of your Hexaly installation," +
             ("add the flag --repo_env=%s=<hexaly/installation/path>" % env_var_name) +
             "or use the 'default_pah' param in the repository rule")

    if (rctx.attr.build_file == None) == (rctx.attr.build_file_content == UNSET):
        fail("exactly one of `build_file` and `build_file_content` must be specified")

    path = rctx.workspace_root.get_child(local_repository_home)
    if not path.is_dir:
        fail(
            ("The repository's path is \"%s\" (absolute: \"%s\") but it does not exist or is not " +
             "a directory.") % (local_repository_home, path),
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
