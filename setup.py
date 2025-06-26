"""Install script for setuptools."""

import glob
import os
import re
import shutil
import subprocess
import sysconfig

import setuptools
import setuptools.errors
from setuptools.command import build_ext


def get_bazel_target_args(command):
    if command == "build":
        return [
            "bazel",
            "build",
            "--config=py",
            f"--python_version={sysconfig.get_python_version()}",
            f"--enable_gurobi_build={'True' if 'GUROBI_HOME' in os.environ else 'False'}",
        ]
    if command == "cquery":
        return [
            "bazel",
            "cquery",
            "--output=files",
            "--config=py",
            f"--python_version={sysconfig.get_python_version()}",
            f"--enable_gurobi_build={'True' if 'GUROBI_HOME' in os.environ else 'False'}",
        ]
    if command == "query":
        return [
            "bazel",
            "query",
            "--output=build",
        ]


class GlobalVariables:
    def __init__(self):
        self.LUCID_VERSION = self._get_value_from_query("lucid_version")
        self.LUCID_DESCRIPTION = self._get_value_from_query("lucid_description")

    def _get_value_from_query(self, target):
        bazel_argv = [*get_bazel_target_args("query"), f"//tools:{target}"]
        matches: "list[str]" = re.findall(r"value = +(.+),", subprocess.getoutput(" ".join(bazel_argv)))
        if len(matches) == 0:
            raise ValueError(f"Could not parse {target} from Bazel query output.")
        return matches[0].rstrip('"').lstrip('"')


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, ext_name, bazel_target):
        self.bazel_target = bazel_target
        folders = ("lucid", "bindings/pylucid", "tools", "third_party")
        files = []
        for folder in folders:
            files += glob.glob(f"{folder}/**/*", recursive=True)
        files += ["BUILD.bazel", "MODULE.bazel", ".bazelversion", ".bazelrc", ".bazelignore"]
        setuptools.Extension.__init__(self, ext_name, sources=files)


class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)
        # Run the Bazel shutdown command to clean up
        self.spawn(["bazel", "shutdown"])

    def bazel_build(self, ext: BazelExtension):
        if shutil.which("bazel") is None:
            raise setuptools.errors.CompileError(
                "Bazel not found (https://bazel.build/). It is required to install this package from source."
            )

        # Build all needed files
        bazel_argv = [*get_bazel_target_args("build"), "//bindings/pylucid:pylucid_files"]
        self.spawn(bazel_argv)
        paths = (
            subprocess.check_output([*get_bazel_target_args("cquery"), "//bindings/pylucid:pylucid_files"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        # Copy the built files to the extension directory
        for path in paths:
            file = re.sub(r"^.*bindings/pylucid/", "", path)
            ext_dest_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
            os.makedirs(os.path.join(ext_dest_dir, os.path.dirname(file)), exist_ok=True)
            shutil.copyfile(path, os.path.join(ext_dest_dir, file))


config_vars = GlobalVariables()
setuptools.setup(
    version=config_vars.LUCID_VERSION,
    description=config_vars.LUCID_DESCRIPTION,
    ext_modules=[BazelExtension("pylucid._pylucid", "//bindings/pylucid:_pylucid")],
    cmdclass={"build_ext": BuildBazelExtension},
)
