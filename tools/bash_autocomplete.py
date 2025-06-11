#!/usr/bin/env python3
import argparse
import sys
from typing import TypedDict

import shtab


class Completions(TypedDict):
    bash: str
    zsh: str


class Preamble:

    def __init__(self):
        self.extensions: dict[str, int] = {}
        self.files: list[Completions] = []
        self.preamble: Completions = {
            "bash": "",
            "zsh": "",
        }

    def file_by_extension(self, *extensions: "str"):
        extensions = tuple(ex.lower() for ex in extensions)
        if extensions in self.extensions:
            return self.files[self.extensions[extensions]]

        zsh_compgen = [f"*.{extension}|*.{extension.upper()}" for extension in extensions]
        zsh_compgen = "|".join(zsh_compgen)
        self.files.append(
            {
                "bash": f"_shtab_pydlinear_compgen_{'_'.join(extensions)}_files",
                "zsh": f"_files -g '({zsh_compgen})'",
            }
        )
        bash_compgen = [
            f"compgen -f -X '!*?.{extension}' -- $1\ncompgen -f -X '!*?.{extension.upper()}' -- $1"
            for extension in extensions
        ]
        bash_compgen = "\n".join(bash_compgen)
        self.preamble[
            "bash"
        ] += f"""
_shtab_pydlinear_compgen_{'_'.join(extensions)}_files() {{
  compgen -d -- $1  # recurse into subdirs
  {bash_compgen}
}}

"""
        self.extensions[extensions] = len(self.files) - 1
        return self.files[-1]

    @property
    def bash_preamble(self):
        return self.preamble["bash"]


def get_parser(prog: str, preamble: Preamble):
    parser = argparse.ArgumentParser(prog=prog)
    return parser


def complete_bash(prog: str, output: str):
    preamble = Preamble()
    autocomplete_script = shtab.complete_bash(get_parser(prog, preamble), preamble=preamble.bash_preamble)
    with open(output, "w", encoding="utf-8") as f:
        f.write(autocomplete_script)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <dlinear|dlinear> <out_file.sh>")
        exit(1)
