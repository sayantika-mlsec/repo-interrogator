"""The pinned repository set: one reader, in the package.

WHY THIS IS NOT IN A SCRIPT
---------------------------
This started as two functions inside the survey script. That was fine while the
survey was the only thing that read the pin file. It stopped being fine the
moment something inside the package needed the same data, because a package
cannot import from a script directory -- and the alternative was a second YAML
parser with its own idea of the schema.

Two readers of a pin file drift, and the field that drifts is the SHA. Every
number this project reports is meaningful only because the code under test could
not move, so the file that says which commit that is gets exactly one reader.

WHAT THIS MODULE DOES NOT DO
----------------------------
It reads the file and returns what the file says. It does not decide that an
unpinned entry is an error, that a held-out repository may not be touched, or
that a placeholder should be skipped. Those are policies, and different callers
hold different ones: the survey treats an unpinned entry as a hole in the set,
while the runner never reaches it because it resolves one repository by name.

Policy in a loader is policy that every caller inherits whether or not it
applies to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GROUPS = ("dev", "held_out", "memorization_check")
"""Every group in the file, in the order they are declared.

Read from a tuple rather than from the file's own keys so that a group added to
the YAML without being added here fails visibly -- as a repository that never
appears -- rather than silently entering a survey or a sweep.
"""

HELD_OUT = "held_out"
"""Named once. Callers enforcing the read discipline compare against this."""


@dataclass(frozen=True)
class RepoEntry:
    """One pinned repository, and which group it belongs to.

    Group membership is carried through rather than discarded. A held-out
    repository excluded by a bound is a different problem from a dev repository
    excluded by one: the first changes the number the project reports, the
    second only changes what is convenient to develop against.

    It is also what any read-discipline check has to consult, and a check that
    has to re-open the pin file to find out what it is guarding is a check that
    will eventually be given the wrong answer.
    """

    name: str
    url: str
    sha: str
    group: str
    domain: str | None

    @property
    def is_held_out(self) -> bool:
        return self.group == HELD_OUT

    @property
    def short_sha(self) -> str:
        return self.sha[:8]


@dataclass(frozen=True)
class RepoSet:
    """Every entry in the pin file, plus the date the set was frozen.

    ``pinned_on`` travels with the entries rather than being read separately.
    A results row naming a repository and a commit but not the date the set was
    frozen cannot be checked against the set that produced it.
    """

    entries: tuple[RepoEntry, ...]
    pinned_on: str | None

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def in_group(self, group: str) -> tuple[RepoEntry, ...]:
        return tuple(e for e in self.entries if e.group == group)

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entries)

    def find(self, name: str) -> RepoEntry:
        """Resolve one repository by name, or raise.

        Raises rather than returning ``None``. A caller that mistypes a name and
        receives nothing back has to remember to check; a caller that mistypes a
        name and gets an exception naming every valid option cannot proceed with
        no repository at all.
        """
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(
            f"no repository named {name!r} in the pin file. "
            f"Known: {', '.join(sorted(self.names()))}"
        )


def load_repos(path: Path) -> RepoSet:
    """Read the pin file into a flat set, preserving group membership."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    entries: list[RepoEntry] = []
    for group in GROUPS:
        for item in raw.get(group) or []:
            entries.append(
                RepoEntry(
                    name=item["name"],
                    url=item["url"],
                    sha=item["sha"],
                    group=group,
                    domain=item.get("domain"),
                )
            )

    # PyYAML parses an unquoted 2026-07-28 as a datetime.date. YAML has a date
    # type and JSON does not, so it is converted here, at the boundary where the
    # type is known, rather than by handing json.dumps a custom encoder that
    # would silently stringify anything unexpected arriving later.
    pinned_on = raw.get("pinned_on")

    return RepoSet(
        entries=tuple(entries),
        pinned_on=str(pinned_on) if pinned_on is not None else None,
    )


def load_entries(path: Path) -> tuple[list[RepoEntry], str | None]:
    """The survey script's original signature, kept so it did not have to change.

    Thin by design. New callers should use ``load_repos`` and hold the
    ``RepoSet``, which carries the pin date alongside the entries instead of
    beside them in a tuple the caller has to keep paired.
    """
    repo_set = load_repos(path)
    return list(repo_set.entries), repo_set.pinned_on