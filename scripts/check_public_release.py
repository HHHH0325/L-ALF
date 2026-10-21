from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    category: str
    path: Path
    line: int = 0
    detail: str = ""


_SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
_FORBIDDEN_DIRS = {
    "data", "runs", "results", "raw_csv", "api_cache", "feature_cache",
    "test", "tests", "fixtures", "examples", "test-results", "coverage",
}
_FORBIDDEN_SUFFIXES = {
    ".gbw", ".h5", ".hdf5", ".zip", ".csv", ".tsv", ".out", ".xyz", ".log",
}
_TEST_NAME_PATTERN = re.compile(r"^(?:test_.*\.py|.*_test\.py|conftest\.py)$", re.I)
_TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".tpl",
    ".txt",
    ".yaml",
    ".yml",
}
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|access[_-]?token)\b"
    r"\s*[:=]\s*['\"][^'\"]{4,}['\"]"
)
_PRIVATE_KEY_PATTERN = re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")
_PERSONAL_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:\\Users\\[^\\\s]+|/(?:home|Users)/[^/\s]+/)"
)
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _known_private_terms(root: Path) -> list[str]:
    denylist = root / ".private-denylist"
    if not denylist.exists():
        return []
    return [
        line.strip()
        for line in denylist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def scan_repository(root: str | Path) -> list[Finding]:
    repository = Path(root).resolve()
    findings: list[Finding] = []
    private_terms = _known_private_terms(repository)

    for path in sorted(repository.rglob("*")):
        relative = path.relative_to(repository)
        if any(part in _SKIP_DIRS or part.startswith(".pytest-") for part in relative.parts):
            continue
        forbidden_parts = [
            i for i, part in enumerate(relative.parts)
            if part.lower() in _FORBIDDEN_DIRS
        ]
        if forbidden_parts:
            if path.is_dir() and forbidden_parts[0] == len(relative.parts) - 1:
                findings.append(Finding("forbidden_directory", relative))
            continue
        if not path.is_file() or path.is_symlink():
            continue
        suffix = path.suffix.lower()
        if suffix in _FORBIDDEN_SUFFIXES or _TEST_NAME_PATTERN.fullmatch(path.name):
            findings.append(Finding("forbidden_file", relative))
            continue
        if path.stat().st_size > 2_000_000:
            findings.append(Finding("large_file", relative, detail=str(path.stat().st_size)))
        if suffix not in _TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if relative.as_posix() == "scripts/check_public_release.py" and line_number <= 60:
                continue
            if _CREDENTIAL_PATTERN.search(line):
                findings.append(Finding("credential", relative, line_number))
            if _PRIVATE_KEY_PATTERN.search(line):
                findings.append(Finding("private_key", relative, line_number))
            if _PERSONAL_PATH_PATTERN.search(line):
                findings.append(Finding("personal_path", relative, line_number))
            if _EMAIL_PATTERN.search(line):
                findings.append(Finding("identity", relative, line_number))
            for term in private_terms:
                if term in line:
                    findings.append(Finding("private_term", relative, line_number, term))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan a L-ALF release tree for leaks")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    findings = scan_repository(args.root)
    if findings:
        for finding in findings:
            location = f"{finding.path}:{finding.line}" if finding.line else str(finding.path)
            print(f"{finding.category}: {location} {finding.detail}".rstrip())
        return 1
    print("Public-release scan passed: no findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
