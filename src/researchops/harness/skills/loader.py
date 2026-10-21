"""Read metadata first; load bodies only after selection from a trusted root.

This implementation uses this project's binding, policy and artifact contracts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from researchops.core.artifacts import canonical_hash, file_hash


class SkillLoader:
    def __init__(self, root: Path, bindings: dict[str, str]):
        self.root, self.bindings = root.resolve(), dict(bindings)

    def _paths(self, name):
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name not in self.bindings:
            raise ValueError(f"Unknown host-bound skill: {name}")
        folder = (self.root / name).resolve()
        if folder.parent != self.root or folder.is_symlink():
            raise PermissionError("Skill directory escapes the trusted root")
        paths = [folder / "SKILL.md", folder / "agents" / "researchops.json"]
        for path in paths:
            if path.is_symlink() or self.root not in path.resolve().parents or not path.is_file():
                raise PermissionError("Skill resource is missing or outside the trusted root")
            if path.stat().st_size > 100000:
                raise ValueError("Skill resource size limit exceeded")
        return paths

    def metadata(self, name):
        doc, manifest = self._paths(name)
        lines, count = [], 0
        with doc.open(encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise ValueError("Skill frontmatter required")
            for line in handle:
                count += len(line.encode("utf-8"))
                if count > 4096:
                    raise ValueError("Skill metadata limit exceeded")
                if line.strip() == "---":
                    break
                lines.append(line)
            else:
                raise ValueError("Unclosed skill frontmatter")
        meta = yaml.safe_load("".join(lines))
        if not isinstance(meta, dict) or set(meta) != {"name", "description"} or meta["name"] != name:
            raise ValueError("Skill frontmatter must contain matching name and description")
        if not isinstance(meta["description"], str) or not 1 <= len(meta["description"]) <= 1024:
            raise ValueError("Invalid skill description")
        config = json.loads(manifest.read_text(encoding="utf-8"))
        if config.get("tool") != self.bindings[name] or not isinstance(config.get("version"), str):
            raise ValueError("Skill manifest cannot change the host tool binding")
        return {**meta, "version": config["version"], "tool": config["tool"]}

    def catalog(self):
        # Only explicitly host-bound project skills, never downloaded papers/READMEs.
        return [self.metadata(name) for name in sorted(self.bindings)]

    def load(self, name):
        meta = self.metadata(name)
        doc, manifest = self._paths(name)
        text = doc.read_text(encoding="utf-8")
        match = re.match(r"\A---\s*\n.*?\n---\s*\n(.*)\Z", text, re.S)
        if not match or not match[1].strip():
            raise ValueError("Skill instructions are empty")
        return {**meta, "body": match[1].strip(), "skill_path": str(doc),
                "sha256": canonical_hash([file_hash(doc), file_hash(manifest)])}


def search_tokens(value):
    latin = re.findall(r"[a-z0-9]+", value.casefold())
    cjk = re.findall(r"[\u3400-\u9fff]+", value)
    return set(latin + [s[i:i + 2] for s in cjk for i in range(len(s) - 1)])
