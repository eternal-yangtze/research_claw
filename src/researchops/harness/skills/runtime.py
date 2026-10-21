from __future__ import annotations

import uuid

from researchops.core.artifacts import write_json
from researchops.harness.runtime import ToolCall, ToolSpec, ToolRisk
from .loader import SkillLoader, search_tokens

SKILL_CONTROLS = ("search_skills", "load_skill", "invoke_skill")


class SkillRuntime:
    def __init__(self, root, bindings, registry, traces, artifact_root):
        self.loader = SkillLoader(root, bindings)
        self.registry, self.traces, self.artifact_root = registry, traces, artifact_root

    def visible(self, context):
        return [item for item in self.loader.catalog()
                if context.executor.policy.authorize(item["tool"], context.agent)[0]]

    def catalog(self, agent, policy, query="", limit=8):
        entries = [item for item in self.loader.catalog() if policy.authorize(item["tool"], agent)[0]]
        terms = search_tokens(query)
        entries.sort(key=lambda m: (-len(terms & search_tokens(m["name"] + " " + m["description"])), m["name"]))
        return {"total_available": len(entries), "skills": entries[:limit], "more_available": len(entries) > limit}

    def register(self, agent_name):
        schemas = {
            "search_skills": ({"query": {"type": "string"}}, ["query"], self.search,
                              "Find applicable skills by goal; returns short metadata, never full instructions."),
            "load_skill": ({"name": {"type": "string"}}, ["name"], self.load,
                           "Read the selected skill instructions and exact input schema before invoking it."),
            "invoke_skill": ({"name": {"type": "string"}, "sha256": {"type": "string"}, "arguments": {"type": "object"}},
                             ["name", "sha256", "arguments"], self.invoke,
                             "Execute a skill loaded in this session using its current hash and validated arguments; host policy and gates remain enforced."),
        }
        for name, (properties, required, handler, description) in schemas.items():
            self.registry.register(ToolSpec(name, description,
                {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
                handler, contextual=True, allowed_agents=frozenset({agent_name}),
                risk=ToolRisk.NORMAL if name == "invoke_skill" else ToolRisk.LOW,
                read_only=name != "invoke_skill", side_effect=name == "invoke_skill", max_output_chars=16000))

    def search(self, query, *, _context):
        result = self.catalog(_context.agent, _context.executor.policy, query, limit=5)
        self.traces.emit(_context.run_id, "skill.discovered", _context.session.session_id,
                         {"query": query, "names": [m["name"] for m in result["skills"]]})
        return result

    def load(self, name, *, _context):
        skill = self.loader.load(name)
        allowed, reason = _context.executor.policy.authorize(skill["tool"], _context.agent)
        if not allowed:
            raise PermissionError(reason)
        _context.session.metadata.setdefault("loaded_skills", {})[name] = skill["sha256"]
        self.traces.emit(_context.run_id, "skill.loaded", _context.session.session_id,
                         {"name": name, "version": skill["version"], "sha256": skill["sha256"], "body_chars": len(skill["body"])})
        spec = self.registry.get(skill["tool"])
        return {**skill, "input_schema": spec.input_schema}

    def invoke(self, name, sha256, arguments, *, _context):
        skill = self.loader.load(name)
        loaded = _context.session.metadata.get("loaded_skills", {})
        if loaded.get(name) != sha256 or skill["sha256"] != sha256:
            raise PermissionError("Load this skill in the current session; changed skills require reloading")
        invocations = _context.session.metadata.get("skill_invocations", 0)
        if invocations >= 8:
            raise RuntimeError("Skill invocation limit exceeded")
        _context.session.metadata["skill_invocations"] = invocations + 1
        invocation_id = uuid.uuid4().hex
        self.traces.emit(_context.run_id, "skill.invocation.started", _context.session.session_id,
                         {"name": name, "sha256": sha256, "tool": skill["tool"], "invocation_id": invocation_id})
        result = _context.executor.execute(_context.run_id, _context.session, _context.agent,
                    ToolCall("skill_" + invocation_id, skill["tool"], arguments))
        path = self.artifact_root / "skills" / _context.run_id / (invocation_id + ".json")
        record = {"name": name, "version": skill["version"], "sha256": sha256, "tool": skill["tool"],
                  "status": result.status, "arguments": arguments, "result": result.to_dict(), "artifact_path": str(path)}
        write_json(path, record)
        self.traces.emit(_context.run_id, "skill.invocation.completed" if result.status == "success" else "skill.invocation.failed",
                         _context.session.session_id, {"name": name, "status": result.status, "artifact_path": str(path)})
        if result.status != "success":
            raise RuntimeError(f"Skill action {result.status}: {result.error_message}; retained artifact: {path}")
        return record
