import os
import json


class SkillManager:
    """管理 skill.json 的加载、解析和更新"""

    def __init__(self, exp_dir: str):
        self.skill_path = os.path.join(exp_dir, "skill.json")

    def load(self):
        """加载 skills"""
        if os.path.exists(self.skill_path):
            with open(self.skill_path, "r", encoding="utf-8") as f:
                skills = json.load(f)
                return skills
        return []

    def load_text(self):
        """加载给 Architect 使用的 skill 正文文本。"""
        if not os.path.exists(self.skill_path):
            return ""
        with open(self.skill_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return ""

        try:
            skills = json.loads(raw)
        except json.JSONDecodeError:
            return raw

        if isinstance(skills, list):
            lines = []
            for skill in skills:
                text = skill.get("content", skill) if isinstance(skill, dict) else skill
                if isinstance(text, (dict, list)):
                    text = json.dumps(text, ensure_ascii=False)
                else:
                    text = str(text).strip()
                if text:
                    lines.append(text)
            return "\n".join(lines)

        return raw

    def save(self, skills):
        """保存 skills"""
        with open(self.skill_path, "w", encoding="utf-8") as f:
            json.dump(skills, f, ensure_ascii=False, indent=2)

    def parse_task_skill(self, output: str):
        """解析 LLM 输出中的 JSON object。"""
        text = output.strip()
        if not text:
            raise ValueError("Task draft output is empty; the LLM likely returned no text.")
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if not text:
            raise ValueError("Task draft output is empty after removing markdown fences.")


        section_names = {"model", "data", "training", "loss"}
        parsed = {}
        current_key = None

        def append_item(key: str, value: str):
            value = value.strip()
            if not value:
                return
            parsed.setdefault(key, [])
            parsed[key].append(value)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            while line.startswith(("#", "*")):
                line = line[1:].strip()
            if line.startswith("- "):
                item = line[2:].strip()
                if current_key is None:
                    append_item("entries", item)
                else:
                    append_item(current_key, item)
                continue

            if len(line) > 2 and line[0].isdigit() and line[1] in {".", ")"}:
                item = line[2:].strip()
                if current_key is None:
                    append_item("entries", item)
                else:
                    append_item(current_key, item)
                continue

            key_part, sep, value_part = line.partition(":")
            key = key_part.strip().lower()
            value = value_part.strip()
            if sep and key in section_names:
                current_key = key
                if value:
                    append_item(current_key, value)
                else:
                    parsed.setdefault(current_key, [])
                continue

            if line.lower() in section_names:
                current_key = line.lower()
                parsed.setdefault(current_key, [])
                continue

            if current_key is not None:
                append_item(current_key, line)
            else:
                append_item("entries", line)

        if parsed:
            if "contrasts" in parsed and all(isinstance(x, str) for x in parsed["contrasts"]):
                parsed["contrasts"] = parsed["contrasts"]
            return parsed

        raise ValueError("Task draft output must be a JSON object or a sectioned entry list.")

    def apply_actions(self, output: str, task_num: str):
        """解析 LLM 输出并执行操作"""
        skills = self.load()
        actions = self._parse(output)

        def find_skill_pos(skill_idx):
            for pos, skill in enumerate(skills):
                if skill.get("idx") == skill_idx:
                    return pos
            return None

        def lock_skill(skill):
            skill["_confidence_locked"] = True

        for action in actions:
            if action["action"] == "ADD":
                if task_num == 'global':
                    confidence = 3
                else:
                    confidence = 2

                skills.append({
                    "idx": len(skills) + 1,
                    "content": action["content"],
                    "confidence": confidence,
                    "task": task_num,
                    "_confidence_locked": True,
                })

            elif action["action"] == "UPVOTE":
                pos = find_skill_pos(action["index"])
                if pos is not None and not skills[pos].get("_confidence_locked", False):
                    skills[pos]["confidence"] += 1
                    skills[pos]["task"] += f", {task_num}"
                    lock_skill(skills[pos])

            elif action["action"] == "DOWNVOTE":
                pos = find_skill_pos(action["index"])
                if pos is not None and not skills[pos].get("_confidence_locked", False):
                    skills[pos]["confidence"] -= 1
                    if skills[pos]["confidence"] <= 0:
                        skills.pop(pos)
                    else:
                        lock_skill(skills[pos])

            elif action["action"] == "EDIT":
                pos = find_skill_pos(action["index"])
                if pos is not None:
                    skills[pos]["content"] += " " + action["content"]
                    if not skills[pos].get("_confidence_locked", False):
                        skills[pos]["confidence"] += 1
                        skills[pos]["task"] += f", {task_num}"
                        lock_skill(skills[pos])

        for skill in skills:
            skill.pop("_confidence_locked", None)

        self.save(skills)

    def _parse(self, output: str):
        """解析操作指令"""
        actions = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.startswith("ADD:"):
                actions.append({"action": "ADD", "content": line[4:].strip()})
            elif line.startswith("UPVOTE:"):
                actions.append({"action": "UPVOTE", "index": int(line[7:].strip())})
            elif line.startswith("DOWNVOTE:"):
                actions.append({"action": "DOWNVOTE", "index": int(line[9:].strip())})
            elif line.startswith("EDIT:"):
                parts = line[5:].strip().split("|", 1)
                if len(parts) == 2:
                    actions.append({"action": "EDIT", "index": int(parts[0]), "content": parts[1].strip()})
        return actions
