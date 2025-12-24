import os


def _yaml_escape(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    needs_quote = any(ch in text for ch in ":#{}[],'\"\\\n\t") or text.strip() != text
    if needs_quote:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _yaml_lines(value, indent):
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key in sorted(value.keys(), key=str):
            val = value[key]
            if isinstance(val, (dict, list, tuple)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(val, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_escape(val)}")
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_escape(item)}")
        return lines
    return [f"{prefix}{_yaml_escape(value)}"]


def dump_args_yaml(args, output_dir, filename="train_args.yaml"):
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    data = vars(args).copy()
    lines = _yaml_lines(data, 0)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
