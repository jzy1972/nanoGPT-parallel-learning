"""Static checks for the learning project; this script does not import PyTorch."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def check_project(project: Path, *, model_only: bool = False) -> list[str]:
    errors: list[str] = []
    files = ["model.py"] if model_only else ["model.py", "train.py"]
    for filename in files:
        path = project / filename
        if not path.is_file():
            errors.append(f"{filename}: file not found in {project}")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{filename}: {exc}")
            continue
        if filename == "model.py":
            gpt = next(
                (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPT"),
                None,
            )
            if gpt is None:
                errors.append("model.py: GPT class not found")
            else:
                methods = {n.name: n for n in gpt.body if isinstance(n, ast.FunctionDef)}
                loader = methods.get("load_dense_state_dict")
                if loader is None:
                    errors.append("model.py: GPT.load_dense_state_dict is missing")
                elif any(
                    isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Attribute)
                    and n.value.attr == "state_dict"
                    for n in ast.walk(loader)
                ):
                    errors.append(
                        "model.py: use self.state_dict(), with parentheses, "
                        "when obtaining expected_state"
                    )
                constructor = methods.get("__init__")
                if constructor and "tp_group" not in {
                    a.arg for a in constructor.args.args + constructor.args.kwonlyargs
                }:
                    errors.append("model.py: GPT.__init__ must accept tp_group")
            for name in ("ColumnParallelLinear", "RowParallelLinear"):
                if not any(isinstance(n, ast.ClassDef) and n.name == name for n in tree.body):
                    errors.append(f"model.py: {name} class is missing")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "F"
                    and func.attr == "Linear"
                ):
                    errors.append(f"{filename}:{node.lineno}: use F.linear, lowercase l")
                if isinstance(func, ast.Name) and func.id in ("CopyToTP", "SumFromTP"):
                    errors.append(
                        f"{filename}:{node.lineno}: call {func.id}.apply(...)"
                    )
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                    and func.attr == "barrier"
                ):
                    errors.append(f"{filename}:{node.lineno}: use dist.barrier()")
                if isinstance(func, ast.Name) and func.id == "bool" and len(node.args) > 1:
                    errors.append(
                        f"{filename}:{node.lineno}: bool accepts one argument; "
                        "use bool(control[1].item())"
                    )
            if isinstance(node, ast.Name) and node.id == "contro":
                errors.append(f"{filename}:{node.lineno}: likely typo: contro -> control")
            if (
                filename == "train.py"
                and isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "master_process"
            ):
                for body_node in node.body:
                    for call in ast.walk(body_node):
                        if (
                            isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and isinstance(call.func.value, ast.Name)
                            and call.func.value.id == "dist"
                            and call.func.attr in ("broadcast", "all_reduce", "barrier")
                        ):
                            errors.append(
                                f"train.py:{call.lineno}: collective inside "
                                "'if master_process'; verify that all group members participate"
                            )
        if filename == "train.py":
            prefixes = {
                n.values[0].value
                for n in ast.walk(tree)
                if isinstance(n, ast.JoinedStr)
                and n.values
                and isinstance(n.values[0], ast.Constant)
                and isinstance(n.values[0].value, str)
                and n.values[0].value.startswith("ckpt")
            }
            if "ckpt" in prefixes and "ckpt_tp" in prefixes:
                errors.append(
                    "train.py: both ckpt{rank}.pt and ckpt_tp{rank}.pt are present; "
                    "make resume/save filenames identical"
                )
    return list(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    errors = check_project(args.project.resolve(), model_only=args.model_only)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1
    print("[PASS] Static checks. This does not establish numerical correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
