"""Check source/configuration files without importing or executing the project.

Uses only the standard library. Does not run pytest, models, training or downloads.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def check(root: Path) -> dict:
    errors, trees, hashes, dependencies = [], {}, {}, set()
    sources = sorted(path for path in root.rglob("*.py")
                     if not any(part in {".venv", "__pycache__", ".pytest_cache", ".test-tmp", "build", "dist"}
                                for part in path.relative_to(root).parts))
    for path in sources:
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            tree = ast.parse(source, filename=relative)
            compile(tree, relative, "exec")  # Compile only: no exec(), imports or bytecode files.
            trees[path] = tree
        except (SyntaxError, ValueError) as error:
            errors.append(f"{relative}: {error}")

    package = root / "masked_longtext"

    def module_exists(parts):
        target = root.joinpath(*parts)
        return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()

    forbidden = {"latent_diffusion", "autoencoder", "common", "omegaconf", "lightning", "pytorch_lightning", "einops"}
    for path, tree in trees.items():
        if not path.is_relative_to(package):
            continue
        relative = path.relative_to(root)
        parents = list(relative.parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = parents[:len(parents) - node.level + 1]
                parts = base + (node.module.split(".") if node.module else [])
                if not base or not module_exists(parts):
                    errors.append(f"{relative.as_posix()}:{node.lineno}: unresolved relative module {node.module}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    first = name.split(".")[0]
                    dependencies.add(first)
                    if first in forbidden:
                        errors.append(f"{relative.as_posix()}:{node.lineno}: machine-specific/removed import {name}")
                    if first == "masked_longtext" and not module_exists(name.split(".")):
                        errors.append(f"{relative.as_posix()}:{node.lineno}: missing local module {name}")

    fields = {}
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {
                "CodecConfig", "DenoiserConfig", "LossConfig", "TrainingConfig", "SamplingConfig"
            }:
                fields[node.name] = {item.target.id for item in node.body
                                     if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    configs = sorted((root / "configs").glob("*.json"))
    for path in configs:
        try:
            config = json.loads(path.read_text(encoding="utf-8"),
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))
            if not isinstance(config, dict):
                raise ValueError("expected JSON object")
            if "sampling" in config:
                sections = {"sampling": "SamplingConfig"}
                allowed = {"sampling", "description"}
            else:
                sections = {"model": "CodecConfig" if path.name.startswith("codec_") else "DenoiserConfig",
                            "training": "TrainingConfig", "loss": "LossConfig"}
                allowed = {"model", "training", "loss", "description", "text_length"}
                length = config.get("text_length")
                if type(length) is not int or length < 4 or length % 4:
                    raise ValueError("text_length must be a positive integer divisible by four")
                if "model" not in config or "training" not in config:
                    raise ValueError("model and training are required")
            if set(config) - allowed:
                raise ValueError(f"unknown top-level keys: {sorted(set(config) - allowed)}")
            for section, class_name in sections.items():
                if section not in config:
                    continue
                if class_name not in fields or not isinstance(config[section], dict):
                    raise ValueError(f"missing config class or invalid section: {section}")
                unknown = set(config[section]) - fields[class_name]
                if unknown:
                    raise ValueError(f"unknown {section} keys: {sorted(unknown)}")
        except (ValueError, TypeError) as error:
            errors.append(f"configs/{path.name}: {error}")

    test_functions = sum(1 for path, tree in trees.items() if path.parent.name == "tests"
                         for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
    return {"check": "static syntax, relative module paths and JSON configuration keys",
            "python_version": sys.version.split()[0], "python_files": len(sources),
            "configuration_files": len(configs), "test_function_definitions": test_functions,
            "runtime_tests_run": False, "project_imported": False, "models_executed": False,
            "external_import_names_in_source": sorted(dependencies), "errors": errors,
            "source_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(Path(__file__).resolve().parents[1])
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
