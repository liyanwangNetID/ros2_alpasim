#!/usr/bin/env python3
"""Finish Step 7 test-source hygiene without changing test behavior."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests" / "step7"

DUPLICATE_CLASS_FILES = (
    TEST_ROOT / "test_geometry.py",
    TEST_ROOT / "test_occlusion.py",
    TEST_ROOT / "test_raster.py",
)

MERGES = (
    (
        TEST_ROOT / "test_actor_observability_projection_adapter_v01.py",
        TEST_ROOT / "test_observability.py",
        "projection_adapter",
    ),
    (
        TEST_ROOT / "test_projected_triangle_degenerate_depth_samples_v01.py",
        TEST_ROOT / "test_raster.py",
        "degenerate_depth",
    ),
)


class NameRewriter(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self.mapping.get(node.name, node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self.mapping.get(node.name, node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self.mapping.get(node.name, node.name)
        self.generic_visit(node)
        return node


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def write(path: Path, tree: ast.Module) -> None:
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")


def isolate_redefined_classes(path: Path) -> list[tuple[str, str]]:
    tree = parse(path)
    totals: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            totals[node.name] = totals.get(node.name, 0) + 1

    duplicated = {name for name, count in totals.items() if count > 1}
    occurrence: dict[str, int] = {}
    active_names: dict[str, str] = {}
    replacements: list[tuple[str, str]] = []
    new_body: list[ast.stmt] = []

    for node in tree.body:
        node = NameRewriter(active_names).visit(node)
        if isinstance(node, ast.ClassDef):
            original_name = next(
                (
                    name
                    for name in duplicated
                    if active_names.get(name, name) == node.name
                    or name == node.name
                ),
                node.name,
            )
            if original_name in duplicated:
                index = occurrence.get(original_name, 0) + 1
                occurrence[original_name] = index
                replacement = (
                    original_name
                    if index == 1
                    else f"{original_name}_{index}"
                )
                node.name = replacement
                active_names[original_name] = replacement
                if replacement != original_name:
                    replacements.append((original_name, replacement))
        new_body.append(node)

    tree.body = new_body
    write(path, tree)
    return replacements


def import_key(node: ast.stmt) -> str:
    return ast.dump(node, include_attributes=False)


def top_level_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    }


def merge_test_file(source: Path, target: Path, prefix: str) -> None:
    if not source.exists():
        print("ABSENT", source.relative_to(ROOT))
        return

    source_tree = parse(source)
    target_tree = parse(target)
    used = top_level_names(target_tree)

    source_names = top_level_names(source_tree)
    mapping: dict[str, str] = {}
    for name in sorted(source_names):
        candidate = name
        if not name.startswith("test_"):
            candidate = f"{prefix}_{name}"
        elif candidate in used:
            candidate = f"test_{prefix}_{name[5:]}"
        if candidate in used:
            raise RuntimeError(
                f"Unable to isolate {name} while merging {source.name}"
            )
        mapping[name] = candidate
        used.add(candidate)

    source_tree = NameRewriter(mapping).visit(source_tree)
    ast.fix_missing_locations(source_tree)

    existing_imports = {
        import_key(node)
        for node in target_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    imports: list[ast.stmt] = []
    bodies: list[ast.stmt] = []

    for node in source_tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            key = import_key(node)
            if key not in existing_imports:
                imports.append(node)
                existing_imports.add(key)
        else:
            bodies.append(node)

    insertion_index = 0
    while insertion_index < len(target_tree.body):
        node = target_tree.body[insertion_index]
        if isinstance(node, ast.Expr) and isinstance(
            node.value, ast.Constant
        ) and isinstance(node.value.value, str):
            insertion_index += 1
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_index += 1
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insertion_index += 1
            continue
        break

    target_tree.body[insertion_index:insertion_index] = imports
    target_tree.body.extend(bodies)
    write(target, target_tree)
    source.unlink()
    print("MERGED", source.relative_to(ROOT), "INTO", target.relative_to(ROOT))


def duplicate_top_level_definitions(path: Path) -> list[str]:
    names: list[str] = []
    for node in parse(path).body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            names.append(node.name)
    return sorted({name for name in names if names.count(name) > 1})


def main() -> int:
    for path in DUPLICATE_CLASS_FILES:
        replacements = isolate_redefined_classes(path)
        for original, replacement in replacements:
            print(
                "RENAMED",
                path.relative_to(ROOT),
                original,
                "TO",
                replacement,
            )

    for source, target, prefix in MERGES:
        merge_test_file(source, target, prefix)

    failures: list[tuple[str, list[str]]] = []
    for path in sorted(TEST_ROOT.iterdir()):
        if not path.is_file() or path.suffix != ".py":
            continue
        duplicates = duplicate_top_level_definitions(path)
        if duplicates:
            failures.append((str(path.relative_to(ROOT)), duplicates))

    if failures:
        for failure in failures:
            print("DUPLICATE", failure)
        raise RuntimeError("Duplicate top-level definitions remain")

    for cache in sorted(TEST_ROOT.rglob("__pycache__"), reverse=True):
        for item in cache.iterdir():
            if item.is_file():
                item.unlink()
        cache.rmdir()

    print("duplicate top-level definitions: 0")
    print("PASS: Step 7 test-source hygiene completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
