#!/usr/bin/env python3
"""Print declarations found in a Python module."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Declaration:
    """A declared name and its kind within a scope."""

    kind: str
    name: str
    node: ast.AST | None = None


BLOCK_FIELDS = (
    "body",
    "orelse",
    "finalbody",
)


def collect_declarations(body: list[ast.stmt], *, in_class: bool) -> list[Declaration]:
    """Collect declarations that belong to the current module or class scope."""
    declarations: list[Declaration] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, kind: str, node: ast.AST | None = None) -> None:
        key = (name, kind)
        if key not in seen:
            declarations.append(Declaration(name=name, kind=kind, node=node))
            seen.add(key)

    def visit_statements(statements: Iterable[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                add(statement.name, "class", statement)
                continue

            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
                if in_class:
                    kind = classify_method(statement)
                add(statement.name, kind, statement)
                continue

            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    for name in extract_target_names(target):
                        add(name, declaration_kind(in_class))
                continue

            if isinstance(statement, ast.AnnAssign):
                for name in extract_target_names(statement.target):
                    add(name, declaration_kind(in_class))
                continue

            if isinstance(statement, ast.AugAssign):
                for name in extract_target_names(statement.target):
                    add(name, declaration_kind(in_class))
                continue

            type_alias = getattr(ast, "TypeAlias", None)
            if type_alias is not None and isinstance(statement, type_alias):
                for name in extract_target_names(statement.name):
                    add(name, declaration_kind(in_class))
                continue

            for field_name in BLOCK_FIELDS:
                nested = getattr(statement, field_name, None)
                if isinstance(nested, list) and nested and all(
                    isinstance(item, ast.stmt) for item in nested
                ):
                    visit_statements(nested)
            handlers = getattr(statement, "handlers", None)
            if handlers:
                for handler in handlers:
                    if handler.name:
                        add(handler.name, declaration_kind(in_class))
                    visit_statements(handler.body)
            match_cases = getattr(statement, "cases", None)
            if match_cases:
                for case in match_cases:
                    visit_statements(case.body)

    visit_statements(body)
    return declarations


def classify_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Classify a function declared inside a class."""
    for decorator in node.decorator_list:
        name = last_name(decorator)
        if name == "staticmethod":
            return "staticmethod"
        if name == "classmethod":
            return "classmethod"
    return "method"


def declaration_kind(in_class: bool) -> str:
    """Return the label used for assignments in the current scope."""
    return "attribute" if in_class else "variable"


def extract_target_names(target: ast.expr) -> list[str]:
    """Extract names bound by an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in target.elts:
            names.extend(extract_target_names(item))
        return names
    if isinstance(target, ast.Starred):
        return extract_target_names(target.value)
    return []


def last_name(node: ast.AST) -> str | None:
    """Return the right-most name from a decorator expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return last_name(node.func)
    return None


def format_declaration(declaration: Declaration) -> str:
    """Format a declaration for output."""
    if declaration.kind in {"method", "staticmethod", "classmethod"} and isinstance(
        declaration.node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return format_method(declaration.node)
    return declaration.name


def format_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format a method using its first argument."""
    positional_args = [*node.args.posonlyargs, *node.args.args]
    if not positional_args:
        return f"{node.name}()"

    first_arg = positional_args[0].arg
    has_more_args = (
        len(positional_args) > 1
        or bool(node.args.vararg)
        or bool(node.args.kwonlyargs)
        or bool(node.args.kwarg)
    )
    suffix = ", ..." if has_more_args else ""
    return f"{node.name}({first_arg}{suffix})"


def print_scope(body: list[ast.stmt], *, indent: int = 0) -> None:
    """Print declarations for the given scope."""
    declarations = collect_declarations(body, in_class=indent > 0)
    prefix = " " * indent

    for declaration in declarations:
        print(f"{prefix}{format_declaration(declaration)}")
        if declaration.kind == "class" and isinstance(declaration.node, ast.ClassDef):
            print_scope(declaration.node.body, indent=indent + 4)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Print names declared in a Python module."
    )
    parser.add_argument("module", type=Path, help="Path to the Python module to inspect.")
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    source = args.module.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(args.module))
    print_scope(tree.body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
