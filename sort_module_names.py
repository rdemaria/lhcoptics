#!/usr/bin/env python3
"""Sort module definitions according to code_style.md."""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Block:
    """A source block attached to a statement."""

    node: ast.stmt
    outer_start: int
    code_start: int
    outer_end: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sort module definitions according to code_style.md."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Rewrite files in place instead of opening a dry-run diff.",
    )
    mode_group.add_argument(
        "--check-only",
        action="store_true",
        help="Only run sorting and sanity checks without diffing or writing.",
    )
    parser.add_argument(
        "modules",
        nargs="+",
        type=Path,
        help="Path(s) to the Python module(s) to sort.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    internal_roots = discover_internal_roots(Path.cwd())
    exit_code = 0

    for module in args.modules:
        try:
            changed = process_file(
                module,
                in_place=args.in_place,
                check_only=args.check_only,
                internal_roots=internal_roots,
            )
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            print(f"{module}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        if not changed:
            action = "ok" if args.check_only else "already sorted"
            print(f"{module}: {action}")

    return exit_code


def process_file(
    module: Path,
    *,
    in_place: bool,
    check_only: bool,
    internal_roots: set[str],
) -> bool:
    """Sort one file and either rewrite it, check it, or show a diff."""
    source, encoding = read_source(module)
    sorted_source = sort_module_source(source, internal_roots=internal_roots)
    sanity_check_source(sorted_source, filename=str(module))

    if sorted_source == source:
        return False

    if check_only:
        print(f"{module}: needs sorting")
        return True

    if in_place:
        module.write_text(sorted_source, encoding=encoding)
        print(f"{module}: updated")
        return True

    if launch_meld(module, sorted_source, encoding=encoding):
        print(f"{module}: spawned meld")
        return True

    print_diff(module, source, sorted_source)
    return True


def read_source(path: Path) -> tuple[str, str]:
    """Read a file using its declared source encoding."""
    with path.open("rb") as stream:
        encoding, _ = tokenize.detect_encoding(stream.readline)
    return path.read_text(encoding=encoding), encoding


def sort_module_source(source: str, *, internal_roots: set[str]) -> str:
    """Return the source text with supported definitions sorted."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    if not tree.body:
        return source

    blocks = collect_blocks(tree.body, lines, len(lines) + 1)
    prefix = "".join(lines[: blocks[0].outer_start - 1])
    parts = [prefix]
    start_index = 0

    if is_docstring_stmt(blocks[0].node):
        parts.append(raw_block_text(blocks[0], lines))
        start_index = 1

    rendered = render_sorted_sequence(
        blocks[start_index:],
        classify=classify_module_block,
        sort_key=lambda block, index: module_sort_key(
            block, index=index, internal_roots=internal_roots
        ),
        render_block=lambda block: render_module_block(
            block, lines, internal_roots=internal_roots
        ),
        separator=lambda left, right: module_separator(
            left, right, internal_roots=internal_roots
        ),
    )
    if start_index and start_index < len(blocks) and classify_module_block(
        blocks[start_index].node
    ) != "other" and rendered:
        parts.append("\n")
    parts.append(rendered)
    return "".join(parts)


def sanity_check_source(source: str, *, filename: str) -> None:
    """Verify that generated source can be parsed and compiled."""
    tree = ast.parse(source, filename=filename)
    compile(tree, filename, "exec")


def render_module_block(
    block: Block, lines: Sequence[str], *, internal_roots: set[str]
) -> str:
    """Render a top-level block."""
    if isinstance(block.node, ast.ClassDef):
        return render_class_block(block, lines, internal_roots=internal_roots)
    return raw_block_text(block, lines)


def render_class_block(
    block: Block, lines: Sequence[str], *, internal_roots: set[str]
) -> str:
    """Render a class block with its supported members sorted."""
    node = block.node
    if not isinstance(node, ast.ClassDef) or not node.body:
        return raw_block_text(block, lines)

    suite_end = find_suite_end(node, lines, default_end=block.outer_end)
    child_blocks = collect_blocks(node.body, lines, suite_end)
    header = "".join(lines[block.outer_start - 1 : child_blocks[0].outer_start - 1])
    tail = "".join(lines[suite_end - 1 : block.outer_end - 1])

    body_parts = [header]
    start_index = 0
    if is_docstring_stmt(child_blocks[0].node):
        body_parts.append(raw_block_text(child_blocks[0], lines))
        start_index = 1

    rendered = render_sorted_sequence(
        child_blocks[start_index:],
        classify=classify_class_block,
        sort_key=class_sort_key,
        render_block=lambda child: raw_block_text(child, lines),
        separator=class_separator,
    )
    if start_index and start_index < len(child_blocks) and classify_class_block(
        child_blocks[start_index].node
    ) != "other" and rendered:
        body_parts.append("\n")
    body_parts.append(rendered)
    body_parts.append(tail)
    return "".join(body_parts)


def render_sorted_sequence(
    blocks: Sequence[Block],
    *,
    classify,
    sort_key,
    render_block,
    separator,
) -> str:
    """Sort supported runs while leaving unsupported blocks in place."""
    parts: list[str] = []
    sortable_run: list[tuple[int, Block]] = []

    def flush_run() -> None:
        if not sortable_run:
            return
        ordered = sorted(
            sortable_run, key=lambda item: sort_key(item[1], item[0])
        )
        previous_block: Block | None = None
        for _, block in ordered:
            text = trim_blank_edges(render_block(block))
            if previous_block is not None:
                parts.append(separator(previous_block, block))
            parts.append(text)
            previous_block = block
        sortable_run.clear()

    for index, block in enumerate(blocks):
        if classify(block.node) == "other":
            flush_run()
            parts.append(render_block(block))
        else:
            sortable_run.append((index, block))

    flush_run()
    return "".join(parts)


def collect_blocks(
    statements: Sequence[ast.stmt], lines: Sequence[str], end_limit: int
) -> list[Block]:
    """Collect source blocks for a suite of statements."""
    if not statements:
        return []

    raw_blocks: list[tuple[ast.stmt, int, int]] = []
    previous_code_end = 0

    for statement in statements:
        code_start = statement_start_line(statement)
        outer_start = attached_comment_start(
            code_start, previous_code_end=previous_code_end, lines=lines
        )
        raw_blocks.append((statement, outer_start, code_start))
        previous_code_end = statement.end_lineno or code_start

    blocks: list[Block] = []
    for index, (statement, outer_start, code_start) in enumerate(raw_blocks):
        next_start = (
            raw_blocks[index + 1][1] if index + 1 < len(raw_blocks) else end_limit
        )
        blocks.append(
            Block(
                node=statement,
                outer_start=outer_start,
                code_start=code_start,
                outer_end=next_start,
            )
        )

    return blocks


def statement_start_line(statement: ast.stmt) -> int:
    """Return the first source line for a statement."""
    if isinstance(
        statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ) and statement.decorator_list:
        return min(decorator.lineno for decorator in statement.decorator_list)
    return statement.lineno


def find_suite_end(node: ast.ClassDef, lines: Sequence[str], *, default_end: int) -> int:
    """Extend a class body to include trailing indented comments and blank lines."""
    class_indent = indentation_width(lines[node.lineno - 1])
    line_number = (node.end_lineno or node.lineno) + 1

    while line_number < default_end:
        line = lines[line_number - 1]
        stripped = line.strip()
        if not stripped:
            line_number += 1
            continue
        if indentation_width(line) > class_indent:
            line_number += 1
            continue
        break

    return line_number


def raw_block_text(block: Block, lines: Sequence[str]) -> str:
    """Return the original source text for a block."""
    return "".join(lines[block.outer_start - 1 : block.outer_end - 1])


def attached_comment_start(
    code_start: int, *, previous_code_end: int, lines: Sequence[str]
) -> int:
    """Return the earliest line attached to a statement via leading comments."""
    outer_start = code_start
    seen_comment = False

    while outer_start > previous_code_end + 1:
        line = lines[outer_start - 2]
        stripped = line.strip()
        if stripped.startswith("#"):
            outer_start -= 1
            seen_comment = True
            continue
        if not stripped and seen_comment:
            outer_start -= 1
            continue
        break

    return outer_start


def indentation_width(line: str) -> int:
    """Return the indentation width of a line."""
    return len(line) - len(line.lstrip(" \t"))


def trim_blank_edges(text: str) -> str:
    """Strip blank lines from the start and end of a block."""
    lines = text.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "".join(lines)


def is_docstring_stmt(statement: ast.stmt) -> bool:
    """Return whether a statement is a literal string docstring."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def classify_module_block(statement: ast.stmt) -> str:
    """Classify a top-level statement for sorting."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return "import"
    if is_assignment(statement):
        return "variable"
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(statement, ast.ClassDef):
        return "class"
    return "other"


def module_sort_key(
    block: Block, *, index: int, internal_roots: set[str]
) -> tuple[object, ...]:
    """Return the sort key for a top-level block."""
    statement = block.node
    kind = classify_module_block(statement)
    if kind == "import":
        return (
            0,
            import_group(statement, internal_roots=internal_roots),
            alphabetical_key(import_name(statement)),
            index,
        )
    if kind == "variable":
        return (1, alphabetical_key(declaration_name(statement)), index)
    if kind == "function":
        return (2, alphabetical_key(statement.name), index)
    if kind == "class":
        return (3, alphabetical_key(statement.name), index)
    return (4, index)


def module_separator(
    left: Block, right: Block, *, internal_roots: set[str]
) -> str:
    """Return the separator used between sorted top-level blocks."""
    left_kind = classify_module_block(left.node)
    right_kind = classify_module_block(right.node)

    if left_kind == "import" and right_kind == "import":
        left_group = import_group(left.node, internal_roots=internal_roots)
        right_group = import_group(right.node, internal_roots=internal_roots)
        return "\n" if left_group != right_group else ""

    if left_kind == "variable" and right_kind == "variable":
        return ""

    if left_kind in {"function", "class"} or right_kind in {"function", "class"}:
        return "\n"

    return "\n"


def classify_class_block(statement: ast.stmt) -> str:
    """Classify a class member for sorting."""
    if is_assignment(statement):
        return "attribute"
    if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "other"

    property_kind = property_kind_for(statement)
    if property_kind is not None:
        return property_kind
    if has_decorator(statement, "staticmethod"):
        return "staticmethod"
    if statement.name == "__init__":
        return "init"
    if has_decorator(statement, "classmethod"):
        return "classmethod"
    if is_dunder_name(statement.name):
        return "dunder"
    if statement.name.startswith("_") and not statement.name.startswith("__"):
        return "private"
    return "method"


def class_sort_key(block: Block, index: int) -> tuple[object, ...]:
    """Return the sort key for a class member."""
    statement = block.node
    kind = classify_class_block(statement)
    order = {
        "attribute": 0,
        "staticmethod": 1,
        "init": 2,
        "classmethod": 3,
        "dunder": 4,
        "property": 5,
        "private": 6,
        "method": 7,
        "other": 8,
    }
    if kind == "attribute":
        return (order[kind], alphabetical_key(declaration_name(statement)), index)
    if kind == "property" and isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        property_name, property_order = property_sort_info(statement)
        return (
            order[kind],
            alphabetical_key(property_name),
            property_order,
            index,
        )
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (order[kind], alphabetical_key(statement.name), index)
    return (order[kind], index)


def class_separator(left: Block, right: Block) -> str:
    """Return the separator used between sorted class members."""
    left_kind = classify_class_block(left.node)
    right_kind = classify_class_block(right.node)
    if left_kind == "attribute" and right_kind == "attribute":
        return ""
    return "\n"


def is_assignment(statement: ast.stmt) -> bool:
    """Return whether a statement is treated as a declaration assignment."""
    type_alias = getattr(ast, "TypeAlias", None)
    return isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)) or (
        type_alias is not None and isinstance(statement, type_alias)
    )


def declaration_name(statement: ast.stmt) -> str:
    """Return a representative bound name for a declaration."""
    names: list[str] = []
    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            names.extend(extract_target_names(target))
    elif isinstance(statement, ast.AnnAssign):
        names.extend(extract_target_names(statement.target))
    elif isinstance(statement, ast.AugAssign):
        names.extend(extract_target_names(statement.target))
    else:
        type_alias = getattr(ast, "TypeAlias", None)
        if type_alias is not None and isinstance(statement, type_alias):
            names.extend(extract_target_names(statement.name))
    if not names:
        return ""
    return min(names, key=alphabetical_key)


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


def import_group(statement: ast.stmt, *, internal_roots: set[str]) -> int:
    """Group imports as future, standard-library, installed, or internal."""
    if isinstance(statement, ast.ImportFrom):
        if statement.level > 0:
            return 3
        if statement.module == "__future__":
            return 0

    roots = import_roots(statement)
    if roots and all(root in sys.stdlib_module_names for root in roots):
        return 1
    if roots and all(root in internal_roots for root in roots):
        return 3
    return 2


def import_roots(statement: ast.stmt) -> list[str]:
    """Return the imported root module names for a statement."""
    if isinstance(statement, ast.Import):
        return [alias.name.split(".", 1)[0] for alias in statement.names]
    if isinstance(statement, ast.ImportFrom):
        if statement.level > 0:
            return []
        if statement.module:
            return [statement.module.split(".", 1)[0]]
    return []


def import_name(statement: ast.stmt) -> str:
    """Return a stable text key for an import."""
    if isinstance(statement, ast.Import):
        return ",".join(alias.name for alias in statement.names)
    if isinstance(statement, ast.ImportFrom):
        module = statement.module or ""
        aliases = ",".join(alias.name for alias in statement.names)
        return f"{'.' * statement.level}{module}:{aliases}"
    return ""


def has_decorator(
    statement: ast.FunctionDef | ast.AsyncFunctionDef, decorator_name: str
) -> bool:
    """Return whether a function has a decorator with the given leaf name."""
    return any(last_name(decorator) == decorator_name for decorator in statement.decorator_list)


def property_kind_for(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    """Return whether a function is property-related."""
    for decorator in statement.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return "property"
        if isinstance(decorator, ast.Attribute) and decorator.attr in {
            "getter",
            "setter",
            "deleter",
        }:
            return "property"
    return None


def property_sort_info(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, int]:
    """Return the name and relative order for a property-related method."""
    for decorator in statement.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return statement.name, 0
        if isinstance(decorator, ast.Attribute) and decorator.attr in {
            "getter",
            "setter",
            "deleter",
        }:
            base_name = last_name(decorator.value) or statement.name
            order = {"getter": 0, "setter": 1, "deleter": 2}[decorator.attr]
            return base_name, order
    return statement.name, 0


def last_name(node: ast.AST) -> str | None:
    """Return the right-most name from a decorator expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return last_name(node.func)
    return None


def is_dunder_name(name: str) -> bool:
    """Return whether a name is a dunder method name."""
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def alphabetical_key(value: str) -> tuple[tuple[int, int], ...]:
    """Apply the repository's _A-Za-z0-9 ordering to a string."""
    return tuple((alphabetical_rank(char), ord(char)) for char in value)


def alphabetical_rank(char: str) -> int:
    """Return the group rank for a single character."""
    if char == "_":
        return 0
    if "A" <= char <= "Z":
        return 1
    if "a" <= char <= "z":
        return 2
    if "0" <= char <= "9":
        return 3
    return 4


def discover_internal_roots(base: Path) -> set[str]:
    """Discover import roots that belong to the current repository."""
    roots: set[str] = set()
    for candidate in (base, base / "src"):
        if not candidate.is_dir():
            continue
        for path in candidate.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_dir() and (path / "__init__.py").exists():
                roots.add(path.name)
            elif path.suffix == ".py":
                roots.add(path.stem)
    return roots


def launch_meld(path: Path, sorted_source: str, *, encoding: str) -> bool:
    """Launch meld for a dry-run diff if the environment supports it."""
    if shutil.which("meld") is None:
        return False
    if not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")):
        return False

    with tempfile.NamedTemporaryFile(
        "w",
        encoding=encoding,
        suffix=path.suffix,
        prefix=f"{path.stem}.sorted.",
        delete=False,
    ) as stream:
        stream.write(sorted_source)
        temp_path = Path(stream.name)

    try:
        subprocess.Popen(["meld", str(path), str(temp_path)])
    except OSError:
        temp_path.unlink(missing_ok=True)
        return False
    return True


def print_diff(path: Path, source: str, sorted_source: str) -> None:
    """Print a unified diff when meld is unavailable."""
    diff = difflib.unified_diff(
        source.splitlines(),
        sorted_source.splitlines(),
        fromfile=str(path),
        tofile=f"{path} (sorted)",
        lineterm="",
    )
    for line in diff:
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
