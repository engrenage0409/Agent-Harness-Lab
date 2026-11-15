"""Local UTF-8 document loading and section-aware Markdown chunking."""
from pathlib import Path
import re


def chunk_text(text, size=500, overlap=80):
    """Legacy character splitter for plain .txt documents."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk settings")
    if not text.strip():
        return []
    return [text[start:start + size] for start in range(0, len(text), size - overlap)
            if text[start:start + size].strip()]


def _safe_cut(text, limit):
    """Prefer the last whitespace boundary; never split English words when possible."""
    if len(text) <= limit:
        return len(text)
    matches = list(re.finditer(r"\s+", text[:limit + 1]))
    if matches:
        cut = matches[-1].start()
        if cut > 0:
            return cut
    # CJK and unbroken strings have no word boundaries; progress is mandatory.
    return limit


def _split_large_block(block, budget, overlap):
    """Split oversized paragraphs on word boundaries, retaining bounded context."""
    remaining = block.strip()
    parts = []
    while remaining:
        if len(remaining) <= budget:
            parts.append(remaining)
            break
        cut = _safe_cut(remaining, budget)
        piece = remaining[:cut].rstrip()
        parts.append(piece)
        # Reuse only complete words and avoid repeating the entire piece.
        context = piece[-min(overlap, max(0, len(piece) // 3)):]
        # Drop a partial leading word from the overlap window.
        if context and not context[0].isspace():
            match = re.search(r"\\s+", context)
            context = context[match.end():] if match else ""
        remaining = (context + " " if context else "") + remaining[cut:].lstrip()
    return parts


def chunk_markdown(text, size=500, overlap=80):
    """Split Markdown by headings/paragraphs, repeating heading context per chunk.

    Headings are recognized outside fenced code blocks. Source IDs are assigned
    by the caller, so the same passage continues to use '<file>#chunk-<n>'.
    """
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk settings")
    if not text.strip():
        return []

    # Preserve heading hierarchy; headings inside fenced code are not sections.
    sections = []
    headings = []
    body = []
    in_fence = False

    def flush():
        nonlocal body
        if any(line.strip() for line in body):
            sections.append((list(headings), "\n".join(body).strip()))
            body = []

    for line in text.splitlines():
        if re.match(r"^\s{0,3}(`{3,}|~{3,})", line):
            in_fence = not in_fence
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line) if not in_fence else None
        if heading:
            flush()
            level = len(heading.group(1))
            headings = headings[:level - 1] + [heading.group(2).strip()]
        else:
            body.append(line)
    flush()

    output = []
    for titles, content in sections:
        prefix = (" / ".join(titles) + "\n\n") if titles else ""
        # A long heading path is truncated only if it exceeds the whole budget.
        prefix = prefix[:max(0, size // 2)]
        budget = max(1, size - len(prefix))
        # A simple paragraph-first splitter keeps headings, bullets and code lines
        # together when they fit; oversized paragraphs fall back to word boundaries.
        blocks = re.split(r"\n\s*\n", content) if content else [""]
        current = ""
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            new = block if not current else current + "\n\n" + block
            if len(new) <= budget:
                current = new
                continue
            if current:
                output.append(prefix + current)
                current = ""
            if len(block) <= budget:
                current = block
            else:
                pieces = _split_large_block(block, budget, min(overlap, budget // 4))
                output.extend(prefix + p for p in pieces[:-1])
                current = pieces[-1]
        if current:
            output.append(prefix + current)
        elif not content.strip() and prefix:
            output.append(prefix.strip())
    return [part for part in output if part.strip()]


def load_documents(root, extensions=(".md", ".txt")):
    root = Path(root).resolve()
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if not path.resolve().is_relative_to(root):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        result.append({"source": str(path.relative_to(root)),
                       "text": path.read_text(encoding="utf-8")})
    return result
