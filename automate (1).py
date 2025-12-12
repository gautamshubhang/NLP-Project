"""
generate_flowcharts_gemini_cleaned.py (with QA step)

This version keeps your original pipeline but adds a QA pass per process:
 - After saving gemini_response.txt we call Gemini again with QA_PROMPT_TEMPLATE
   and the full Gemini response embedded in triple backticks.
 - We save the raw QA response to qa_response.txt and also attempt to parse
   numbered answers into qa.json (mapping question number -> answer text).

Behavioral notes:
 - Reuses call_gemini_chat (so same retry/timeout behavior)
 - If QA parsing fails, qa.json will contain the raw text under key "raw"
 - All other pipeline steps are unchanged.

"""
from dotenv import load_dotenv
load_dotenv()
import os
import re
import time
import json
import textwrap
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from tqdm import tqdm

# NEW GEMINI CLIENT USAGE
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    raise ImportError("The 'google-genai' package is required. Install it with: pip install google-genai")

# Ensure API key present
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")

try:
    client = genai.Client()
except Exception as e:
    raise RuntimeError(f"Gemini client initialization failed: {e}")

# ---------------------------
# Configuration
# ---------------------------

MODEL_NAME = "gemini-2.5-flash"
MMDC_SCALE = 3
OUTPUT_ROOT = Path("flowchart_outputs_gemini")
OUTPUT_ROOT.mkdir(exist_ok=True)

PROCESS_NAMES = [
    "SMR Process"
]

PROMPT_TEMPLATE = textwrap.dedent("""
Create a clean hierarchical process flowchart for the <PROCESS NAME>. Break the full process into logical stages such as feed preparation, reaction section, separation, purification, recycling, utilities, and product handling.

For each feed stream, provide:

Feed stream name

Phase

Complete composition in mole percent mass percent or a typical industrial range

Temperature and pressure entering the process

Any pretreatment requirements

For every major processing step, include:

Typical temperature range

Typical pressure range

Catalyst type and catalyst loading if applicable

Conversion or selectivity range

Phase of reaction or separation

Reactor type such as fixed bed tubular CSTR slurry fluidized bed

Separator type such as absorber stripper distillation tower knockout drum cyclone decanter membrane unit PSA

Any compressors pumps heat exchangers heaters coolers or recycle loops involved

After describing the entire process in hierarchical form generate a Mermaid flowchart code showing all steps clearly. The mermaid code should contain all the numerical data.
Important instruction The Mermaid code must contain no parentheses in any node labels Do not use characters like ( or ). Use hyphens or plain text.
""").strip()
# --- NEW QA PROMPT TEMPLATE ---
QA_PROMPT_TEMPLATE = textwrap.dedent("""
Analyze the following process description and associated Mermaid flowchart code for the <PROCESS NAME>.
The text is provided below enclosed in triple backticks.
Based *only* on the provided text, answer the following questions.

Process Text:
Questions to Answer (Provide the answer immediately after the question):

1.  How do the different feed stream compositions influence the reaction pathway and overall efficiency of this process?
2.  Which parts of the flowchart represent major heat sources and sinks?
3.  What are the main operational bottlenecks or rate‑limiting steps in the process?
4.  List the full process path from the primary feed stream to the final product, identifying each step in order without skipping any intermediate units.
5.  Label each node (unit) as an entry point, exit point, intermediate node, branching node, or merging node based on how many inputs and outputs it has.
6.  How many input and output streams does the flowchart have?
    """).strip()

MAX_TOKENS = 8192000
TEMPERATURE = 0.2
RETRY_COUNT = 3
SLEEP_BETWEEN_REQUESTS = 1.0

# If mmdc is in PATH just use 'mmdc'; otherwise set absolute path to your mmdc.cmd
MMDC_CMD = "mmdc"

# ---------------------------
# Helper functions
# ---------------------------

def call_gemini_chat(prompt: str, model: str = MODEL_NAME, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
    config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            # genai response shape may vary; use text property if available
            text = ""
            if hasattr(resp, "text") and resp.text:
                text = resp.text
            else:
                # attempt to assemble from candidate messages
                try:
                    # Some SDKs return structured candidates
                    text = "".join([c.content[0].text for c in resp.candidates]) if hasattr(resp, "candidates") else str(resp)
                except Exception:
                    text = str(resp)
            return text
        except APIError as e:
            print(f"Gemini API request failed attempt {attempt}/{RETRY_COUNT}: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as e:
            print(f"General error on attempt {attempt}/{RETRY_COUNT}: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("Gemini call failed after retries")


def extract_mermaid_code(text: str) -> Optional[str]:
    mermaid_code = None
    # 1. Look for fenced block
    m = re.search(r"```mermaid\s*(.*?)```", text, re.S | re.I)
    if m:
        mermaid_code = m.group(1).strip()
    
    # 2. Look for unfenced block (might be within general code block)
    if not mermaid_code:
        m2 = re.search(r"```(.*?)```", text, re.S)
        if m2:
            inner = m2.group(1)
            if "graph" in inner.lower():
                mermaid_code = inner.strip()
    
    # 3. Look for naked chart start (graph TD...)
    if not mermaid_code:
        idx = re.search(r"(^|\n)(graph\s+(?:TD|LR|TB|BT|RL)\b.*)", text, re.I | re.S)
        if idx:
            # Take everything from the chart start to the end of the text
            mermaid_code = idx.group(2).strip()
            # If there's non-mermaid text after it, truncate
            try:
                # Find where the mermaid structure likely ends (e.g., before the next header or block)
                end_marker = re.search(r"\n\s*(?:#|I\.|II\.|subgraph|end)\b", mermaid_code, re.I)
                if end_marker:
                    mermaid_code = mermaid_code[:end_marker.start()]
            except:
                pass # keep full block if truncation fails
    return mermaid_code


def normalize_and_clean_mermaid(raw: str, default_chart: str = "graph TD") -> str:
    """
    Enhanced normalization and cleanup:
      - Remove parentheses
      - Normalize structural keywords
      - Fix the common LLM error: missing brackets on linked node labels (e.g., A --> B-Label-)
      - Enforce separation between statements on different lines
      - ***New Fix: Ensures only a single 'graph TD' header exists.***
    """
    import re
    if not raw:
        return default_chart

    s = raw

    # 1. Replace parentheses (and fullwidth variants) with hyphen to obey your rule
    s = s.replace("（", "-").replace("）", "-").replace("(", "-").replace(")", "-")

    # 2. Normalize arrow spacing
    s = re.sub(r"\s*(-->|---|-\.?->)\s*", r" \1 ", s)

    # 3. Fix the critical syntax error: A --> B-Label- or A -- text --> B-Label-
    def fix_label_syntax(match):
        full_match = match.group(0)
        link = match.group(1)
        target_id = match.group(2)
        label_text = match.group(3).strip("-") # remove leading/trailing hyphens from the label part
        return f"{link}{target_id}[{label_text}]"

    # Regex to find links ending in an identifier immediately followed by a hyphenated phrase
    s = re.sub(r"((?:--|--.+?--|-\.?->)\s*)\b([A-Z]+[0-9]*[-_]*)([-].+?[-])(?=\s*\n|\s*\Z)", fix_label_syntax, s, flags=re.I)
    
    # 4. Put structural keywords on their own lines (safe)
    s = re.sub(r"\s*%%\s*", r"\n%% ", s)
    # Convert all graph headers to just a newline for now, we'll re-insert the single valid one later.
    s = re.sub(r"(?i)\b(graph\s+(?:TD|LR|TB|BT|RL))\b", r"\n", s)
    s = re.sub(r"(?i)\bsubgraph\b", r"\nsubgraph", s)
    s = re.sub(r"(?i)\bend\b", r"\nend\n", s)
    s = re.sub(r"(?i)\bdirection\b\s*\w+", lambda m: "\n" + m.group(0), s)

    # 5. Split statements jammed on one line
    s = re.sub(r"\]\s+(?=[A-Za-z][A-Za-z0-9_\-]*\s*\[)", "]\n", s)
    s = re.sub(r"\]\s+(?=[A-Za-z][A-Za-z0-9_\-]*\s*(-->|---|-.->))", "]\n", s)
    s = re.sub(r"([A-Za-z0-9_\-]+)\s+(?=[A-Za-z][A-Za-z0-9_\-]*\s*(-->|---|-.->))", r"\1\n", s)
    
    # 6. Collapse runs of blank lines, strip trailing spaces per line
    lines = [ln.rstrip() for ln in s.splitlines()]
    
    # --- CRITICAL FIX: Ensure only one graph header ---
    # First, strip all remaining empty lines and comments from the beginning
    while lines and (lines[0].strip() == "" or lines[0].strip().startswith("%%")):
        lines.pop(0)

    # Then, insert the definitive graph header at the start
    if lines:
        lines.insert(0, default_chart)
    else:
        return default_chart # Return only the chart type if file is empty
    # --- END CRITICAL FIX ---
    
    cleaned = []
    prev_empty = False
    for ln in lines:
        if ln.strip() == "":
            if not prev_empty:
                cleaned.append("") 
            prev_empty = True
        else:
            # Ensure links are on one line
            if cleaned and cleaned[-1].endswith("-->") and ln.startswith("-->"):
                cleaned[-1] = cleaned[-1] + ln.lstrip() # Merge broken arrow parts
            else:
                cleaned.append(ln)
            prev_empty = False

    # 7. Final tidy
    final = "\n".join(cleaned)
    final = re.sub(r"[ \t]{2,}", " ", final)
    final = re.sub(r"-{3,}", "---", final)
    final = re.sub(r"\n{3,}", "\n\n", final)

    return final.strip()


def prepare_mermaid_for_resolver(raw_mermaid: str) -> str:
    """
    Small wrapper to run normalize_and_clean_mermaid and ensure minimal safe formatting
    before applying resolve_bracket_only_labels.
    """
    # Use the enhanced normalize_and_clean_mermaid
    cleaned = normalize_and_clean_mermaid(raw_mermaid, default_chart="graph TD")
    # If there are lines that look like an isolated label enclosed in brackets but not alone,
    # ensure they are on their own lines (only when safe). This uses a narrow pattern to avoid breaking tokens.
    import re
    cleaned = re.sub(r"(?<=\n)([^\n]*\])\s+(?=\[)", r"\1\n", cleaned)   # close bracket followed by a bracketed label -> newline
    return cleaned



def slugify_label(label: str) -> str:
    # create a simple safe id from label (letters numbers underscores hyphens)
    s = label.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        s = "node"
    # ensure it doesn't start with a digit (mermaid still accepts but safer)
    if re.match(r"^\d", s):
        s = "n-" + s
    return s

def resolve_bracket_only_labels(mermaid_text: str) -> str:
    """
    Convert standalone [Label] lines into ID[label], assigning them to the
    previous link target if that target has no definition yet, otherwise
    create a new unique id from the label.
    (Kept for compatibility, though the enhanced normalize function may make it less necessary)
    """
    lines = mermaid_text.splitlines()
    defined_ids = set()
    
    # Regex patterns (must be re-compiled here if imported globally)
    node_def_re = re.compile(r"^([A-Za-z0-9_\-]+)\s*\[")
    link_re = re.compile(r"^([A-Za-z0-9_\-]+)\s*(-->|---|-.->)\s*([A-Za-z0-9_\-]+)\s*$")
    bracket_only_re = re.compile(r"^\[([^\]]+)\]\s*$")
    subgraph_re = re.compile(r"^\s*subgraph\b", re.I)
    end_re = re.compile(r"^\s*end\s*$", re.I)
    chart_re = re.compile(r"^\s*graph\b", re.I)

    # first pass find defined ids
    for ln in lines:
        m = node_def_re.match(ln.strip())
        if m:
            defined_ids.add(m.group(1))

    out_lines = []
    last_link_target = None
    # generator for unique ids
    generated = {}
    counter = 1

    for i, ln in enumerate(lines):
        stripped = ln.strip()

        # keep chart/subgraph/end lines as-is
        if subgraph_re.match(stripped) or end_re.match(stripped) or chart_re.match(stripped) or stripped.startswith("%%"):
            out_lines.append(ln)
            last_link_target = None
            continue

        # track link lines
        mlink = link_re.match(stripped)
        if mlink:
            src, arrow, tgt = mlink.groups()
            out_lines.append(stripped)
            last_link_target = tgt
            # if source or target are node-defs present later, they will be resolved; keep set
            continue

        # track node defs
        mnode = node_def_re.match(stripped)
        if mnode:
            defined_ids.add(mnode.group(1))
            out_lines.append(stripped)
            last_link_target = None
            continue

        # bracket only
        mbr = bracket_only_re.match(stripped)
        if mbr:
            label = mbr.group(1).strip()
            assigned_id = None
            # if last link target exists and not defined yet, use it
            if last_link_target and (last_link_target not in defined_ids):
                assigned_id = last_link_target
            else:
                # create slug-based id; ensure uniqueness
                base = slugify_label(label)
                candidate = base
                while candidate in defined_ids or candidate in generated.values():
                    candidate = f"{base}-{counter}"
                    counter += 1
                assigned_id = candidate
                generated[label] = assigned_id

            # mark assigned id as defined
            defined_ids.add(assigned_id)
            # append the definition line
            out_lines.append(f"{assigned_id}[{label}]")
            last_link_target = None
            continue

        # otherwise leave line as-is
        out_lines.append(stripped)
        last_link_target = None

    # return with original newline style preserved (use \n)
    return "\n".join(out_lines)


def ensure_clean_mermaid_and_save(mmd_path: Path, raw_mermaid_text: str, default_chart: str = "graph TD"):
    """
    Normalize raw mermaid output, save to mmd_path, and print a short preview for debugging.
    """
    cleaned = normalize_and_clean_mermaid(raw_mermaid_text, default_chart=default_chart)
    # Ensure directory exists
    parent = mmd_path.parent
    if parent.exists() and not parent.is_dir():
        # remove file that blocks directory creation
        try:
            parent.unlink()
        except Exception:
            pass
    parent.mkdir(parents=True, exist_ok=True)
    mmd_path.write_text(cleaned, encoding="utf-8")
    print(f"Saved sanitized mermaid to {mmd_path}")
    print("Sanitized preview (first 1000 chars):")
    print(cleaned[:1000])


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_mermaid_to_png(mmd_path: Path, png_path: Path, timeout: int = 120):
    # Debug prints for easier diagnosis
    print(f"DEBUG: Rendering {mmd_path} to {png_path}")
    if not mmd_path.exists():
        raise FileNotFoundError(f"Mermaid file not found: {mmd_path}")

    content = mmd_path.read_text(encoding="utf-8")
    print(f"DEBUG: mmd length {len(content)} chars")
    print("DEBUG: mmd preview:\n", content[:800])

    # Attempt local mmdc commands
    tried = []
    explicit = Path(MMDC_CMD)
    cmds = []
    if explicit.exists():
        cmds.append([str(explicit), "-i", str(mmd_path), "-o", str(png_path), "--scale", str(MMDC_SCALE)])
    # check PATH
    mmdc_path = shutil.which("mmdc")
    if mmdc_path:
        cmds.append([mmdc_path, "-i", str(mmd_path), "-o", str(png_path), "--scale", str(MMDC_SCALE)])
    mmdc_cmd_path = shutil.which("mmdc.cmd")
    if mmdc_cmd_path:
        cmds.append([mmdc_cmd_path, "-i", str(mmd_path), "-o", str(png_path), "--scale", str(MMDC_SCALE)])

    for c in cmds:
        try:
            print("DEBUG: Running local command:", " ".join(c))
            proc = subprocess.run(c, capture_output=True, text=True, timeout=timeout)
            print("DEBUG mmdc stdout:", proc.stdout)
            print("DEBUG mmdc stderr:", proc.stderr)
            if proc.returncode == 0 and png_path.exists():
                print("DEBUG: Successfully rendered PNG with local mmdc")
                return
        except Exception as e:
            print("DEBUG: mmdc run error:", e)

    # Docker fallback
    docker_exec = shutil.which("docker")
    if docker_exec:
        folder = str(mmd_path.parent.resolve())
        container_input = f"/data/{mmd_path.name}"
        container_output = f"/data/{png_path.name}"
        docker_cmd = [
            docker_exec, "run", "--rm", "-v", f"{folder}:/data",
            "minlag/mermaid-cli", "mmdc", "-i", container_input, "-o", container_output, "--scale", str(MMDC_SCALE)
        ]
        try:
            print("DEBUG: Trying Docker fallback:", " ".join(docker_cmd))
            proc = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=300)
            print("DEBUG docker stdout:", proc.stdout)
            print("DEBUG docker stderr:", proc.stderr)
            if proc.returncode == 0 and png_path.exists():
                print("DEBUG: Successfully rendered PNG with Docker fallback")
                return
        except Exception as e:
            print("DEBUG: Docker fallback error:", e)

    raise RuntimeError("Rendering failed: neither local mmdc nor Docker produced output. See debug logs above.")


def png_to_jpg(png_path: Path, jpg_path: Path, quality: int = 100): # Changed default to 100
    img = Image.open(png_path)
    rgb = img.convert("RGB")
    # You can also explicitly pass 100 here if you don't want to change the function signature default
    rgb.save(jpg_path, "JPEG", quality=quality)

# ---------------------------
# NEW QA helper functions
# ---------------------------

def build_qa_prompt(process_name: str, gemini_response_text: str) -> str:
    """Construct the QA prompt by inserting the process name and embedding the full
    Gemini response inside triple backticks so the model only uses that content.
    """
    header = QA_PROMPT_TEMPLATE.replace("<PROCESS NAME>", process_name)
    # Embed the gemini response in triple backticks. Use a safe escape if necessary.
    return f"{header}\n\n```\n{gemini_response_text}\n```\n"


def parse_numbered_qa(raw: str) -> Dict[str, str]:
    """Attempt a robust extraction of numbered answers from the QA response.
    Returns a dict mapping stringified question numbers ("1", "2", ...) to answer text.
    If parsing fails, returns {"raw": raw}.
    """
    # Look for patterns like "1.", "1)", "Q1:", or explicit question text followed by an answer.
    answers: Dict[str, str] = {}
    # Normalize newlines
    text = raw.replace('\r\n', '\n')

    # First, try to split by numbered items like "1." at line starts
    parts = re.split(r"^\s*(\d+)[\.)]\s*", text, flags=re.M)
    # re.split will produce leading text then pairs of (num, content)
    if len(parts) >= 3:
        # parts[0] is leading, then num, content, num, content...
        it = iter(parts[1:])
        for num, content in zip(it, it):
            num = num.strip()
            answers[num] = content.strip()
    else:
        # fallback: try to find "1." markers anywhere
        m = re.findall(r"(\d+)[\.)]\s*(.+?)(?=(?:\n\d+[\.)]\s)|\Z)", text, flags=re.S)
        if m:
            for num, ans in m:
                answers[num.strip()] = ans.strip()

    if not answers:
        # As last resort, include raw text
        return {"raw": raw}

    return answers


def generate_qa_for_process(process_name: str, gemini_response_text: str, proc_dir: Path) -> Dict[str, Any]:
    """Call Gemini with the QA prompt and save outputs. Returns a small dict with paths and parsed answers.
    """
    prompt = build_qa_prompt(process_name, gemini_response_text)
    qa_raw = call_gemini_chat(prompt)
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    qa_txt_path = proc_dir / "qa_response.txt"
    save_text(qa_txt_path, qa_raw)

    parsed = parse_numbered_qa(qa_raw)
    qa_json_path = proc_dir / "qa.json"
    save_text(qa_json_path, json.dumps(parsed, indent=2))

    print(f"Saved QA raw response to {qa_txt_path}")
    print(f"Saved parsed QA to {qa_json_path}")

    return {
        "qa_text": str(qa_txt_path.resolve()),
        "qa_json": str(qa_json_path.resolve()),
        "parsed": parsed
    }

# ---------------------------
# Main pipeline (with QA)
# ---------------------------

def generate_for_process(process_name: str, output_root: Path):
    print(f"\n=== Generating flowchart for: {process_name} ===")
    prompt = PROMPT_TEMPLATE.replace("<PROCESS NAME>", process_name)

    raw_text = call_gemini_chat(prompt)
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    proc_dir = output_root / re.sub(r"[^\w\-]+", "_", process_name).strip("_")
    proc_dir.mkdir(parents=True, exist_ok=True)
    raw_txt_path = proc_dir / "gemini_response.txt"
    save_text(raw_txt_path, raw_text)
    print("Saved Gemini response to", raw_txt_path)
    print("DEBUG: Gemini response length", len(raw_text))
    print("DEBUG: Gemini response preview:\n", raw_text[:1200])

    # NEW: run QA pass on the full gemini_response
    try:
        qa_info = generate_qa_for_process(process_name, raw_text, proc_dir)
    except Exception as e:
        print("QA generation failed:", e)
        qa_info = {"error": str(e)}

    mermaid = extract_mermaid_code(raw_text)
    if not mermaid:
        print("No fenced mermaid block found, attempting heuristic extraction of last section.")
        tail = raw_text[-4000:]
        m = re.search(r"(graph\s+(?:TD|LR|TB|BT|RL).*)", tail, re.S | re.I)
        if m:
            mermaid = m.group(1).strip()

    if not mermaid:
        raise RuntimeError(f"Could not extract mermaid code for {process_name}")

    # 1. Normalize and fix common syntax errors (e.g., A --> B-Label- to A --> B[Label])
    mermaid_sanitized = prepare_mermaid_for_resolver(mermaid)
    print("DEBUG: Sanitized mermaid preview:\n", mermaid_sanitized[:1000])

    # 2. Resolve bracket-only labels safely (if any are left)
    mermaid_final = resolve_bracket_only_labels(mermaid_sanitized)
    print("DEBUG: Final mermaid after resolving bracket-only labels preview:\n", mermaid_final[:1200])

    # 3. Save final cleaned mermaid to file
    mmd_path = proc_dir / "diagram.mmd"
    # Call again to ensure final formatting is applied, especially for the H[] node and empty lines
    ensure_clean_mermaid_and_save(mmd_path, mermaid_final, default_chart="graph TD")


    print("Saved sanitized mermaid to", mmd_path)

    # 4) Render to PNG
    png_path = proc_dir / "diagram.png"
    try:
        render_mermaid_to_png(mmd_path, png_path)
    except Exception as e:
        print("mmdc rendering failed:", e)
        raise

    # 5) Convert to JPG
    jpg_path = proc_dir / "diagram.jpg"
    png_to_jpg(png_path, jpg_path)
    print("Saved JPG to", jpg_path)

    result = {
        "process_name": process_name,
        "dir": str(proc_dir.resolve()),
        "mermaid_file": str(mmd_path.resolve()),
        "png_file": str(png_path.resolve()),
        "jpg_file": str(jpg_path.resolve()),
        "gpt_response": str(raw_txt_path.resolve()),
        "qa_text": qa_info.get("qa_text") if isinstance(qa_info, dict) else None,
        "qa_json": qa_info.get("qa_json") if isinstance(qa_info, dict) else None,
    }

    return result


def main():
    results = []
    for pname in tqdm(PROCESS_NAMES, desc="Processes"):
        try:
            res = generate_for_process(pname, OUTPUT_ROOT)
            results.append(res)
        except Exception as e:
            print(f"Failed for {pname}: {e}")

    summary_path = OUTPUT_ROOT / "summary.json"
    save_text(summary_path, json.dumps(results, indent=2))
    print("\nAll done. Summary saved to", summary_path)

if __name__ == "__main__":
    main()
