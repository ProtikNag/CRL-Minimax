#!/usr/bin/env python3
"""Stop-hook verification gate (fail-open).

Blocks the main agent from finishing a turn while there is UNVERIFIED gated work,
signalled by `.claude/gate-status.json`. The main agent (per ~/.claude/CLAUDE.md)
sets `{"gate": "pending", "reason": ...}` when it starts gated work (a
visualization / a paper-implementation / user's math) and sets `{"gate":"clear"}`
after the relevant verifier subagent returns PASS.

Safety:
- FAIL-OPEN: any missing/unreadable status file, or any hook error, allows the
  stop. A broken gate must never brick a session.
- LOOP-ESCAPE: after MAX consecutive blocks it allows the stop (with a loud
  warning) so the session can never be trapped.
- `{"override": true}` in the status file bypasses the gate.
"""
import sys, os, json

MAX_BLOCKS = 4

def allow():
    sys.exit(0)

def main():
    try:
        raw = sys.stdin.read()
        inp = json.loads(raw) if raw.strip() else {}
    except Exception:
        allow()  # fail-open
    proj = os.environ.get("CLAUDE_PROJECT_DIR") or inp.get("cwd") or os.getcwd()
    sf = os.path.join(proj, ".claude", "gate-status.json")
    if not os.path.exists(sf):
        allow()
    try:
        st = json.load(open(sf))
    except Exception:
        allow()  # unreadable -> fail-open
    if not isinstance(st, dict) or st.get("gate") != "pending" or st.get("override"):
        allow()

    blocks = int(st.get("blocks", 0))
    # Loop-escape: if we've already blocked repeatedly this turn-chain, let go.
    if inp.get("stop_hook_active") and blocks >= MAX_BLOCKS:
        st["gate"] = "escaped"
        try:
            json.dump(st, open(sf, "w"))
        except Exception:
            pass
        sys.stderr.write(
            "[verification-gate] ESCAPE after %d blocks: allowing stop, but the "
            "gated work remains UNVERIFIED. Tell the user explicitly.\n" % blocks)
        sys.exit(0)

    st["blocks"] = blocks + 1
    try:
        json.dump(st, open(sf, "w"))
    except Exception:
        pass
    reason = st.get("reason", "gated work is pending verification")
    sys.stderr.write(
        "[verification-gate] BLOCKED: %s\n"
        "Do NOT finish yet. Delegate to the relevant verifier subagent "
        "(visualization-expert / code-verifier / math-verifier), act on its "
        "findings, and once it returns a PASS verdict write "
        "`.claude/gate-status.json` = {\"gate\": \"clear\"}. Then finish.\n" % reason)
    sys.exit(2)  # exit code 2 blocks the Stop and feeds stderr back to the agent

if __name__ == "__main__":
    main()
