"""The chat-review page's stylesheet and font link.

Kept apart from `chat_review` so that module stays readable. Every
read of every chat agent renders with these same rules, as the same
document. See the project rule
`chat-agent-validation.md`.

**Colour on this page means exactly one thing: what kind of finding
a note is.** Structure -- the rule above the title, the caveat's
edge, the pressed toggle, focus rings -- is ink and slate, and the
two speakers are told apart by depth rather than by hue: the agent's
bubble sits on its own surface with an edge and a shadow, the
counterpart's is recessed into the page. That leaves the four note
colours as the only chroma, so a reader learns them once and can
then find a defect by scanning rather than by reading.

Reworked 6th September 2026, at the operator's request: the first
palette put a warm cream bubble and a brick-red note block on a cool
slate ground, and the temperatures fought.
"""

FONT_LINK = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'


#: The dark palette, written once. CSS cannot share one declaration
#: block between a media-conditional selector and a plain one, so the
#: page needs the same tokens twice -- once for a viewer on the system
#: default, once for a viewer who chose dark. Keeping two hand-written
#: copies in step failed twice in one afternoon, in both directions,
#: and the second time shipped white bubbles with invisible text to
#: everyone whose system is dark. So they are generated from here.
_DARK: dict[str, str] = {
    "--ground": "#1c222c",
    "--surface": "#242b36",
    "--surface-2": "#2c3440",
    "--line": "#3a4452",
    "--line-soft": "#313945",
    "--text": "#ecf0f5",
    "--text-2": "#b4bdcb",
    "--text-3": "#8794a6",
    "--mark": "#ecf0f5",
    "--mark-ink": "#1c222c",
    "--bubble-agent": "#303945",
    "--bubble-other": "#232a34",
    "--worked": "#66cca0",
    "--worked-wash": "#1d3830",
    "--call": "#8eb6f4",
    "--call-wash": "#1f2d41",
    # A bright, saturated red rather than the rose this was through
    # several attempts. The block stays dark and low-chroma so the
    # label and the left edge carry the colour -- a bright red over
    # a strong red block glares, and the note's prose has to stay
    # readable on it.
    "--defect": "#ff5d55",
    "--defect-wash": "#2d2027",
    "--guard": "#b4a5f4",
    "--guard-wash": "#2d2545",
    "--shadow": (
        "0 1px 2px rgba(0, 0, 0, .38), 0 8px 24px rgba(0, 0, 0, .25)"
    ),
}


def _dark_block(indent: str) -> str:
    """The dark tokens as CSS declarations at one indent."""
    return "\n".join(
        f"{indent}{name}: {value};" for name, value in _DARK.items()
    )


_CSS = """  /* ---- Light palette: the complete set lives here. ---- */
  :root {
    --ground: #eef1f6;
    --surface: #ffffff;
    --surface-2: #e2e7f0;
    --line: #d1d9e4;
    --line-soft: #e4e9f1;
    --text: #111721;
    --text-2: #4a5567;
    --text-3: #76829a;

    /* Structure is ink, never a hue. See the note above. */
    --mark: #111721;
    --mark-ink: #ffffff;

    /* The agent's bubble is always the lighter of the two against
       the ground, in both themes, so the side under review reads as
       the raised one. Separate from `--surface` because a card and
       a bubble want opposite treatment when the page goes dark. */
    --bubble-agent: #ffffff;
    --bubble-other: #e2e7f0;

    --worked: #176b4b;
    --worked-wash: #e2f0e9;
    --call: #2a5cae;
    --call-wash: #e4ecf9;
    --defect: #a63049;
    --defect-wash: #f9e6eb;
    --guard: #5f45b6;
    --guard-wash: #eae5f8;

    --shadow: 0 1px 2px rgba(17, 23, 33, .05),
              0 8px 24px rgba(17, 23, 33, .05);
    --radius: 14px;
  }

  /* ---- Dark, for the un-stamped (system) state. Tokens only. ---- */
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
/* DARK-6 */
    }
  }

  /* ---- Dark, when explicitly chosen. The same tokens, written
     from the same source -- see `_DARK` below. ---- */
  :root[data-theme="dark"] {
/* DARK-4 */
  }

  * { box-sizing: border-box; }

  body {
    background: var(--ground);
    color: var(--text);
    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }

  .wrap {
    max-width: 940px;
    margin: 0 auto;
    padding: 0 20px 96px;
  }

  /* ---------- masthead ---------- */

  .mast {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 56px 0 28px;
  }

  .eyebrow {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--text-3);
  }

  .eyebrow::before {
    content: "";
    width: 26px;
    height: 2px;
    background: var(--mark);
  }

  h1 {
    font-family: Newsreader, Georgia, serif;
    font-weight: 600;
    font-size: clamp(2.4rem, 6vw, 3.5rem);
    line-height: 1.04;
    letter-spacing: -.015em;
    margin: 0;
    text-wrap: balance;
  }

  .standfirst {
    font-family: Newsreader, Georgia, serif;
    font-size: 1.2rem;
    line-height: 1.6;
    color: var(--text-2);
    max-width: 63ch;
    margin: 0;
  }

  .caveat {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--mark);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: .95rem;
    color: var(--text-2);
    max-width: 68ch;
  }

  .caveat strong { color: var(--text); }

  /* ---------- verdicts ---------- */

  h2 {
    font-family: Newsreader, Georgia, serif;
    font-weight: 600;
    font-size: 1.95rem;
    letter-spacing: -.01em;
    margin: 0 0 4px;
    text-wrap: balance;
  }

  .section { padding-top: 52px; }

  .section-note {
    color: var(--text-2);
    max-width: 65ch;
    margin: 0 0 20px;
  }

  .verdicts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  .verdict {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 20px 22px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    box-shadow: var(--shadow);
  }

  .verdict-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .verdict h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .verdict p {
    margin: 0;
    font-size: .94rem;
    color: var(--text-2);
  }

  .stamp {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .09em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 999px;
    white-space: nowrap;
  }

  .stamp.pass { background: var(--worked-wash); color: var(--worked); }
  .stamp.fail { background: var(--defect-wash); color: var(--defect); }
  .stamp.new  { background: var(--guard-wash);  color: var(--guard); }

  /* ---------- numbers ---------- */

  .figures {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 22px;
  }

  .fig {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 16px;
  }

  .fig .n {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.7rem;
    font-weight: 600;
    letter-spacing: -.02em;
    display: block;
    line-height: 1.1;
  }

  .fig .k {
    font-size: .82rem;
    color: var(--text-3);
    display: block;
    margin-top: 4px;
  }

  .fig .d {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: .78rem;
    color: var(--text-3);
    display: block;
    margin-top: 2px;
  }

  .scroller { overflow-x: auto; }

  table {
    border-collapse: collapse;
    width: 100%;
    font-size: .93rem;
    min-width: 460px;
  }

  th, td {
    text-align: left;
    padding: 9px 14px 9px 0;
    border-bottom: 1px solid var(--line-soft);
  }

  th {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: var(--text-3);
    font-weight: 500;
  }

  td.num, th.num {
    text-align: right;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    padding-right: 0;
  }

  /* ---------- sticky control ---------- */

  .control {
    position: sticky;
    top: 0;
    z-index: 20;
    background: color-mix(in srgb, var(--ground) 88%, transparent);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--line);
    margin: 46px -20px 0;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
  }

  /* ---------- tabs ---------- */

  /* The report is long enough that reading it as one scroll buries
     the conversations under the apparatus. Each section is a tab,
     and Read mode drops every tab but the conversations, so a first
     pass is the transcripts and nothing else. */
  .tabs {
    display: flex;
    gap: 3px;
    overflow-x: auto;
    scrollbar-width: none;
    flex: 1 1 auto;
    min-width: 0;
  }

  .tabs::-webkit-scrollbar { display: none; }

  .tabs button {
    font: inherit;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    letter-spacing: .07em;
    text-transform: uppercase;
    white-space: nowrap;
    border: 0;
    border-bottom: 2px solid transparent;
    background: transparent;
    color: var(--text-3);
    padding: 7px 11px 6px;
    cursor: pointer;
    transition: color .15s, border-color .15s;
  }

  .tabs button:hover { color: var(--text-2); }

  .tabs button[aria-selected="true"] {
    color: var(--text);
    border-bottom-color: var(--mark);
    font-weight: 600;
  }

  .tabs button:focus-visible {
    outline: 2px solid var(--mark);
    outline-offset: -2px;
  }

  .panel[hidden] { display: none; }

  body[data-mode="read"] .tabs button[data-review="1"] { display: none; }

  .switch {
    display: inline-flex;
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 3px;
    gap: 2px;
  }

  .switch button {
    font: inherit;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
    border: 0;
    background: transparent;
    color: var(--text-2);
    padding: 6px 15px;
    border-radius: 999px;
    cursor: pointer;
    transition: background .15s, color .15s;
  }

  /* The pressed button is a solid block of ink. `--mark-ink` flips
     with the theme, so the label reads on it either way -- the old
     rule hardcoded one colour and then overrode itself. */
  .switch button[aria-pressed="true"] {
    background: var(--mark);
    color: var(--mark-ink);
    font-weight: 600;
  }

  .switch button:focus-visible {
    outline: 2px solid var(--mark);
    outline-offset: 2px;
  }

  .legend {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    font-size: .82rem;
    color: var(--text-2);
  }

  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .legend i {
    width: 9px;
    height: 9px;
    border-radius: 3px;
    display: inline-block;
  }

  body[data-mode="read"] .legend { display: none; }

  /* ---------- conversations ---------- */

  .conv { padding-top: 44px; }

  .conv-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    border-bottom: 2px solid var(--text);
    padding-bottom: 8px;
    margin-bottom: 6px;
  }

  .conv-head h2 { margin: 0; }

  .conv-head .tag {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--text-3);
  }

  .thread {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 26px;
  }

  .turn {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .turn + .turn { margin-top: 14px; }

  .row { display: flex; }
  .row.mine { justify-content: flex-end; }

  .bubble {
    max-width: min(76%, 46ch);
    padding: 9px 14px;
    border-radius: 16px;
    font-size: .98rem;
    line-height: 1.5;
    overflow-wrap: anywhere;
  }

  /* The agent's side sits on top: its own surface, an edge and a
     shadow. The counterpart's is recessed into the page with
     neither. The two sides are told apart by depth rather than by
     colour, which leaves every hue on the page meaning one thing
     -- what kind of finding a note is. */
  .row.his .bubble {
    background: var(--bubble-agent);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    color: var(--text);
    border-bottom-left-radius: 5px;
  }

  .row.mine .bubble {
    background: var(--bubble-other);
    border: 1px solid var(--line-soft);
    color: var(--text-2);
    border-bottom-right-radius: 5px;
  }

  .turnno {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 10.5px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 8px 0 1px;
  }

  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
    color: var(--text-3);
    margin: 5px 0 0 2px;
  }

  .meta b {
    font-weight: 500;
    color: var(--text-2);
  }

  .meta .hot { color: var(--defect); font-weight: 600; }

  /* The kind is carried by the label and the left edge. The note's
     own prose stays in the page's text colour: a paragraph set in
     red is harder to read than one in ink, and a column of them
     turns a page of findings into a warning sign. */
  .note {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    font-size: .88rem;
    line-height: 1.5;
    padding: 10px 14px;
    border-radius: 4px;
    border-left: 3px solid currentColor;
    margin-top: 6px;
    max-width: 72ch;
  }

  .note code {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: .86em;
    background: color-mix(in srgb, currentColor 11%, transparent);
    padding: 1px 4px;
    border-radius: 4px;
  }

  .note .lbl {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .09em;
    text-transform: uppercase;
    flex: 0 0 auto;
    padding-top: 3px;
    width: 66px;
  }

  .note.worked { background: var(--worked-wash); color: var(--worked); }
  .note.call   { background: var(--call-wash);   color: var(--call); }
  .note.defect { background: var(--defect-wash); color: var(--defect); }
  .note.guard  { background: var(--guard-wash);  color: var(--guard); }

  .note p {
    margin: 0;
    color: var(--text);
  }

  .note p strong { font-weight: 600; }

  .note p em {
    color: var(--text-2);
    font-style: italic;
  }

  body[data-mode="read"] .meta,
  body[data-mode="read"] .note,
  body[data-mode="read"] .turnno,
  body[data-mode="read"] .figures { display: none; }

  body[data-mode="read"] .turn + .turn { margin-top: 10px; }

  /* ---------- guard detail ---------- */

  details.attempts {
    margin: 8px 0 0 2px;
    max-width: 72ch;
    border: 1px solid var(--guard);
    border-radius: 9px;
    overflow: hidden;
  }

  details.attempts summary {
    cursor: pointer;
    padding: 8px 13px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11.5px;
    letter-spacing: .05em;
    text-transform: uppercase;
    color: var(--guard);
    background: var(--guard-wash);
  }

  details.attempts summary:focus-visible {
    outline: 2px solid var(--guard);
    outline-offset: -2px;
  }

  details.attempts ol {
    margin: 0;
    padding: 10px 13px 12px 30px;
    font-size: .87rem;
    color: var(--text-2);
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  details.attempts .rej {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 10px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--defect);
  }

  details.attempts .ok {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 10px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--worked);
  }

  body[data-mode="read"] details.attempts { display: none; }

  /* ---------- closing ---------- */

  .closing {
    margin-top: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .closing p {
    margin: 0;
    max-width: 65ch;
    color: var(--text-2);
  }

  .closing p strong { color: var(--text); }

  .pricing {
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid var(--line);
    font-size: .84rem;
    color: var(--text-3);
    max-width: 72ch;
  }

  .pricing code {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
  }

  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
  }

  @media (max-width: 640px) {
    .bubble { max-width: 84%; }
    .note .lbl { width: 100%; padding-top: 0; }
    .note { flex-direction: column; gap: 4px; }
  }"""


REVIEW_CSS = _CSS.replace("/* DARK-6 */", _dark_block("      ")).replace(
    "/* DARK-4 */", _dark_block("    ")
)


__all__ = ["FONT_LINK", "REVIEW_CSS"]
