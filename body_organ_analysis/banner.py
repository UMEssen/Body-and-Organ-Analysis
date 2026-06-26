import os
import sys
from typing import TextIO

from body_organ_analysis import __version__


def _supports_color(stream: TextIO) -> bool:
    """Return whether ANSI color should be written to ``stream``."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return stream.isatty()


RGB = tuple[int, int, int]

_GRADIENT_STOPS: tuple[RGB, ...] = (
    (194, 241, 106),  # #C2F16A
    (90, 225, 186),  # #5AE1BA
    (178, 172, 246),  # #B2ACF6
    (255, 140, 140),  # #FF8C8C
)


def _gradient_at(t: float, stops: tuple[RGB, ...]) -> RGB:
    """Sample a multi-stop gradient at position ``t`` in [0, 1]."""
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    seg = t * (len(stops) - 1)
    i = int(seg)
    frac = seg - i
    r0, g0, b0 = stops[i]
    r1, g1, b1 = stops[i + 1]
    return (
        round(r0 + (r1 - r0) * frac),
        round(g0 + (g1 - g0) * frac),
        round(b0 + (b1 - b0) * frac),
    )


def _colorize(line: str, width: int) -> str:
    """Wrap ``line`` so each column gets its smooth horizontal gradient color."""
    last = max(width - 1, 1)
    body = "".join(
        f"\x1b[38;2;{r};{g};{b}m{char}"
        for col, char in enumerate(line)
        for r, g, b in (_gradient_at(col / last, _GRADIENT_STOPS),)
    )
    return "\x1b[1m" + body + "\x1b[0m"


def render_banner(color: bool, margin_left: int = 3) -> str:
    """Return the BOA banner, with a smooth Gemini-style gradient when colored."""
    spaces_left = margin_left * " "
    doi_url = "https://doi.org/10.1097/RLI.0000000000001040"
    boa_art = (
        "██████╗  ██████╗  █████╗ ",
        "██╔══██╗██╔═══██╗██╔══██╗",
        "██████╔╝██║   ██║███████║",
        "██╔══██╗██║   ██║██╔══██║",
        "██████╔╝╚██████╔╝██║  ██║",
        "╚═════╝  ╚═════╝ ╚═╝  ╚═╝",
    )
    ikim_art = (
        "███╗███╗ ████╗███╗ ███████╗    ███████╗",
        "███║███║███╔═╝███║ ███╔████╗  ████╔███║",
        "███║███████║  ███║ ███║╚███║  ███╔╝███║",
        "███║███╔███║  ███║ ███║ ████╗████║ ███║",
        "███║███║╚████╗███║ ███║ ╚███████╔╝ ███║",
        "╚══╝╚══╝ ╚═══╝╚══╝ ╚══╝  ╚══════╝  ╚══╝",
    )
    # V1
    # ship_art = (
    #     "     ▒▒▒▒▓▓▓▓╗     ",
    #     "     ▒▒▒▒▓▓▓▓║     ",
    #     "█████▓▓▓▓▒▒▒▒█████╗",
    #     "╚████▓▓▓▓▒▒▒▒████╔╝",
    #     " ╚════██████╔════╝ ",
    #     "      ╚═════╝      ",
    # )
    # V2
    ship_art = (
        "     ▆▆▆▖▆▆▆▖╗     ",
        "     ███▌███▌║     ",
        "████▌▆▆▆▖▆▆▆▖████▌╗",
        "╚███▌███▌███▌███▌╔╝",
        " ╚════▆▆▆▆▆▆╔════╝ ",
        "      ╚═════╝      ",
    )
    ascii_art = tuple(
        f"{spaces_left}{a}{spaces_left}{b}{spaces_left}{c}"
        for a, b, c in zip(boa_art, ship_art, ikim_art, strict=True)
    )
    subtitle = f"{spaces_left}BOA-MR  |  v{__version__}"
    cite = f"{spaces_left}If you use this tool please cite: {doi_url}"
    if not color:
        return "\n".join(["", *ascii_art, "", subtitle, cite, ""])

    width = max(len(line) for line in ascii_art)
    art = [_colorize(line, width) for line in ascii_art]
    r, g, b = _gradient_at(0.5, _GRADIENT_STOPS)
    subtitle = f"\x1b[38;2;{r};{g};{b}m{subtitle}\x1b[0m"
    cite = f"\x1b[2m{cite}\x1b[0m"
    return "\n".join(["", *art, "", subtitle, cite, "\n\n"])


def print_banner() -> None:
    """Print the BOA startup banner to stdout."""
    sys.stdout.write(render_banner(color=_supports_color(sys.stdout)))
    sys.stdout.flush()


if __name__ == "__main__":
    print_banner()
