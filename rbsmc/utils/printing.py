RESET = "\033[0m"
BOLD  = "\033[1m"

COL = {
    "black":  "\033[30m",
    "red":    "\033[31m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "blue":   "\033[34m",
    "magenta":"\033[35m",
    "cyan":   "\033[36m",
    "white":  "\033[37m",
}

def ctext(text: str, color: str = "white", bold: bool = False) -> str:
    return f"{BOLD if bold else ''}{COL[color]}{text}{RESET}"