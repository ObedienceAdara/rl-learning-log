from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

# Robot2 Sprite sheet by davilj on OpenGameArt.org.
# License: GPL-2.0.
URL = "https://opengameart.org/sites/default/files/spritesheet_53.png"
OUT = ASSET_DIR / "robot_spritesheet.png"


def main() -> None:
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        OUT.write_bytes(response.read())
    print(f"Saved {OUT}")
    print("Source: https://opengameart.org/content/robot2-sprite-sheet")
    print("Author: davilj")
    print("License: GPL-2.0")


if __name__ == "__main__":
    main()
