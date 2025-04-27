import os, requests, zipfile, io

# 1) URLs for the CC0 ZIP
ZIP_URL = "https://opengameart.org/sites/default/files/SpaceShooterRedux.zip"

# 2) Download
resp = requests.get(ZIP_URL)
resp.raise_for_status()

# 3) Extract only the sprite PNGs
with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
    for member in z.namelist():
        # filter to top-level PNGs (e.g. Player*.png, Enemy*.png, Missile*.png, Bomb*.png)
        if member.endswith(".png") and ("Player" in member or "Enemy" in member or "Missile" in member or "Bomb" in member):
            target = os.path.join("assets", "sprites", os.path.basename(member))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(z.read(member))

print("Sprites fetched to assets/sprites/")  # end
