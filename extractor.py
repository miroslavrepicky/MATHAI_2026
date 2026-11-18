"""
Tvorba datasetu rukou písaných znakov (MNIST štýl) zo scanov / fotiek.

Vstup : scans/<znak>.jpg  (jeden súbor = jeden znak napísaný 10+ krát)
Výstup: dataset.npz  (X: [N, 784] v rozsahu 0..1, písmo ~ 1, pozadie ~ 0
                      Y: [N, 10] one-hot v poradí CHARS)
        debug/        (kontrolné obrázky: nájdené boxy + vybrané vzorky)

Postup pre každý obrázok:
  1. otočenie (ak je fotka bokom) a zmenšenie na rozumnú veľkosť
  2. maska atramentu: odčítanie pozadia (rieši nerovnomerné osvetlenie) + Otsu
  3. zlúčenie častí jedného znaku (napr. háčik a slučka deväťky) dilatáciou
     -> connected components -> bounding boxy (nie je treba deliť na rovnaké časti)
  4. každý znak: zhrubnutie ťahu, zmenšenie na 20 px, vycentrovanie podľa
     ťažiska do 28x28 (rovnako ako MNIST)
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- NASTAVENIA -----------------------------
SHOW = True                    # zobraziť okná s výsledkom extrakcie
CHARS = ["9", "H", "L", "P", "O", "T", "M", "5", "2", "Z"]                  # <- poradie z tvojej tabuľky (index = pozícia v one-hot)
SCAN_DIR = "scans"
OUT_FILE = "dataset.npz"
DEBUG_DIR = "debug"
ROTATE = None   # None, ak je fotka správne otočená
MAX_SIDE = 2000                # fotku zmenšíme, aby parametre fungovali stabilne
MERGE_FRAC = 0.02              # ako ďaleko sa spájajú časti jedného znaku
                               # (zväčši, ak sa znak rozpadá na 2 boxy;
                               #  zmenši, ak sa zlievajú susedné znaky)
MIN_REL_SIZE = 0.4             # box menší než 40 % mediánu = šum, zahodí sa
MIN_SAMPLES = 10
# ----------------------------------------------------------------------


def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if ROTATE is not None:
        img = cv2.rotate(img, ROTATE)
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def ink_mask(img):
    """Binárna maska: atrament = 255, pozadie = 0."""
    # minimum cez kanály: farebný (modrý) aj čierny atrament je tmavý, papier svetlý
    g = img.min(axis=2)
    # odhad pozadia: closing zmaže tenké tmavé ťahy, blur vyhladí
    k = odd(max(img.shape[:2]) * 0.03)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (odd(k), odd(k)), 0)
    diff = cv2.subtract(bg, g)                       # atrament = vysoká hodnota
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def find_chars(mask):
    """Vráti zoznam (x, y, w, h, label) a labels z dilatovanej masky."""
    k = odd(max(mask.shape) * MERGE_FRAC)
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    comps = [(*stats[i][:4], i) for i in range(1, n)]
    # hrubý filter drobného šumu
    floor = 0.01 * max(mask.shape)
    comps = [c for c in comps if max(c[2], c[3]) >= floor]
    if not comps:
        return [], labels
    med = np.median([max(c[2], c[3]) for c in comps])
    comps = [c for c in comps if max(c[2], c[3]) >= MIN_REL_SIZE * med]
    comps.sort(key=lambda c: c[0])                   # zľava doprava
    return comps, labels


def to_28x28(crop):
    """Orezaný znak (255 = ťah) -> 28x28 uint8 ako MNIST."""
    h, w = crop.shape
    # ťah zhrubneme úmerne veľkosti znaku, inak po zmenšení takmer zmizne
    k = odd(max(3, max(h, w) * 0.06))
    crop = np.pad(crop, k)
    crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.where(crop > 0)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape

    s = 20 / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    yo = (28 - crop.shape[0]) // 2
    xo = (28 - crop.shape[1]) // 2
    canvas[yo:yo + crop.shape[0], xo:xo + crop.shape[1]] = crop

    # vycentrovanie podľa ťažiska (MNIST)
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))
    return canvas


def process_scan(path, name):
    img = load_image(path)
    mask = ink_mask(img)
    comps, labels = find_chars(mask)

    samples = []
    for x, y, w, h, i in comps:
        region = (labels[y:y + h, x:x + w] == i) & (mask[y:y + h, x:x + w] > 0)
        if region.sum() == 0:
            continue
        samples.append(to_28x28(region.astype(np.uint8) * 255))

    # ---- debug výstupy ----
    os.makedirs(DEBUG_DIR, exist_ok=True)
    vis = img.copy()
    for x, y, w, h, _ in comps:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite(f"{DEBUG_DIR}/{name}_boxes.jpg", vis)
    cv2.imwrite(f"{DEBUG_DIR}/{name}_mask.png", mask)
    if samples:
        sheet = np.hstack([cv2.resize(s, (84, 84), interpolation=cv2.INTER_NEAREST)
                           for s in samples])
        cv2.imwrite(f"{DEBUG_DIR}/{name}_samples.png", 255 - sheet)  # tmavé písmo na bielom
    if SHOW:
        show_scan(name, vis, mask, samples)
    return samples


def show_scan(name, vis, mask, samples):
    """Okno 1: nájdené boxy + maska. Okno 2: všetky vyextrahované 28x28 vzorky."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"'{name}': nájdené znaky = {len(samples)}")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Maska atramentu")
    for a in ax:
        a.axis("off")
    fig.tight_layout()

    cols = 10
    rows = max(1, -(-len(samples) // cols))
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.3), squeeze=False)
    for k, a in enumerate(axes.ravel()):
        a.axis("off")
        if k < len(samples):
            a.imshow(samples[k], cmap="gray_r", vmin=0, vmax=255)   # písmo = tmavé
            a.set_title(str(k + 1), fontsize=8)
    fig2.suptitle(f"Vzorky 28x28 pre '{name}'")
    fig2.tight_layout()
    plt.show()


def show_dataset(X, Y):
    """Prehľad celého datasetu: riadok = znak, stĺpce = prvých 10 vzoriek."""
    labels = Y.argmax(axis=1)
    fig, axes = plt.subplots(len(CHARS), 10, figsize=(10, max(1.3, len(CHARS) * 1.1)),
                             squeeze=False)
    for c, ch in enumerate(CHARS):
        idxs = np.where(labels == c)[0][:10]
        for j in range(10):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(idxs):
                ax.imshow(X[idxs[j]].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            if j == 0:
                ax.set_title(ch, fontsize=10, loc="left")
    fig.suptitle("Dataset (z uloženého X)")
    fig.tight_layout()
    plt.show()


def main():
    X, Y = [], []
    for idx, c in enumerate(CHARS):
        path = f"{SCAN_DIR}/{c}.jpg"
        if not os.path.exists(path):
            print(f"[{c}] súbor {path} chýba, preskakujem")
            continue
        samples = process_scan(path, c)
        flag = "OK" if len(samples) >= MIN_SAMPLES else "!!! MÁLO / skontroluj debug/"
        print(f"[{c}] nájdených vzoriek: {len(samples)}  {flag}")
        for s in samples:
            X.append((s.astype(np.float32) / 255.0).reshape(784))   # písmo ~ 1
            y = np.zeros(10, np.float32)
            y[idx] = 1
            Y.append(y)

    X, Y = np.array(X, np.float32), np.array(Y, np.float32)
    np.savez(OUT_FILE, X=X, Y=Y)
    print(f"Uložené {OUT_FILE}: X{X.shape}, Y{Y.shape}")
    if SHOW and len(X):
        show_dataset(X, Y)


if __name__ == "__main__":
    main()