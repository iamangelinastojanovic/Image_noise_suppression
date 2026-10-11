"""
2022/0081 - Angelina Stojanovic
Zadatak 75, slika 47
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def zero_padding(img: np.ndarray, pad: int) -> np.ndarray:
    # Jednostavna zero-padding funkcija za konvoluciju.
    if pad <= 0:
        return img
    # padduje nulama
    return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Ručna implementacija 2D konvolucije (za sive slike, kernel neparne veličine).
    # Koristi zero-padding tako da izlaz ima iste dimenzije kao ulaz.

    # konvertuje sliku i kernel u float64
    img = img.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # proverava da li je kernel kvadratan i neparne dimenzije
    kh, kw = kernel.shape
    if kh != kw or kh % 2 == 0:
        raise ValueError("Kernel mora biti kvadratan i neparne dimenzije.")

    pad = kh // 2
    padded = zero_padding(img, pad)
    h, w = img.shape
    # inicijalizuje izlaznu sliku sa nulama
    out = np.zeros_like(img, dtype=np.float64)

    # konvolucija
    for i in range(h):
        for j in range(w):
            region = padded[i: i + kh, j: j + kw]
            out[i, j] = np.sum(region * kernel)

    return out


# ucitavanje slike u nijansama sive
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nije moguće učitati sliku: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ucitava sliku u nijansama sive
    return gray.astype(np.float64)


def ordinary_averaging(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    out = conv2d(img, kernel)
    return np.clip(out, 0, 255)


def adaptive_averaging(img: np.ndarray, ksize: int = 5, noise_var: float = None) -> np.ndarray:
    pad = ksize // 2
    # sluzi za konvoluciju
    padded_img = zero_padding(img, pad)

    out = np.zeros_like(img)
    h, w = img.shape[0], img.shape[1]

    if noise_var is None:
        noise_var = noise_var_estimate(img)
    if noise_var <= 1e-3:
        return img
    else:
        for i in range(h):
            for j in range(w):
                win = padded_img[i: i + ksize, j: j + ksize]
                z_sxy = np.mean(win)
                var_sxy = np.var(win)
                if var_sxy - noise_var <= 1e-3:
                    out[i, j] = z_sxy
                else:
                    out[i, j] = img[i, j] - (noise_var / var_sxy) * (img[i, j] - z_sxy)

    return np.clip(out, 0, 255)


def noise_var_estimate(img: np.ndarray, sample_frac: float = 0.1, h1: int = None, h2: int = None, w1: int = None,
                       w2: int = None) -> float:
    h, w = img.shape

    if h1 is None or h1 < 0:
        h1 = 0
    if h2 is None or h2 > h:
        h2 = h - 1
    if w1 is None or w1 < 0:
        w1 = 0
    if w2 is None or w2 < 0:
        w2 = w - 1

    region = img[h1:h2, w1:w2]
    return float(np.var(region))


def laplacian_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    # Za ksize=3 koristimo standardni 8-susedni Laplacian kernel
    if ksize != 3:
        raise ValueError("Ručna implementacija trenutno podržava samo ksize=3 za Laplacian.")

    kernel = np.array(
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]],
        dtype=np.float64,
    )

    lap = conv2d(img, kernel)
    return lap


def laplacian_enhance(img: np.ndarray, ksize: int = 3, alpha: float = 1.0) -> np.ndarray:
    # Primena Laplacian filtra i sabiranje sa originalnom slikomradi isticanja ivica (edge enhancement).
    lap = laplacian_edges(img, ksize=ksize)
    enhanced = img.astype(np.float64) + alpha * lap
    return np.clip(enhanced, 0, 255)


def main():
    image_path = Path(__file__).parent / "slika 47.jpg"
    img = load_image(str(image_path))

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap="gray")
    plt.title("Originalna slika")
    plt.axis("off")
    plt.savefig(Path(__file__).parent / "zadatak_75_original.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Originalna slika sačuvana u: zadatak_75_original.png")

    ksizes = [3, 5, 7]

    noise_var = noise_var_estimate(img)

    # 1) Obično i adaptivno usrednjavanje – 2x3 potplota u jednoj figuri
    fig = plt.figure()
    fig.suptitle("Zadatak 75: Obično i adaptivno usrednjavanje za različite veličine kernela", fontsize=10)
    images_ordinary, images_adaptive = [], []
    i = 1
    for ksize in ksizes:
        img_ordinary = ordinary_averaging(img, ksize=ksize)
        img_adaptive = adaptive_averaging(img, ksize=ksize, noise_var=noise_var)
        images_ordinary.append(img_ordinary)
        images_adaptive.append(img_adaptive)

        ax1 = plt.subplot(2, 3, i)
        ax1.imshow(img_ordinary, cmap="gray")
        ax1.set_title(f"Obično - k={ksize}")
        ax1.axis("off")

        ax2 = plt.subplot(2, 3, i + 3)
        ax2.imshow(img_adaptive, cmap="gray")
        ax2.set_title(f"Adaptivno - k={ksize}")
        ax2.axis("off")
        i += 1

    # uklanjamo sve ose/tickove i zbijamo slike
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.05, hspace=0.15)
    plt.savefig(Path(__file__).parent / "zadatak_75_rezultat.png", dpi=150, bbox_inches="tight")
    # plt.title("Zadatak 75: Potiskivanje šuma za različite veličine kernela (Slika 47)", fontsize=14)
    plt.close()
    print("Rezultat usrednjavanja sačuvan u: zadatak_75_rezultat.png")

    # 3) Laplacian – isticanje ivica sabiranjem sa slikom
    #    (original, posle običnog i posle adaptivnog usrednjavanja – koristimo k=3 iz gore sačuvanih lista)
    img_ordinary3 = images_ordinary[0]
    img_adaptive3 = images_adaptive[0]

    lap_original = laplacian_enhance(img, ksize=3, alpha=1.0)
    lap_ordinary = laplacian_enhance(img_ordinary3, ksize=3, alpha=1.0)
    lap_adaptive = laplacian_enhance(img_adaptive3, ksize=3, alpha=1.0)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle("Laplacian + slika: istaknute ivice", fontsize=12)

    axes2[0].imshow(lap_original, cmap="gray")
    axes2[0].set_title("Original + Laplacian")
    axes2[0].axis("off")

    axes2[1].imshow(lap_ordinary, cmap="gray")
    axes2[1].set_title("Obično usrednjavanje + Laplacian")
    axes2[1].axis("off")

    axes2[2].imshow(lap_adaptive, cmap="gray")
    axes2[2].set_title("Adaptivno usrednjavanje + Laplacian")
    axes2[2].axis("off")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "zadatak_75_laplacian.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Slike sa istaknutim ivicama sačuvane u: zadatak_75_laplacian.png")


if __name__ == "__main__":
    main()
"""
2022/0081 - Angelina Stojanovic
Zadatak 75, slika 47
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def zero_padding(img: np.ndarray, pad: int) -> np.ndarray:
    # Jednostavna zero-padding funkcija za konvoluciju.
    if pad <= 0:
        return img
    # padduje nulama
    return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Ručna implementacija 2D konvolucije (za sive slike, kernel neparne veličine).
    # Koristi zero-padding tako da izlaz ima iste dimenzije kao ulaz.

    # konvertuje sliku i kernel u float64
    img = img.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # proverava da li je kernel kvadratan i neparne dimenzije
    kh, kw = kernel.shape
    if kh != kw or kh % 2 == 0:
        raise ValueError("Kernel mora biti kvadratan i neparne dimenzije.")

    pad = kh // 2
    padded = zero_padding(img, pad)
    h, w = img.shape
    # inicijalizuje izlaznu sliku sa nulama
    out = np.zeros_like(img, dtype=np.float64)

    # konvolucija
    for i in range(h):
        for j in range(w):
            region = padded[i: i + kh, j: j + kw]
            out[i, j] = np.sum(region * kernel)

    return out


# ucitavanje slike u nijansama sive
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nije moguće učitati sliku: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ucitava sliku u nijansama sive
    return gray.astype(np.float64)


def ordinary_averaging(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    out = conv2d(img, kernel)
    return np.clip(out, 0, 255)


def adaptive_averaging(img: np.ndarray, ksize: int = 5, noise_var: float = None) -> np.ndarray:
    pad = ksize // 2
    # sluzi za konvoluciju
    padded_img = zero_padding(img, pad)

    out = np.zeros_like(img)
    h, w = img.shape[0], img.shape[1]

    if noise_var is None:
        noise_var = noise_var_estimate(img)
    if noise_var <= 1e-3:
        return img
    else:
        for i in range(h):
            for j in range(w):
                win = padded_img[i: i + ksize, j: j + ksize]
                z_sxy = np.mean(win)
                var_sxy = np.var(win)
                if var_sxy - noise_var <= 1e-3:
                    out[i, j] = z_sxy
                else:
                    out[i, j] = img[i, j] - (noise_var / var_sxy) * (img[i, j] - z_sxy)

    return np.clip(out, 0, 255)


def noise_var_estimate(img: np.ndarray, sample_frac: float = 0.1, h1: int = None, h2: int = None, w1: int = None,
                       w2: int = None) -> float:
    h, w = img.shape

    if h1 is None or h1 < 0:
        h1 = 0
    if h2 is None or h2 > h:
        h2 = h - 1
    if w1 is None or w1 < 0:
        w1 = 0
    if w2 is None or w2 < 0:
        w2 = w - 1

    region = img[h1:h2, w1:w2]
    return float(np.var(region))


def laplacian_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    # Za ksize=3 koristimo standardni 8-susedni Laplacian kernel
    if ksize != 3:
        raise ValueError("Ručna implementacija trenutno podržava samo ksize=3 za Laplacian.")

    kernel = np.array(
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]],
        dtype=np.float64,
    )

    lap = conv2d(img, kernel)
    return lap


def laplacian_enhance(img: np.ndarray, ksize: int = 3, alpha: float = 1.0) -> np.ndarray:
    # Primena Laplacian filtra i sabiranje sa originalnom slikomradi isticanja ivica (edge enhancement).
    lap = laplacian_edges(img, ksize=ksize)
    enhanced = img.astype(np.float64) + alpha * lap
    return np.clip(enhanced, 0, 255)


def main():
    image_path = Path(__file__).parent / "slika 47.jpg"
    img = load_image(str(image_path))

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap="gray")
    plt.title("Originalna slika")
    plt.axis("off")
    plt.savefig(Path(__file__).parent / "zadatak_75_original.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Originalna slika sačuvana u: zadatak_75_original.png")

    ksizes = [3, 5, 7]

    noise_var = noise_var_estimate(img)

    # 1) Obično i adaptivno usrednjavanje – 2x3 potplota u jednoj figuri
    fig = plt.figure()
    fig.suptitle("Zadatak 75: Obično i adaptivno usrednjavanje za različite veličine kernela", fontsize=10)
    images_ordinary, images_adaptive = [], []
    i = 1
    for ksize in ksizes:
        img_ordinary = ordinary_averaging(img, ksize=ksize)
        img_adaptive = adaptive_averaging(img, ksize=ksize, noise_var=noise_var)
        images_ordinary.append(img_ordinary)
        images_adaptive.append(img_adaptive)

        ax1 = plt.subplot(2, 3, i)
        ax1.imshow(img_ordinary, cmap="gray")
        ax1.set_title(f"Obično - k={ksize}")
        ax1.axis("off")

        ax2 = plt.subplot(2, 3, i + 3)
        ax2.imshow(img_adaptive, cmap="gray")
        ax2.set_title(f"Adaptivno - k={ksize}")
        ax2.axis("off")
        i += 1

    # uklanjamo sve ose/tickove i zbijamo slike
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.05, hspace=0.15)
    plt.savefig(Path(__file__).parent / "zadatak_75_rezultat.png", dpi=150, bbox_inches="tight")
    # plt.title("Zadatak 75: Potiskivanje šuma za različite veličine kernela (Slika 47)", fontsize=14)
    plt.close()
    print("Rezultat usrednjavanja sačuvan u: zadatak_75_rezultat.png")

    # 3) Laplacian – isticanje ivica sabiranjem sa slikom
    #    (original, posle običnog i posle adaptivnog usrednjavanja – koristimo k=3 iz gore sačuvanih lista)
    img_ordinary3 = images_ordinary[0]
    img_adaptive3 = images_adaptive[0]

    lap_original = laplacian_enhance(img, ksize=3, alpha=1.0)
    lap_ordinary = laplacian_enhance(img_ordinary3, ksize=3, alpha=1.0)
    lap_adaptive = laplacian_enhance(img_adaptive3, ksize=3, alpha=1.0)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle("Laplacian + slika: istaknute ivice", fontsize=12)

    axes2[0].imshow(lap_original, cmap="gray")
    axes2[0].set_title("Original + Laplacian")
    axes2[0].axis("off")

    axes2[1].imshow(lap_ordinary, cmap="gray")
    axes2[1].set_title("Obično usrednjavanje + Laplacian")
    axes2[1].axis("off")

    axes2[2].imshow(lap_adaptive, cmap="gray")
    axes2[2].set_title("Adaptivno usrednjavanje + Laplacian")
    axes2[2].axis("off")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "zadatak_75_laplacian.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Slike sa istaknutim ivicama sačuvane u: zadatak_75_laplacian.png")

if __name__ == "__main__":
    main()
