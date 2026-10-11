# Image Noise Suppression: Adaptive vs. Ordinary Averaging

Digital Image Processing homework (*Digitalna obrada slike*), implemented in Python with NumPy, OpenCV and Matplotlib. The filters are written by hand, including the 2D convolution with zero padding.

## Task

For the test image (`slika 47.jpg`):

1. Suppress noise with an **adaptive mean filter**.
2. Compare it with **ordinary spatial averaging** for different kernel sizes (3x3, 5x5, 7x7).
3. Apply a **Laplacian filter** to highlight edges on the original and on the filtered images.

## Method

- **Ordinary averaging:** convolution with a `k x k` kernel of equal weights `1/k^2`.
- **Adaptive averaging:** the filter uses the local variance in each window. In flat regions it averages strongly, near edges it keeps the original pixel:
  `out = z - (noise_var / local_var) * (z - local_mean)`
  The noise variance is estimated from a sample region of the image.
- **Laplacian:** 3x3 (8-neighbour) kernel. The result is added to the image to enhance edges.

## Getting started

```bash
pip install -r requirements.txt
python main.py
```

The script reads `slika 47.jpg` from the same folder and saves these figures next to it:

| File | Content |
|---|---|
| `zadatak_75_original.png` | Original image |
| `zadatak_75_rezultat.png` | Ordinary vs. adaptive averaging for each kernel size |
| `zadatak_75_laplacian.png` | Edge enhancement on the original and filtered images |

## Project structure

```
main.py            # filters, convolution and plotting
slika 47.jpg       # test image
requirements.txt   # opencv-python, numpy, matplotlib
```
