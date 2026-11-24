import argparse
import csv
import json
from pathlib import Path

import numpy as np

from baselines import median_restore
from dataset import AMPLITUDES, FREQUENCIES, add_stripes, load_gray
from notch import restore

FIELDS = ["image_id", "scene_type", "split", "ky", "kx", "amplitude", "method",
          "setting", "psnr_db", "gain_db", "peak_y", "peak_x", "peak_error_bins"] #defines the columns of our output csv files


def psnr(clean, estimate): #this function measures how close the original image is to the restored image
    mse = np.mean((clean - estimate) ** 2)
    return np.inf if mse == 0 else -10 * np.log10(mse)

def evaluate_case(photo, clean, ky, kx, amplitude, sizes, sigmas):
    noisy = add_stripes(clean, ky, kx, amplitude)
    initial = psnr(clean, noisy)
    outputs = [] #empty output array
    outputs.append(("unchanged", "", noisy, None)) # adding base values to the array

    for size in sizes:
        filtered = median_restore(noisy, size)
        outputs.append(("median", size, filtered, None)) # for loop for the size

    for sigma in sigmas:
        image, peak = restore(noisy, sigma)
        outputs.append(("fourier", sigma, image, peak))

    rows = []

    for method, setting, image, peak in outputs:
        score = psnr(clean, image)
        gain = score - initial

        error = ""
        peak_y = ""
        peak_x = ""

        if peak is not None:
            peak_y = peak[0]
            peak_x = peak[1]

            error1 = np.hypot(peak_y - ky, peak_x - kx)
            error2 = np.hypot(peak_y + ky, peak_x + kx)

            error = min(error1, error2)

        row = {
            "image_id": photo["image_id"],
            "scene_type": photo["scene_type"],
            "split": photo["split"],
            "ky": ky,
            "kx": kx,
            "amplitude": amplitude,
            "method": method,
            "setting": setting,
            "psnr_db": score,
            "gain_db": gain,
            "peak_y": peak_y,
            "peak_x": peak_x,
            "peak_error_bins": error
        }

        rows.append(row)

    return rows

def run_split(split, sizes, sigmas):
    photos = []

    with open("data/manifest.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] == split:
                photos.append(row)

    image_ids = set()

    for photo in photos:
        image_ids.add(photo["image_id"])

    if len(photos) != 6 or len(image_ids) != 6:
        raise ValueError("Expected six unique images")

    rows = []

    for photo in photos:
        clean = load_gray(photo["path"])

        for ky, kx in FREQUENCIES:
            for amplitude in AMPLITUDES:
                results = evaluate_case(photo, clean, ky, kx, amplitude, sizes, sigmas)
                rows.extend(results)

    filename = "results/" + split + ".csv"

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows

def select_setting(rows, method):
    gains = {}

    for row in rows:
        if row["method"] == method:
            setting = row["setting"]

            if setting not in gains:
                gains[setting] = []

            gains[setting].append(row["gain_db"])

    best_setting = None
    best_gain = -np.inf

    for setting in sorted(gains):
        average = np.mean(gains[setting])

        if best_setting is None or average > best_gain:
            best_gain = average
            best_setting = setting

    return best_setting

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--settings", type=Path, default=Path("results/settings.json"))

    args = parser.parse_args()
    split = args.split
    settings_file = args.settings

    if split == "dev":
        sizes = [3, 5, 7]
        sigmas = [0.75, 1.5, 3.0]

        rows = run_split(split, sizes, sigmas)

        median_size = select_setting(rows, "median")
        notch_sigma = select_setting(rows, "fourier")

        chosen = {
            "median_size": median_size,
            "notch_sigma": notch_sigma
        }

