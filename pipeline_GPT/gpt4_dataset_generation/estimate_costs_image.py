import pandas as pd
import base64
from io import BytesIO
from PIL import Image

# Path to the TSV file containing base64-encoded images
TSV_PATH = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"

def estimate_image_cost(image_b64: str) -> float:
    """
    Estimate the cost of a single image based on its resolution, following OpenAI pricing tiers.
    """
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes))
        w, h = image.size
    except Exception:
        # Default to the maximum cost if the image is unreadable
        return 0.02

    if w <= 512 and h <= 512:
        return 0.0045
    elif w <= 720 and h <= 720:
        return 0.0075
    elif w <= 1080 and h <= 1080:
        return 0.01
    else:
        return 0.02

def main():
    # Load the dataset
    df = pd.read_csv(TSV_PATH, sep="\t")

    total_cost = 0
    cost_bins = {0.0045: 0, 0.0075: 0, 0.01: 0, 0.02: 0}

    # Iterate over each image and accumulate cost
    for b64 in df["image"]:
        cost = estimate_image_cost(b64)
        cost_bins[cost] += 1
        total_cost += cost

    # Summary output
    print("\n📊 Image Cost Estimation Summary:")
    for price in sorted(cost_bins):
        count = cost_bins[price]
        print(f"- {count} image(s) at ${price:.4f} : ${price * count:.2f}")

    print(f"\n💰 Total estimated cost for all images: ${total_cost:.2f}")

if __name__ == "__main__":
    main()
