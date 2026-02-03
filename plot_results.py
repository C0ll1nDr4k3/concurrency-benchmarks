import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
import os

def plot_results(results_path, output_dir="paper/plots", dpi=1200):
    if not os.path.exists(results_path):
        print(f"Error: Results file {results_path} not found.")
        return

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Plot 1: Recall vs QPS ---
    if "recall_vs_qps" in data and data["recall_vs_qps"]:
        print("Plotting Recall vs QPS...")
        plt.figure(figsize=(10, 6))
        
        recall_data = data["recall_vs_qps"]
        K = recall_data.get("K", 10)
        DIM = recall_data.get("DIM", 128)
        
        for name, recalls, qps, style in recall_data.get("runs", []):
             plt.plot(recalls, qps, style, label=name)

        plt.xlabel("Recall")
        plt.ylabel("QPS (log scale)")
        plt.yscale("log")
        plt.title(f"Recall vs QPS (K={K}, Dim={DIM})")
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(output_dir, "recall_vs_qps.svg")
        plt.savefig(out_path, dpi=dpi)
        print(f"Saved {out_path}")
        plt.close()

    # --- Plot 2: Throughput vs Threads ---
    if "throughput" in data and data["throughput"]:
        print("Plotting Throughput vs Threads...")
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        throughput_results = data["throughput"]
        thread_counts = data["thread_counts"]
        rw_ratio = data.get("rw_ratio", 0.1)
        external_names = data.get("external_names", [])

        # Icon mapping: "Substring": ("path", zoom)
        ICON_MAPPING = {
            "FAISS": ("paper/imgs/meta.png", 0.005),
            "USearch": ("paper/imgs/usearch.png", 0.005),
            "Weaviate": ("paper/imgs/weaviate.png", 0.005),
        }

        loaded_icons = {}
        for key, (path, zoom) in ICON_MAPPING.items():
            if os.path.exists(path):
                try:
                    loaded_icons[key] = (mpimg.imread(path), zoom)
                except Exception as e:
                    print(f"Could not load icon {path}: {e}")

        for name, res in throughput_results.items():
            # Check for icon match
            icon_data = None
            for key, (img, zoom) in loaded_icons.items():
                if key in name:
                    icon_data = (img, zoom)
                    break

            if icon_data:
                img, zoom = icon_data
                plt.plot(thread_counts, res, "--", label=name, alpha=0.75)
                for x, y in zip(thread_counts, res):
                    im = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                    ax.add_artist(ab)
            else:
                style = "*--" if name in external_names else "o-"
                plt.plot(thread_counts, res, style, label=name, alpha=0.75)

        plt.xlabel("Threads")
        plt.ylabel("Ops/sec")
        plt.title(f"Throughput (W:{rw_ratio:.1f}, R:{1.0 - rw_ratio:.1f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "throughput_scaling.svg")
        plt.savefig(out_path, dpi=dpi)
        print(f"Saved {out_path}")
        plt.close()

    # --- Plot 3: Conflict Rates ---
    if "conflicts" in data and data["conflicts"]:
        print("Plotting Conflict Rates...")
        plt.figure(figsize=(8, 6))
        conflict_results = data["conflicts"]
        thread_counts = data["thread_counts"]
        
        has_conflicts = False
        for name, conflicts in conflict_results.items():
            if "Opt" in name:
                plt.plot(thread_counts, conflicts, "x--", label=name)
                has_conflicts = True

        if has_conflicts:
            plt.xlabel("Threads")
            plt.ylabel("Conflict Rate (%)")
            plt.title("Conflict Rate (Optimistic)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(output_dir, "conflict_rate.svg")
            plt.savefig(out_path, dpi=dpi)
            print(f"Saved {out_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from cached file")
    parser.add_argument("--results", type=str, default="benchmark_results.pkl", help="Path to pickle file")
    parser.add_argument("--out", type=str, default="paper/plots", help="Output directory")
    parser.add_argument("--dpi", type=int, default=1200, help="DPI for plots")
    
    args = parser.parse_args()
    plot_results(args.results, args.out, args.dpi)
