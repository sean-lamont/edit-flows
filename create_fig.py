import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_comparison_figure():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set limits and turn off axes
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # --- Section 1: Standard Edit Flows (Left Side) ---

    # Title
    ax.text(3.5, 7.5, "Standard Edit Flows\n(Incremental Growth)",
            fontsize=14, fontweight='bold', ha='center', va='top')

    # Timeline arrow
    ax.annotate("", xy=(1, 0.5), xytext=(1, 6.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(0.7, 3.5, "Time (t)", rotation=90, va='center', fontsize=12)

    # t=0.0
    ax.text(1.2, 6.5, "t=0.0", fontsize=10)
    rect_bos = patches.FancyBboxPatch((2, 6.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    rect_eos = patches.FancyBboxPatch((4, 6.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    ax.add_patch(rect_bos)
    ax.add_patch(rect_eos)
    ax.text(2.5, 6.5, "[BOS]", ha='center', va='center', fontsize=9)
    ax.text(4.5, 6.5, "[EOS]", ha='center', va='center', fontsize=9)

    # Action arrow
    ax.annotate("Insert 1", xy=(3.5, 5.8), xytext=(3.5, 6.1),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # t=0.1
    ax.text(1.2, 5.0, "t=0.1", fontsize=10)
    rect_bos2 = patches.FancyBboxPatch((2, 4.7), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    # Standard EF inserts Token/Mask. Usually tokens directly or masks one by one.
    # Let's assume masks for direct comparison.
    rect_m1 = patches.FancyBboxPatch((3.2, 4.7), 0.6, 0.6, boxstyle="round,pad=0.1", fc="#FFCCCC", ec="red")
    rect_eos2 = patches.FancyBboxPatch((4, 4.7), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    ax.add_patch(rect_bos2)
    ax.add_patch(rect_m1)
    ax.add_patch(rect_eos2)
    ax.text(2.5, 5.0, "[BOS]", ha='center', va='center', fontsize=9)
    ax.text(3.5, 5.0, "The", ha='center', va='center', fontsize=9, color="red")  # Assuming token insertion
    ax.text(4.5, 5.0, "[EOS]", ha='center', va='center', fontsize=9)

    # Action arrow 2
    ax.annotate("Insert 1", xy=(3.5, 4.3), xytext=(3.5, 4.6),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # t=0.2
    ax.text(1.2, 3.5, "t=0.2", fontsize=10)
    rect_bos3 = patches.FancyBboxPatch((2, 3.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    rect_m2 = patches.FancyBboxPatch((3.2, 3.2), 0.6, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    rect_m3 = patches.FancyBboxPatch((4.0, 3.2), 0.6, 0.6, boxstyle="round,pad=0.1", fc="#FFCCCC", ec="red")
    rect_eos3 = patches.FancyBboxPatch((4.8, 3.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    ax.add_patch(rect_bos3)
    ax.add_patch(rect_m2)
    ax.add_patch(rect_m3)
    ax.add_patch(rect_eos3)
    ax.text(2.5, 3.5, "[BOS]", ha='center', va='center', fontsize=9)
    ax.text(3.5, 3.5, "The", ha='center', va='center', fontsize=9, color="black")
    ax.text(4.3, 3.5, "cat", ha='center', va='center', fontsize=9, color="red")
    ax.text(5.3, 3.5, "[EOS]", ha='center', va='center', fontsize=9)

    # Annotation for standard
    ax.text(3.5, 2.0, "Structure grows\nincrementally (O(N)).\nContext is unstable.",
            fontsize=10, ha='center', style='italic', bbox=dict(boxstyle="round", fc="white", ec="gray"))

    # --- Separator Line ---
    ax.plot([7, 7], [0.5, 7.5], color='gray', linestyle='--', lw=1)

    # --- Section 2: Targeted Edit Flows (Right Side) ---

    # Title
    ax.text(10.5, 7.5, "Targeted Edit Flows (TEF)\n(Instant Scaffolding)",
            fontsize=14, fontweight='bold', ha='center', va='top')

    # Timeline arrow
    ax.annotate("", xy=(8, 0.5), xytext=(8, 6.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(7.7, 3.5, "Time (t)", rotation=90, va='center', fontsize=12)

    # t=0.0
    ax.text(8.2, 6.5, "t=0.0", fontsize=10)
    rect_bos_tef = patches.FancyBboxPatch((9, 6.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    rect_eos_tef = patches.FancyBboxPatch((11, 6.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    ax.add_patch(rect_bos_tef)
    ax.add_patch(rect_eos_tef)
    ax.text(9.5, 6.5, "[BOS]", ha='center', va='center', fontsize=9)
    ax.text(11.5, 6.5, "[EOS]", ha='center', va='center', fontsize=9)

    # Action arrow (Predict k=5)
    ax.annotate("Predict Count k=5", xy=(10.5, 5.8), xytext=(10.5, 6.1),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(12.5, 5.95, "(Structure Head)", fontsize=9, color="green")

    # t=0.01 (Instant Allocation)
    ax.text(8.2, 5.0, "t=0.01", fontsize=10)
    rect_bos_tef2 = patches.FancyBboxPatch((8.8, 4.7), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")

    # 5 Masks
    for i in range(5):
        x_pos = 10.0 + i * 0.7
        rect_m = patches.FancyBboxPatch((x_pos, 4.7), 0.6, 0.6, boxstyle="round,pad=0.1", fc="#CCFFCC", ec="green")
        ax.add_patch(rect_m)
        ax.text(x_pos + 0.3, 5.0, "M", ha='center', va='center', fontsize=9, color="green")

    rect_eos_tef2 = patches.FancyBboxPatch((13.6, 4.7), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    ax.add_patch(rect_bos_tef2)
    ax.add_patch(rect_eos_tef2)
    ax.text(9.3, 5.0, "[BOS]", ha='center', va='center', fontsize=9)
    ax.text(14.1, 5.0, "[EOS]", ha='center', va='center', fontsize=9)

    # Action arrow (Parallel Filling)
    ax.annotate("Parallel Filling", xy=(10.5, 4.0), xytext=(10.5, 4.6),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))

    # t > 0.02 (Filling)
    ax.text(8.2, 3.5, "t > 0.02", fontsize=10)
    rect_bos_tef3 = patches.FancyBboxPatch((8.8, 3.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")

    # Filled Tokens
    tokens = ["The", "cat", "sat", "on", "mat"]
    for i, tok in enumerate(tokens):
        x_pos = 10.0 + i * 0.7
        rect_t = patches.FancyBboxPatch((x_pos, 3.2), 0.6, 0.6, boxstyle="round,pad=0.1", fc="#CCE5FF", ec="blue")
        ax.add_patch(rect_t)
        ax.text(x_pos + 0.3, 3.5, tok, ha='center', va='center', fontsize=7, color="blue")

    rect_eos_tef3 = patches.FancyBboxPatch((13.6, 3.2), 1, 0.6, boxstyle="round,pad=0.1", fc="#E0E0E0", ec="black")
    ax.add_patch(rect_bos_tef3)
    ax.add_patch(rect_eos_tef3)
    ax.text(9.3, 3.5, "[BOS]", ha='center', va='center', fontsize=9)
    ax.text(14.1, 3.5, "[EOS]", ha='center', va='center', fontsize=9)

    # Annotation for TEF
    ax.text(11.5, 2.0, "Structure is O(1).\nContent fills in parallel.\nContext is stable.",
            fontsize=10, ha='center', style='italic', bbox=dict(boxstyle="round", fc="white", ec="green"))

    plt.tight_layout()
    plt.savefig("targeted_edit_flows_comparison_v2.png")


if __name__ == "__main__":
    create_comparison_figure()
