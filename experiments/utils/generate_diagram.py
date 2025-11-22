import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def draw_box(x, y, width, height, text, color='#E1F5FE', edge='#01579B'):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2",
                                      linewidth=2, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=12, fontweight='bold', color='#000000')
        return rect


    draw_box(0.5, 2.5, 2, 1, "User Prompt\n(Text)", color='#FFF3E0', edge='#EF6C00')
    ax.annotate("", xy=(3.0, 3.0), xytext=(2.7, 3.0), arrowprops=dict(arrowstyle="->", lw=2))

    draw_box(3.2, 2.0, 2.5, 2, "Source Model\n(TinyLlama-1.1B)\n\nEncoder", color='#E3F2FD', edge='#1565C0')
    ax.annotate("", xy=(6.2, 3.0), xytext=(5.9, 3.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(6.05, 3.1, "2048d", ha='center', fontsize=9)

    draw_box(6.4, 2.2, 2.0, 1.6, "LIP Adapter\n(Dual-Encoder)\n\nBottleneck 512d", color='#FFF8E1', edge='#FBC02D')
    ax.annotate("", xy=(9.0, 3.0), xytext=(8.6, 3.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(8.8, 3.1, "LIP Packet\n(4096d)", ha='center', fontsize=9)

    draw_box(9.2, 2.0, 2.5, 2, "Target Model\n(Llama-3-8B)\n\nDecoder", color='#E8F5E9', edge='#2E7D32')
    ax.annotate("", xy=(10.45, 1.8), xytext=(10.45, 2.0), arrowprops=dict(arrowstyle="->", lw=2))

    draw_box(9.45, 0.8, 2.0, 0.8, "Generated Code", color='#F3E5F5', edge='#7B1FA2')

    plt.title("LIP Architecture: Heterogeneous Latent Injection", fontsize=16, pad=20)
    plt.tight_layout()

    output_dir = os.path.join("paper", "figures")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "architecture_diagram.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Diagram saved to {save_path}")

if __name__ == "__main__":
    draw_architecture()