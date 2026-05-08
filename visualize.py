"""
Grafikler ve GIF üretimi.

Burada eğitim verilerini ve Q tablosunu kullanıp ödevin görsel
çıktılarını üretiyorum. Tüm dosyalar 'assets/' klasörüne kaydoluyor.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio


ASSETS_DIR = "assets"


def klasoru_hazirla():
    if not os.path.isdir(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)


def hareketli_ortalama(x, pencere=50):
    # Ödül eğrisi çok dalgalı çıkıyor, hareketli ortalama
    # ile yumuşatınca öğrenme eğilimi netleşiyor.
    if len(x) < pencere:
        return x.copy()
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[pencere:] - cumsum[:-pencere]) / pencere


def odul_grafigi(odul_listesi):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(odul_listesi, alpha=0.3, label="Ham toplam ödül")
    yumusak = hareketli_ortalama(odul_listesi, 50)
    ax.plot(np.arange(49, 49 + len(yumusak)), yumusak,
            color="red", linewidth=2, label="50 episode'luk ortalama")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Toplam Ödül")
    ax.set_title("Episode Başına Toplam Ödül")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "reward_curve.png"), dpi=120)
    plt.close(fig)


def adim_grafigi(adim_listesi):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(adim_listesi, alpha=0.3, label="Ham adım sayısı")
    yumusak = hareketli_ortalama(adim_listesi.astype(float), 50)
    ax.plot(np.arange(49, 49 + len(yumusak)), yumusak,
            color="green", linewidth=2, label="50 episode'luk ortalama")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Adım Sayısı")
    ax.set_title("Episode Başına Adım Sayısı")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "steps_curve.png"), dpi=120)
    plt.close(fig)


def epsilon_grafigi(epsilon_listesi):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epsilon_listesi, color="purple", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Azalma Eğrisi (Keşif → Sömürü)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "epsilon_decay.png"), dpi=120)
    plt.close(fig)


def basari_grafigi(basari_listesi, pencere=100):
    fig, ax = plt.subplots(figsize=(9, 5))
    yumusak = hareketli_ortalama(basari_listesi.astype(float), pencere) * 100
    ax.plot(np.arange(pencere - 1, pencere - 1 + len(yumusak)), yumusak,
            color="darkorange", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Başarı Oranı (%) — {pencere} ep. ortalaması")
    ax.set_title("Hedefe Ulaşma Başarı Oranı")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "success_rate.png"), dpi=120)
    plt.close(fig)


def deger_haritasi(Q, env):
    """Her hücrenin değeri (V(s) = max_a Q(s,a)) ısı haritası olarak."""
    V = Q.max(axis=2)

    # Engelleri NaN yapayım ki ısı haritasında belirgin olsunlar.
    V_gosterim = V.astype(float).copy()
    for (r, c) in env.obstacles:
        V_gosterim[r, c] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(V_gosterim, cmap="viridis")
    ax.set_title("Öğrenilen Değer Fonksiyonu V(s) = max_a Q(s,a)")

    # Engelleri gri kutu olarak çizeyim.
    for (r, c) in env.obstacles:
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1, color="black", alpha=0.85))

    # Başlangıç ve hedef işaretleri.
    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr, "S", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    fig.colorbar(im, ax=ax, label="V(s)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "q_value_heatmap.png"), dpi=120)
    plt.close(fig)


def politika_oklari(Q, env):
    """Her hücrede en iyi aksiyonu ok ile gösteriyorum."""
    fig, ax = _ortam_cizici(env)
    ax.set_title("Öğrenilen Politika (her hücrede en iyi aksiyon)")

    # Aksiyon -> ok yönü (dx, dy). matplotlib'in y ekseni ters
    # olduğu için dikkat etmek gerek; imshow ile hücreleri çiziyorum
    # ve y ekseni aşağı doğru artıyor.
    ok_dx = {0: 0, 1: 0, 2: 0.35, 3: -0.35}    # YUKARI/ASAGI/SAG/SOL
    ok_dy = {0: -0.35, 1: 0.35, 2: 0, 3: 0}

    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r, c) in env.obstacles:
                continue
            if (r, c) == env.goal:
                continue
            best_a = int(np.argmax(Q[r, c]))
            ax.arrow(c, r, ok_dx[best_a], ok_dy[best_a],
                     head_width=0.18, head_length=0.18,
                     fc="white", ec="white", length_includes_head=True)

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "policy_arrows.png"), dpi=120)
    plt.close(fig)


def _ortam_cizici(env):
    """Izgarayı çizen yardımcı; engel/başlangıç/hedef renklendirilir.
    Diğer çizimlerin üstüne ekleme yapabilmek için ax döndürüyorum.
    """
    m = env.grid_matrisi()

    # 0: boş (beyaz), 1: engel (siyah), 2: başlangıç (mavi),
    # 3: hedef (yeşil)
    cmap = ListedColormap(["#f5f5f5", "#222222", "#3b82f6", "#22c55e"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(m, cmap=cmap, vmin=0, vmax=3)

    # Hücre sınırlarını çizeyim, ızgara izlenimi versin.
    ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))

    # Başlangıç ve hedef hücrelerine etiket ekleyeyim.
    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr, "S", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")

    return fig, ax


def gif_uret(env, yol, dosya="agent_run.gif", fps=4):
    """Ajanın izlediği yolu kare kare çiz, GIF olarak kaydet."""
    klasoru_hazirla()
    kareler = []

    for adim_idx, (r, c) in enumerate(yol):
        fig, ax = _ortam_cizici(env)
        ax.set_title(f"AGV — Adım {adim_idx} / {len(yol) - 1}")

        # Şu ana kadarki yolu açık çizgiyle göster.
        if adim_idx > 0:
            ys = [p[0] for p in yol[: adim_idx + 1]]
            xs = [p[1] for p in yol[: adim_idx + 1]]
            ax.plot(xs, ys, color="#ef4444", linewidth=2, alpha=0.7)

        # Ajanın güncel konumu.
        ax.scatter([c], [r], s=380, color="#ef4444",
                   edgecolors="white", linewidths=2, zorder=5)

        # Figure'u numpy görüntüsüne çevirip belleğe ekle.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        kare = np.asarray(renderer.buffer_rgba())[..., :3].copy()
        kareler.append(kare)
        plt.close(fig)

    # Hedef karesinde biraz beklesin diye sonu birkaç kez tekrarlıyorum.
    for _ in range(fps * 2):
        kareler.append(kareler[-1])

    yol_dosya = os.path.join(ASSETS_DIR, dosya)
    imageio.mimsave(yol_dosya, kareler, fps=fps, loop=0)
    return yol_dosya


def harita_resmi(env):
    """Sadece ham haritayı gösteren bir resim, README'de kullanıyorum."""
    fig, ax = _ortam_cizici(env)
    ax.set_title("Fabrika Zemini (10x10) — S: Başlangıç, G: Şarj İstasyonu")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "map.png"), dpi=120)
    plt.close(fig)
