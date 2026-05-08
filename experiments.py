"""
Hiperparametre karşılaştırma deneyleri.

main.py varsayılan parametrelerle bir eğitim koşturuyor; burada ise
aynı kodu farklı alpha, gamma ve max_steps değerleriyle çalıştırıp
sonuçları üst üste plotluyorum. README'deki "Hiperparametre
Karşılaştırması" bölümü bu dosyanın çıktısına dayanıyor.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import numpy as np
import matplotlib.pyplot as plt

from train import egit
from visualize import ASSETS_DIR, hareketli_ortalama, klasoru_hazirla


N_EPISODES = 1500   # main.py'dakinden biraz az tutuyorum, deneyler hızlansın
SEED = 42


def koş(etiket, agent_kwargs=None, env_max_steps=None):
    """Tek bir konfigürasyonla eğitim çalıştırıp logları döndür."""
    print(f"  -> {etiket}")
    _, _, log = egit(
        n_episodes=N_EPISODES,
        seed=SEED,
        agent_kwargs=agent_kwargs,
        env_max_steps=env_max_steps,
        sessiz=True,
    )
    return log


def cizdir(deneyler, baslik, dosya, pencere=50):
    """Birden fazla koşumun ödül eğrilerini tek grafikte üst üste çizer."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for etiket, log in deneyler.items():
        yumusak = hareketli_ortalama(log["odul"], pencere)
        x = np.arange(pencere - 1, pencere - 1 + len(yumusak))
        ax.plot(x, yumusak, linewidth=2, label=etiket)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Toplam Ödül ({pencere} ep. ortalaması)")
    ax.set_title(baslik)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    yol = os.path.join(ASSETS_DIR, dosya)
    fig.savefig(yol, dpi=120)
    plt.close(fig)
    print(f"     kaydedildi: {yol}")


def basari_ozeti(log, son_n=200):
    """Son N episode'da başarı oranı ve ortalama ödül."""
    son_basari = log["basari"][-son_n:].mean() * 100
    son_odul = log["odul"][-son_n:].mean()
    return son_basari, son_odul


def main():
    klasoru_hazirla()

    print("\n[Deney 1] Öğrenme oranı (alpha) etkisi")
    alpha_deneyleri = {
        "alpha=0.01 (çok yavaş)": koş("a=0.01", {"alpha": 0.01}),
        "alpha=0.1 (varsayılan)":  koş("a=0.1",  {"alpha": 0.1}),
        "alpha=0.5 (agresif)":     koş("a=0.5",  {"alpha": 0.5}),
    }
    cizdir(alpha_deneyleri,
           "Öğrenme Oranı (alpha) Karşılaştırması",
           "exp_alpha.png")

    print("\n[Deney 2] İskonto faktörü (gamma) etkisi")
    gamma_deneyleri = {
        "gamma=0.5 (kısa görüşlü)":  koş("g=0.5",  {"gamma": 0.5}),
        "gamma=0.95 (varsayılan)":   koş("g=0.95", {"gamma": 0.95}),
        "gamma=0.99 (uzak görüşlü)": koş("g=0.99", {"gamma": 0.99}),
    }
    cizdir(gamma_deneyleri,
           "İskonto Faktörü (gamma) Karşılaştırması",
           "exp_gamma.png")

    print("\n[Deney 3] Episode başına maksimum adım etkisi")
    maxstep_deneyleri = {
        "max_steps=50 (kısa)":   koş("ms=50",  env_max_steps=50),
        "max_steps=200 (varsayılan)": koş("ms=200", env_max_steps=200),
        "max_steps=500 (uzun)":  koş("ms=500", env_max_steps=500),
    }
    cizdir(maxstep_deneyleri,
           "Episode Başına Maksimum Adım Karşılaştırması",
           "exp_maxsteps.png")

    # Konsol özeti — README'ye yazarken kullanışlı.
    print("\n" + "=" * 70)
    print("Son 200 episode özeti (başarı %, ortalama ödül)")
    print("=" * 70)
    for grup_adi, grup in [("Alpha", alpha_deneyleri),
                            ("Gamma", gamma_deneyleri),
                            ("Max Steps", maxstep_deneyleri)]:
        print(f"\n{grup_adi}:")
        for etiket, log in grup.items():
            b, o = basari_ozeti(log)
            print(f"  {etiket:35s} | başarı={b:5.1f}% | ort.ödül={o:7.2f}")


if __name__ == "__main__":
    main()
