"""
Tek noktadan çalıştırma:
    python main.py

Sırasıyla:
  1) Eğitimi başlatır
  2) Q tablosunu kaydeder
  3) Tüm grafikleri ve GIF'i 'assets/' klasörüne üretir
"""

import sys

# Windows konsolunun varsayılan kodlaması (cp1252) Türkçe karakterleri
# desteklemiyor. Çıktıyı UTF-8'e çeviriyorum ki print'lerde 'ş', 'ğ',
# 'ı' gibi harfler hata vermesin.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from train import egit, politikayi_oynat
import visualize as vis


def main():
    print("=" * 60)
    print("AGV Q-Learning eğitimi başlıyor...")
    print("=" * 60)

    agent, env, log = egit(n_episodes=2000, seed=42)

    print("\nEğitim bitti. Sonuçlar üretiliyor...")

    # Q tablosunu diske kaydediyorum, tekrar eğitime gerek
    # kalmadan başkası kontrol edebilsin diye.
    np.save("q_table.npy", agent.Q)
    print("Q tablosu kaydedildi: q_table.npy")

    # Klasörü hazırlayıp grafikleri üretiyorum.
    vis.klasoru_hazirla()
    vis.harita_resmi(env)
    vis.odul_grafigi(log["odul"])
    vis.adim_grafigi(log["adim"])
    vis.epsilon_grafigi(log["epsilon"])
    vis.basari_grafigi(log["basari"])
    vis.deger_haritasi(agent.Q, env)
    vis.politika_oklari(agent.Q, env)

    # Eğitilmiş ajanı greedy modda oynatıp GIF üret.
    yol, toplam_odul = politikayi_oynat(env, agent)
    print(f"\nEğitilmiş politika ile yol: {len(yol) - 1} adım, "
          f"toplam ödül = {toplam_odul:.1f}")
    print(f"Ulaşılan son hücre: {yol[-1]} (hedef: {env.goal})")

    gif_yolu = vis.gif_uret(env, yol, dosya="agent_run.gif", fps=4)
    print(f"GIF kaydedildi: {gif_yolu}")

    print("\nTüm görseller 'assets/' klasöründe.")


if __name__ == "__main__":
    main()
