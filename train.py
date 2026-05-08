"""
Eğitim döngüsü.

Burada her episode'da:
  1) Ortamı sıfırla
  2) Episode bitene kadar aksiyon seç -> uygula -> Q'yu güncelle
  3) Episode bitiminde epsilonu azalt
adımlarını uyguluyorum. Her episode'un toplam ödülünü ve adım
sayısını kaydederek sonra grafikleyeceğim.
"""

import numpy as np

from environment import GridWorld
from agent import QAgent


def egit(n_episodes=2000, seed=42, log_her=100, agent_kwargs=None,
         env_max_steps=None, sessiz=False):
    """
    Q-Learning eğitimini çalıştırır.

    agent_kwargs: QAgent'a geçirilecek ekstra parametreler. Hiperparametre
                  karşılaştırmaları için kullanıyorum (mesela alpha=0.5).
    env_max_steps: ortamın episode başına adım sınırı. Default 200.
    sessiz: True ise terminale log basmaz (karşılaştırma koşumları için).
    """
    env = GridWorld()
    if env_max_steps is not None:
        env.max_steps = env_max_steps

    agent_kwargs = agent_kwargs or {}
    agent = QAgent(grid_size=env.grid_size, seed=seed, **agent_kwargs)

    # İstatistikler.
    odul_listesi = np.zeros(n_episodes)
    adim_listesi = np.zeros(n_episodes, dtype=int)
    epsilon_listesi = np.zeros(n_episodes)
    basari_listesi = np.zeros(n_episodes, dtype=int)  # 1 = hedefe ulaştı

    for ep in range(n_episodes):
        state = env.reset()
        toplam_odul = 0.0
        adim = 0
        basari = 0

        done = False
        while not done:
            action = agent.aksiyon_sec(state)
            yeni_state, reward, done = env.step(action)

            agent.guncelle(state, action, reward, yeni_state, done)

            state = yeni_state
            toplam_odul += reward
            adim += 1

            if done and reward == env.r_hedef:
                basari = 1

        odul_listesi[ep] = toplam_odul
        adim_listesi[ep] = adim
        epsilon_listesi[ep] = agent.epsilon
        basari_listesi[ep] = basari

        agent.epsilonu_azalt()

        # Belirli aralıklarla terminale ilerleme yazıyorum, eğitimin
        # nasıl gittiğini görebilmek için.
        if not sessiz and (ep + 1) % log_her == 0:
            son_100_odul = odul_listesi[max(0, ep - 99): ep + 1].mean()
            son_100_basari = basari_listesi[max(0, ep - 99): ep + 1].mean()
            print(
                f"Episode {ep + 1:4d} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"son100_ortalama_odul={son_100_odul:7.2f} | "
                f"basari_orani={son_100_basari * 100:5.1f}%"
            )

    return agent, env, {
        "odul": odul_listesi,
        "adim": adim_listesi,
        "epsilon": epsilon_listesi,
        "basari": basari_listesi,
    }


def politikayi_oynat(env, agent, max_steps=200):
    """Eğitilmiş ajanı greedy modda çalıştırıp izlediği yolu döndürür.
    GIF üretirken bu yolu kullanacağım.
    """
    state = env.reset()
    yol = [state]
    odul_toplami = 0.0

    for _ in range(max_steps):
        action = agent.aksiyon_sec(state, greedy=True)
        state, reward, done = env.step(action)
        yol.append(state)
        odul_toplami += reward
        if done:
            break

    return yol, odul_toplami
