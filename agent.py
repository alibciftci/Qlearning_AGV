"""
Q-Learning ajanı.

Q tablosu: shape = (satir, sutun, aksiyon_sayisi)
Yani her hücre için her aksiyonun tahmini değerini tutuyorum.

Q-Learning güncelleme kuralı (Bellman):
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Bu satır dersin tam kalbi. Burada öğrenme oranı alpha ile yeni
deneyimi eski tahmine ne kadar karıştıracağımızı, gamma ile de
gelecekteki ödülleri ne kadar değerli gördüğümüzü ayarlıyoruz.
"""

import numpy as np


class QAgent:
    def __init__(
        self,
        grid_size=10,
        n_actions=4,
        alpha=0.1,        # öğrenme oranı (learning rate)
        gamma=0.95,       # iskonto faktörü
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=42,
    ):
        # Q tablosunu sıfırlarla başlatıyorum. Optimistic init de
        # denenebilirdi ama burada klasik haliyle gidiyorum.
        self.Q = np.zeros((grid_size, grid_size, n_actions))

        self.alpha = alpha
        self.gamma = gamma

        # Epsilon-greedy parametreleri.
        # Başlarda her şeyi rastgele deniyorum (keşif),
        # zamanla öğrendiklerime güvenmeye başlıyorum (sömürü).
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_actions = n_actions

        # Tekrarlanabilirlik için sabit tohum.
        self.rng = np.random.default_rng(seed)

    def aksiyon_sec(self, state, greedy=False):
        """Epsilon-greedy aksiyon seçimi.
        greedy=True ise eğitilmiş politikayı oynatmak için kullanılır
        (GIF üretirken keşif olmasın diye).
        """
        if not greedy and self.rng.random() < self.epsilon:
            # Keşif: rastgele aksiyon.
            return int(self.rng.integers(0, self.n_actions))

        # Sömürü: en yüksek Q değerine sahip aksiyonu seç.
        r, c = state
        q_degerleri = self.Q[r, c]

        # Birden fazla aksiyon aynı en yüksek değere sahipse aralarından
        # rastgele birini seçiyorum. Aksi halde 0. aksiyon hep
        # öncelikli olur ve ajan her zaman "yukarı" demeye eğilim
        # gösterir.
        en_iyiler = np.flatnonzero(q_degerleri == q_degerleri.max())
        return int(self.rng.choice(en_iyiler))

    def guncelle(self, state, action, reward, next_state, done):
        """Bellman güncellemesi."""
        r, c = state
        nr, nc = next_state

        eski_q = self.Q[r, c, action]

        if done:
            # Terminal durumda gelecek ödül yok.
            hedef = reward
        else:
            hedef = reward + self.gamma * np.max(self.Q[nr, nc])

        # Hatayı kademeli olarak Q tablosuna yansıtıyorum.
        self.Q[r, c, action] = eski_q + self.alpha * (hedef - eski_q)

    def epsilonu_azalt(self):
        # Her episode sonunda biraz daha az keşif, biraz daha sömürü.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
