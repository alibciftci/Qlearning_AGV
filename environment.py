"""
GridWorld ortamı.
Fabrika zeminini 10x10 bir ızgara olarak modelliyorum.
Robot bir hücreden komşu hücreye gidebiliyor (yukarı/aşağı/sağ/sol).
Bazı hücreler engel (raf, makine vs.). Robot oraya girerse kaza
sayılıp episode biter.
"""

import numpy as np


# Aksiyon kodları. Sayılar yerine isim kullanmak kodu okunaklı yapıyor.
YUKARI = 0
ASAGI = 1
SAG = 2
SOL = 3

ACTIONS = [YUKARI, ASAGI, SAG, SOL]
ACTION_NAMES = {YUKARI: "Yukarı", ASAGI: "Aşağı", SAG: "Sağ", SOL: "Sol"}


class GridWorld:
    def __init__(self):
        # Izgara boyutu. Ödevde 10x10 isteniyor.
        self.grid_size = 10

        # Başlangıç ve hedef hücreleri.
        # (satır, sütun) sırasıyla tutuyorum.
        self.start = (0, 0)
        self.goal = (9, 9)

        # Engellerin yerleri. Elle koyduğumda labirent benzeri
        # bir yapı çıkıyor; tamamen rastgele yapsam bazen hedefe
        # giden yol kapanabiliyor, bu yüzden manuel seçtim.
        self.obstacles = {
            (1, 2), (1, 3), (1, 6),
            (2, 6),
            (3, 1), (3, 2), (3, 3), (3, 6), (3, 7),
            (5, 5),
            (6, 1), (6, 2), (6, 3), (6, 5), (6, 7), (6, 8),
            (8, 2), (8, 3), (8, 4), (8, 7),
        }

        # Ödüller. Soruda verilen değerleri kullandım.
        self.r_hedef = 100
        self.r_engel = -50
        self.r_adim = -1

        # Episode'un takıldığı durumlarda sonsuza kadar dönmesin
        # diye bir adım sınırı koyuyorum.
        self.max_steps = 200

        # İçeride tuttuğum durum bilgisi.
        self.agent_pos = self.start
        self.step_count = 0

    def reset(self):
        # Her episode başında ajanı başlangıca koyuyorum.
        self.agent_pos = self.start
        self.step_count = 0
        return self.agent_pos

    def _hareket_uygula(self, state, action):
        # Verilen aksiyona göre yeni koordinatı hesaplıyorum.
        r, c = state
        if action == YUKARI:
            r -= 1
        elif action == ASAGI:
            r += 1
        elif action == SAG:
            c += 1
        elif action == SOL:
            c -= 1
        return (r, c)

    def _gecerli_mi(self, state):
        # Koordinat ızgaranın dışına çıkmış mı kontrolü.
        r, c = state
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def step(self, action):
        """Bir aksiyon uygula, (yeni_state, reward, done) döndür."""
        self.step_count += 1

        yeni_pos = self._hareket_uygula(self.agent_pos, action)

        # Duvara çarpma durumu: ajanı yerinde tutuyorum,
        # ama yine de adım cezası alıyor (zaman kaybı sayılır).
        if not self._gecerli_mi(yeni_pos):
            yeni_pos = self.agent_pos
            reward = self.r_adim
            done = False
        elif yeni_pos in self.obstacles:
            # Engele girdi -> kaza, episode biter.
            self.agent_pos = yeni_pos
            reward = self.r_engel
            done = True
        elif yeni_pos == self.goal:
            # Hedefe ulaştı.
            self.agent_pos = yeni_pos
            reward = self.r_hedef
            done = True
        else:
            # Normal boş hücreye geçiş.
            self.agent_pos = yeni_pos
            reward = self.r_adim
            done = False

        # Çok uzun süredir dolaşıyorsa episode'u sonlandırıyorum.
        if self.step_count >= self.max_steps:
            done = True

        return self.agent_pos, reward, done

    def grid_matrisi(self):
        """Izgarayı sayısal matris olarak döndürür (görselleştirme için).
        0: boş, 1: engel, 2: başlangıç, 3: hedef.
        """
        m = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for (r, c) in self.obstacles:
            m[r, c] = 1
        m[self.start] = 2
        m[self.goal] = 3
        return m
