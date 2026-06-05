import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage

# ── 1. Resmi yükle ────────────────────────────────────────
img = cv2.imread("misir.jpg")  # fotoğrafının adını buraya yaz
if img is None:
    print("HATA: Fotoğraf bulunamadı! 'misir.jpg' dosyasının bu klasörde olduğuna emin ol.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── 2. HSV renk maskesi ───────────────────────────────────
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Sarı-turuncu mısır rengi
mask_sari = cv2.inRange(hsv, np.array([8,  50, 80]),  np.array([32, 255, 240]))
# Bej/krem rengi (mısırın beyaz ucu)
mask_bej  = cv2.inRange(hsv, np.array([15, 15, 110]), np.array([40, 60,  220]))
misir_mask = cv2.bitwise_or(mask_sari, mask_bej)

# Gri zemin piksellerini çıkar
gray_mask = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 25, 255]))
misir_mask = cv2.bitwise_and(misir_mask, cv2.bitwise_not(gray_mask))

# ── 3. Morfolojik işlemler ────────────────────────────────
k5 = np.ones((5, 5), np.uint8)
k3 = np.ones((3, 3), np.uint8)
misir_mask = cv2.morphologyEx(misir_mask, cv2.MORPH_CLOSE, k5, iterations=3)
misir_mask = cv2.morphologyEx(misir_mask, cv2.MORPH_OPEN,  k3, iterations=1)

# ── 4. Mesafe dönüşümü + lokal maksimumlar ────────────────
dist = ndimage.distance_transform_edt(misir_mask)
coords = peak_local_max(dist, min_distance=28, labels=misir_mask)
count = len(coords)

print(f"Tespit edilen mısır sayısı: {count}")

# ── 5. Sonuç görseli ──────────────────────────────────────
result = img_rgb.copy()
for (r, c) in coords:
    cv2.circle(result, (c, r), 20, (255, 0, 0), 2)
    cv2.circle(result, (c, r),  4, (255, 0, 0), -1)

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

axes[0].imshow(img_rgb)
axes[0].set_title("Orijinal Görüntü", fontsize=13, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(misir_mask, cmap='gray')
axes[1].set_title("Mısır Maskesi", fontsize=13, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(result)
axes[2].set_title(f"Tespit Edilen Mısır Sayısı: {count}", fontsize=13, fontweight='bold', color='darkgreen')
axes[2].axis('off')

plt.suptitle("3. BÖLÜM — Tanecik Sayma (Mısır)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("misir_sonuc.jpg", dpi=150, bbox_inches='tight')
print("Sonuç 'misir_sonuc.jpg' olarak kaydedildi!")
plt.show()