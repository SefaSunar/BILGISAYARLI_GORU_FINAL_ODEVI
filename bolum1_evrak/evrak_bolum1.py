import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


img = cv2.imread("evrak.jpg")  
if img is None:
    print("HATA: Fotoğraf bulunamadı! 'evrak.jpg' dosyasının bu klasörde olduğuna emin ol.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

_, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
k = np.ones((15, 15), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest = max(contours, key=cv2.contourArea)

mask = np.zeros((h, w), dtype=np.uint8)
cv2.drawContours(mask, [biggest], -1, 255, -1)
masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)


enhanced_rgb = masked.copy()
enhanced_rgb[mask == 0] = [255, 255, 255]


epsilon = 0.02 * cv2.arcLength(biggest, True)
approx = cv2.approxPolyDP(biggest, epsilon, True)
print(f"Tespit edilen köşe sayısı: {len(approx)}")

if len(approx) != 4:
    hull = cv2.convexHull(biggest)
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    print(f"Hull sonrası köşe: {len(approx)}")

pts = approx.reshape(-1, 2).astype(np.float32)
rect = np.zeros((4, 2), dtype=np.float32)
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]    # sol-üst
rect[2] = pts[np.argmax(s)]    # sağ-alt
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)] # sağ-üst
rect[3] = pts[np.argmax(diff)] # sol-alt

target_w, target_h = 794, 1123
dst = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(img, M, (target_w, target_h))
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
warped_enhanced = clahe.apply(warped_gray)
_, warped_binary = cv2.threshold(warped_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

corners_img = img_rgb.copy()
for pt in rect:
    cv2.circle(corners_img, (int(pt[0]), int(pt[1])), 20, (255, 0, 0), -1)
cv2.polylines(corners_img, [rect.astype(int)], True, (255, 0, 0), 4)


fig, axes = plt.subplots(2, 3, figsize=(18, 13))


fig.text(0.5, 0.98, "1. BÖLÜM — Evrak İşleme",
         fontsize=16, fontweight='bold', ha='center', va='top')

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Adım 1: Orijinal Görüntü", fontsize=11, fontweight='bold', pad=8)
axes[0, 0].axis('off')

axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title("Adım 2: Kağıt Maskesi", fontsize=11, fontweight='bold', pad=8)
axes[0, 1].axis('off')

axes[0, 2].imshow(masked)
axes[0, 2].set_title("Adım 3: Maskelenmiş Görüntü", fontsize=11, fontweight='bold', pad=8)
axes[0, 2].axis('off')

axes[1, 0].imshow(corners_img)
axes[1, 0].set_title("Adım 4: Köşe Noktaları Tespiti", fontsize=11, fontweight='bold', pad=8)
axes[1, 0].axis('off')

axes[1, 1].imshow(warped_rgb)
axes[1, 1].set_title("Adım 5: Perspektif Düzeltme", fontsize=11, fontweight='bold', pad=8)
axes[1, 1].axis('off')

axes[1, 2].imshow(warped_binary, cmap='gray')
axes[1, 2].set_title("Adım 6: Kontrast İyileştirme (CLAHE + Otsu)", fontsize=11, fontweight='bold', pad=8)
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  
plt.savefig("evrak_sonuc.jpg", dpi=150, bbox_inches='tight')
cv2.imwrite("evrak_duzeltilmis.jpg", warped)

print("Sonuçlar kaydedildi: 'evrak_sonuc.jpg' ve 'evrak_duzeltilmis.jpg'")
plt.show()