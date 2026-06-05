import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

CEVAP_ANAHTARI_DOSYA = "CevapAnahtari.jpeg"


DOSYALAR = [
    ("BerkayAnakli_22060372.jpeg",   "Berkay Anaklı",   None),
    ("BatuhanYilmaz_22060363.jpeg",  "Batuhan Yılmaz",  None),
    ("NurlanGuliyev_22060003.jpeg",  "Nurlan Guliyev",  None),
    ("NagihanZan_22060682.jpeg",     "Nagihan Zan",     None),
    ("OzgurErcan_21060996.jpeg",     "Özgür Ercan",     None),
    ("TubaSarikaya_22060374.jpeg",   "Tuba Sarıkaya",   None),
    ("DuyguKaya_22060369.jpeg",      "Duygu Kaya",      None),
    ("NurayNart_22060393.jpeg",      "Nuray Nart",      None),
    ("GokdenizCoban_22060350.jpeg",  "Gökdeniz Çoban",  None),
    ("RevasAkin_22060396.jpeg",      "Revas Akın",      None),
]

SECENEKLER = ['A', 'B', 'C', 'D', 'E']


def perspektif_duzelt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    k = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True)
    if len(approx) != 4:
        hull = cv2.convexHull(biggest)
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    dst = np.array([[0, 0], [794, 0], [794, 1123], [0, 1123]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (794, 1123))


def ogr_no_oku(warped, result):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    y1, y2, x1, x2 = 240, 550, 60, 760
    bolge = gray[y1:y2, x1:x2]
    adaptive = cv2.adaptiveThreshold(bolge, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    daireler = []
    for cnt in contours:    
        area = cv2.contourArea(cnt)
        if area < 50 or area > 800: continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), int(r)
        if r == 0: continue
        if area / (np.pi * r * r) > 0.55:
            daireler.append((cx + x1, cy + y1, r))

    if not daireler:
        return "00000000", result

    
    xs = np.array([d[0] for d in daireler], dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(xs, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    order = np.argsort(centers.flatten())
    label_map = {old: new for new, old in enumerate(order)}
    sutunlar = [[] for _ in range(8)]
    for (x, y, r), lbl in zip(daireler, labels.flatten()):
        sutunlar[label_map[lbl]].append((x, y, r))

    genel_y_min = min(d[1] for s in sutunlar for d in s)
    adim = 26
    ogr_no = ""

    for sutun in sutunlar:
        sutun_s = sorted(sutun, key=lambda d: d[1])
        if not sutun_s:
            ogr_no += "0"; continue
        for x, y, r in sutun_s:
            cv2.circle(result, (x, y), r, (0, 180, 0), 2)
        y_min_sutun = min(d[1] for d in sutun_s)
        offset = round((y_min_sutun - genel_y_min) / adim)
        bulundu = False
        for sira, (x, y, r) in enumerate(sutun_s):
            roi = gray[max(0, y-r):y+r, max(0, x-r):x+r]
            if roi.size == 0: continue
            if np.mean(roi) < 115:
                rakam = max(0, min(9, sira + offset))
                ogr_no += str(rakam)
                cv2.circle(result, (x, y), r, (255, 0, 0), -1)
                cv2.circle(result, (x, y), r, (150, 0, 0), 2)
                cv2.putText(result, str(rakam), (x-5, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                bulundu = True; break
        if not bulundu:
            ogr_no += "0"

    return ogr_no, result


def cevaplari_oku(warped, result, esik=110):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    CVP_Y1, CVP_Y2 = 555, 985
    SOL_X1, SOL_X2 = 60, 390
    SAG_X1, SAG_X2 = 400, 760

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=16, param1=50, param2=22,
        minRadius=8, maxRadius=18)
    if circles is None:
        return {}
    circles = np.round(circles[0]).astype(int)

    sol = [(x, y, r) for x, y, r in circles if SOL_X1 < x < SOL_X2 and CVP_Y1 < y < CVP_Y2]
    sag = [(x, y, r) for x, y, r in circles if SAG_X1 < x < SAG_X2 and CVP_Y1 < y < CVP_Y2]

    def isle(daireler, baslangic):
        if not daireler: return {}
        daireler = sorted(daireler, key=lambda d: d[1])
        satirlar, satir = [], [daireler[0]]
        for d in daireler[1:]:
            if abs(d[1] - satir[-1][1]) < 15: satir.append(d)
            else:
                satirlar.append(sorted(satir, key=lambda d: d[0]))
                satir = [d]
        satirlar.append(sorted(satir, key=lambda d: d[0]))
        cevaplar = {}
        soru = baslangic
        for satir in satirlar:
            if len(satir) != 5: continue
            for idx, (x, y, r) in enumerate(satir):
                maske = np.zeros_like(gray)
                cv2.circle(maske, (x, y), int(r * 0.7), 255, -1)
                piksel = gray[maske == 255]
                if len(piksel) > 0 and np.mean(piksel) < esik:
                    cevaplar[soru] = SECENEKLER[idx]
                    cv2.circle(result, (x, y), r, (255, 0, 0), -1)
                    cv2.circle(result, (x, y), r+2, (150, 0, 0), 2)
                    cv2.putText(result, str(soru), (x-8, y+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                else:
                    cv2.circle(result, (x, y), r, (0, 180, 0), 2)
            soru += 2
            if soru > 20: break
        return cevaplar

    return {**isle(sol, 1), **isle(sag, 2)}


def sonuc_hesapla(cevaplar, anahtar):
    d = y_ = b = 0
    for s in range(1, 21):
        c = cevaplar.get(s)
        if c is None: b += 1
        elif c == anahtar.get(s): d += 1
        else: y_ += 1
    return d, y_, b


def gorsel_olustur(result, isim, ogr_no, cevaplar, anahtar, dogru, yanlis, bos, net, idx):
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor('white')

    ax_img = fig.add_axes([0.01, 0.10, 0.55, 0.85])
    ax_img.imshow(result)
    ax_img.set_title("Kırmızı=İşaretli  Yeşil=Boş", fontsize=11, fontweight='bold')
    ax_img.axis('off')

    ax_info = fig.add_axes([0.58, 0.75, 0.40, 0.20])
    ax_info.axis('off')
    ax_info.text(0.05, 0.85, f"Ad Soyad   : {isim}", fontsize=11, fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.05, 0.62, f"Ogrenci No : {ogr_no}", fontsize=11, transform=ax_info.transAxes)
    ax_info.text(0.05, 0.39, f"Dogru:{dogru}  Yanlis:{yanlis}  Bos:{bos}",
                fontsize=11, fontweight='bold', color='darkgreen', transform=ax_info.transAxes)
    ax_info.text(0.05, 0.16, f"Net Puan   : {net}", fontsize=11, transform=ax_info.transAxes)

    ax_tbl = fig.add_axes([0.58, 0.10, 0.40, 0.62])
    ax_tbl.axis('off')
    tablo_veri, renkler = [], []
    for s in range(1, 21):
        ogr = cevaplar.get(s, "-")
        ans = anahtar.get(s, "-")
        if ogr == "-":     durum = "BOS";      renk = "#fff3cd"
        elif ogr == ans:   durum = "DOGRU";    renk = "#d4edda"
        else:              durum = "YANLIS";   renk = "#f8d7da"
        tablo_veri.append([str(s), ogr, ans, durum])
        renkler.append(renk)

    tbl = ax_tbl.table(cellText=tablo_veri,
                       colLabels=["Soru", "Ogrenci", "Anahtar", "Durum"],
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.15)
    for j in range(4):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for row, renk in enumerate(renkler):
        for j in range(4): tbl[row+1, j].set_facecolor(renk)

    fig.suptitle(f"2. BOLUM - {isim}  ({idx}/10)", fontsize=13, fontweight='bold', y=0.99)
    plt.savefig(f"optik_ogr_{idx:02d}.jpg", dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> optik_ogr_{idx:02d}.jpg kaydedildi")



print("Cevap anahtari okunuyor...")
img_ca = cv2.imread(CEVAP_ANAHTARI_DOSYA)
if img_ca is None:
    print(f"HATA: {CEVAP_ANAHTARI_DOSYA} bulunamadi!")
    exit()

warped_ca = perspektif_duzelt(img_ca)
result_ca = cv2.cvtColor(warped_ca, cv2.COLOR_BGR2RGB).copy()
CEVAP_ANAHTARI = cevaplari_oku(warped_ca, result_ca)
print(f"Cevap anahtari: {CEVAP_ANAHTARI}")


fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0.01, 0.05, 0.55, 0.90])
ax.imshow(result_ca)
ax.set_title("Cevap Anahtari Formu", fontsize=12, fontweight='bold')
ax.axis('off')
ax2 = fig.add_axes([0.58, 0.10, 0.40, 0.80])
ax2.axis('off')
veri_ca = [[str(s), CEVAP_ANAHTARI.get(s, "-")] for s in range(1, 21)]
tbl_ca = ax2.table(cellText=veri_ca, colLabels=["Soru", "Dogru Cevap"],
                   loc='center', cellLoc='center')
tbl_ca.auto_set_font_size(False); tbl_ca.set_fontsize(10); tbl_ca.scale(1, 1.7)
for j in range(2):
    tbl_ca[0, j].set_facecolor('#2c3e50')
    tbl_ca[0, j].set_text_props(color='white', fontweight='bold')
fig.suptitle("2. BOLUM - Cevap Anahtari", fontsize=13, fontweight='bold', y=0.99)
plt.savefig("optik_cevap_anahtari.jpg", dpi=130, bbox_inches='tight', facecolor='white')
plt.close()
print("Cevap anahtari gorseli kaydedildi.")


tum_sonuclar = []
print("\n" + "="*65)
print(f"{'Ad Soyad':<22} {'Ogr.No':^10} {'D':>4} {'Y':>4} {'B':>4} {'Net':>6}")
print("="*65)

for idx, (dosya, isim, ogr_no_elle) in enumerate(DOSYALAR, 1):
    if not os.path.exists(dosya):
        print(f"HATA: {dosya} bulunamadi!"); continue

    img = cv2.imread(dosya)
    warped = perspektif_duzelt(img)
    result = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).copy()
    ogr_no, result = ogr_no_oku(warped, result)

    cevaplar = cevaplari_oku(warped, result, esik=esik)
    dogru, yanlis, bos = sonuc_hesapla(cevaplar, CEVAP_ANAHTARI)
    net = round(dogru - yanlis / 4, 2)
    tum_sonuclar.append((isim, ogr_no, dogru, yanlis, bos, net))
    print(f"{isim:<22} {ogr_no:^10} {dogru:>4} {yanlis:>4} {bos:>4} {net:>6}")
    gorsel_olustur(result, isim, ogr_no, cevaplar, CEVAP_ANAHTARI, dogru, yanlis, bos, net, idx)

print("="*65)

fig, ax = plt.subplots(figsize=(13, 6))
ax.axis('off')
veri = [[str(i+1), ad, no, str(d), str(y), str(b), str(n)]
        for i, (ad, no, d, y, b, n) in enumerate(tum_sonuclar)]
tbl2 = ax.table(cellText=veri,
                colLabels=["#", "Ad Soyad", "Ogrenci No", "Dogru", "Yanlis", "Bos", "Net"],
                loc='center', cellLoc='center')
tbl2.auto_set_font_size(False); tbl2.set_fontsize(11); tbl2.scale(1.2, 2.2)
for j in range(7):
    tbl2[0, j].set_facecolor('#2c3e50')
    tbl2[0, j].set_text_props(color='white', fontweight='bold')
for i in range(len(tum_sonuclar)):
    for j in range(7):
        tbl2[i+1, j].set_facecolor('#f2f2f2' if i % 2 == 0 else 'white')

plt.title("2. BOLUM - Optik Okuma Ozet Tablosu", fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("optik_ozet_tablo.jpg", dpi=150, bbox_inches='tight')
plt.close()
print("\nOzet tablo: optik_ozet_tablo.jpg")
print("Tum gorseller hazir!")