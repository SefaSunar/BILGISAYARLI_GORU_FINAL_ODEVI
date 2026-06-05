# Bilgisayarlı Görüye Giriş - Final Ödevi

## Grup Üyeleri

| İsim | Öğrenci No | GitHub |
|------|-----------|--------|
| Sefa Sunar | 24060772 | [@SefaSunar](https://github.com/SefaSunar) |
| Bahadır Arabacı | 22060322 | [@BahadirArabaci](https://github.com/22060322b) |

# Bilgisayarlı Görüye Giriş - Final Ödevi

Bu repoda Bilgisayarlı Görüye Giriş dersinde yaptığımız üç farklı görüntü işleme uygulamasının kodları ve sonuçları yer almaktadır.

## Klasör Yapısı

- **bolum1_evrak** → Kamera ile çekilmiş bir belgenin işlenmesi
- **bolum2_optik** → Optik cevap kağıtlarının okunması
- **bolum3_misir** → Mısır tanelerinin sayılması

## Bölüm 1 - Evrak İşleme

Siyah zemin üzerine bırakılan bir kağıt belgesi kamera ile çekilmiştir. Görüntü üzerinde sırasıyla şu işlemler yapılmıştır:

- Gri tonlama ve Gaussian bulanıklaştırma
- Eşikleme ile kağıdın zeminden ayrılması ve maskeleme
- approxPolyDP ile köşe noktalarının tespiti
- getPerspectiveTransform ile perspektif bozukluğunun düzeltilmesi
- CLAHE ve Otsu eşiklemesi ile kontrast iyileştirme

## Bölüm 2 - Optik Form Okuma

20 soruluk çoktan seçmeli optik cevap kağıtları işlenmiştir. Cevap anahtarı da dahil olmak üzere 11 form kamera ile çekilmiş ve işlenmiştir.

- Hough Circle Transform ile baloncuk tespiti
- Piksel parlaklık analizi ile dolu/boş ayrımı
- K-means ile öğrenci numarası sütunlarının gruplandırılması
- Cevap anahtarı da formdan otomatik okunmuştur
- Her öğrenci için doğru, yanlış, boş ve net puan hesaplanmıştır

## Bölüm 3 - Mısır Sayma

Beyaz zemin üzerine dökülen mısır taneleri sayılmıştır.

- HSV renk uzayında sarı-turuncu renk maskesi
- Morfolojik işlemlerle gürültü temizleme
- Mesafe dönüşümü ile her tanenin merkezinin bulunması
- peak_local_max ile lokal maksimum tespiti

## Kullanılan Kütüphaneler
pip install opencv-python numpy matplotlib scikit-image scipy

## Geliştirme Ortamı

- Python 3.14.3
- OpenCV 4.13.0
- Windows 11