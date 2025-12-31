# Machine Learning Algorithms in C

## Optimizasyon Algoritmaları ile Görüntü Sınıflandırma

Bu proje, görüntü işleme ve makine öğrenmesi teknikleri kullanılarak oluşturulmuş bir sınıflandırma modelidir. Kedi ve köpek görsellerinden oluşan özel bir veri seti üzerinde **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)** ve **ADAM** optimizasyon algoritmalarının performansları karşılaştırılmıştır.

## 📋 İçindekiler

- [Proje Özeti](#proje-özeti)
- [Veri Seti Hazırlığı](#1-görselleri-veri-setine-dönüştürme)
- [Veri İşleme ve Ayrıştırma](#2-veri-setini-dönüştürme)
- [Kullanılan Algoritmalar](#kullanılan-algoritmalar)
- [Sonuçlar ve Görseller](#sonuçlar-ve-görseller)

---

## Proje Özeti

Projede iki sınıftan (Kedi ve Köpek) oluşan ve her bir sınıftan 101 adet gri tonlamalı görsel içeren bir veri seti oluşturulmuştur. Bu veri seti karıştırıldıktan sonra eğitim ve test kümelerine ayrılmış ve farklı optimizasyon algoritmaları ile modeller eğitilmiştir.

## 1. Görselleri Veri Setine Dönüştürme

Veri seti oluşturma süreci şu adımları içerir:

* **Kaynak:** Kedi ve Köpek sınıfları.
* **Miktar:** Her sınıf için 101 adet görsel (Toplam 202 görsel).
* **Boyutlandırma:** Tüm görseller **50x50 piksel** boyutuna getirilmiş ve **gri tonlamaya** (grayscale) çevrilmiştir.
* **Vektörleştirme:** Her görsel düzleştirilerek (flatten) piksel değerleri alınmış ve sona 1 adet **bias** değeri eklenmiştir. Sonuçta **1x2501** boyutunda vektörler elde edilmiştir.
* **Normalizasyon:** 0-255 arasındaki piksel değerleri **0-1 aralığına** indirgenmiştir.
* **Kayıt:** İşlenen veriler `imageData.csv` dosyasına kaydedilmiştir.

## 2. Veri Setini Dönüştürme

Modelin genelleme yeteneğini artırmak için veri seti üzerinde şu işlemler yapılmıştır:

1. `imageData.csv` dosyasından çekilen vektörler rastgele **karıştırılmıştır (shuffling)**.
2. Veri seti aşağıdaki oranlarda ikiye ayrılmıştır:
    * **Eğitim Kümesi (Train):** %20
    * **Test Kümesi (Test):** %80

## Kullanılan Algoritmalar

Model eğitiminde aşağıdaki optimizasyon algoritmaları kullanılmış ve kıyaslanmıştır:

* Gradient Descent (GD)
* Stochastic Gradient Descent (SGD)
* Adaptive Moment Estimation (ADAM)

---

## Sonuçlar ve Görseller

Aşağıda eğitim süreci sonucunda elde edilen maliyet (cost) grafikleri ve sınıflandırma örnekleri yer almaktadır.

### Eğitim Maliyet Grafikleri

![Maliyet Grafiği](gorseller/cost_graph.png)

### Sınıflandırma Örnekleri

![Örnek Sonuçlar](gorseller/ornek_sonuc.png)