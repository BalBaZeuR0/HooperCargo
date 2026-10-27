# HooperCargo — Tasarım Belgesi

- **Tarih:** 2026-10-27
- **Repo:** https://github.com/BalBaZeuR0/HooperCargo
- **Süre:** bu hafta biter (2026-11-02'ye kadar)
- **Bağlam:** TÜBİTAK 1002 başvurusu (hızlı destek) için simülasyon kanıtı

## 1. Niyet: neden yapıyoruz

Proje fikri: ürünleri uçak ya da gemi yerine roketle, noktadan noktaya ve hızlı taşımak. Bunun dayanağı fırlatma maliyetindeki büyük düşüş. Kilogram başına yörüngeye erişim maliyeti 1960'tan 2025'e yaklaşık %96 azaldı (Terzi & Nicoli, 2026, *PNAS Nexus*, doi:10.1093/pnasnexus/pgag217). Kanıtların tamamı Obsidian'da: `LiteratürSearch/Roket_Maliyet_Dususu_Kanitlari.md`.

Bu repo fikrin teknik ayağını gösteriyor. Küçük ölçekli bir dikey kalkış-iniş roketi (**hopper**), her seferinde farklı ağırlık ve yerleşimde yük taşırken, rüzgârda ve yakıtı azalarak, rastgele bir iniş pistine ayaklarının üstünde güvenle iniyor. Bu davranışı pekiştirmeli öğrenme (RL, Reinforcement Learning: deneme-yanılma ile ödül toplayarak öğrenen yapay zekâ) ile eğitilmiş bir kontrolcü sağlıyor.

## 2. Hedefler

1. **Görev:** Hopper kalkış pistinden havalanır, rastgele konumdaki iniş pistine uçar ve **güvenli iniş** yapar.
2. **Yük:** Her uçuşta yükün **kütlesi** ve gövdedeki **konumu** rastgele seçilir. Toplam kütle, ağırlık merkezi ve eylemsizlik buna göre fiziksel olarak değişir. Yük yalnızca gövde içindeki makul bir kargo bölmesine konur. Bacak ucu, motor altı ya da gövde dışı gibi saçma yerlere konmaz.
3. **Yakıt:** Motor ittikçe yakıt azalır, buna bağlı olarak kütle ve ağırlık merkezi kayar. Yakıt biterse itki kesilir.
4. **Rüzgâr:** Her uçuşta rastgele yön ve şiddette yan rüzgâr ve ani esintiler olur.
5. **Algoritma:** PPO (Proximal Policy Optimization, en yaygın politika-gradyan RL yöntemi).
6. **Simülasyon:** MuJoCo MJX (MuJoCo'nun JAX ile ekran kartında binlerce kopyayı paralel koşturan sürümü).
7. **Takip:** Eğitim rlpanel ile canlı izlenir.
8. **Ölçme:** Binlerce rastgele uçuşta güvenli iniş oranı, iniş hatası ve yakıt tüketimi ölçülür. Bunlar yük kütlesine, ağırlık merkezi kaymasına ve rüzgâr şiddetine göre grafiklere dökülür. Grafikler 1002 başvurusuna konacak.

## 3. Hedef dışı (bu hafta yapılmayacak)

- Gerçek ölçekli (kilometrelerce) uçuş, atmosfer modeli, yörünge
- Gerçek donanım ve donanım-simülasyon döngüsü
- Haktan reposundaki dört aşamalı takla zinciri (tırmanma → takla → toparlanma → asılı kalma) ve masa-gimbal görevi
- SAC, TD3 gibi başka algoritmalarla kıyas
- Hareketli piste ya da eğimli zemine iniş

## 4. Anti-hedefler (teknik olarak çalışsa bile başarısızlık sayılır)

- Yükün fiziksel olmayan bir şekilde eklenmesi (örneğin ağırlık merkezi değişmeden sadece kütle sayısını artırmak) ya da gövde dışında saçma bir noktaya konması
- Hopper'ın devrilmesinin, ya da pist dışına inişin "başarı" sayılması
- Sonuçların eğitim sırasındaki ödülle raporlanması. Başarı oranı, eğitimden **ayrı** ve deterministik bir değerlendirme koşusunda ölçülmeli.
- Haktan'ın kodunun kaynak gösterilmeden kullanılması

## 5. Kararlar

### Kullanıcının kararları
- Taban model: GulogluHaktan/3DRocketRl reposundaki hopper (`assets/hopper_default.xml`), değiştirilerek kullanılır.
- Kalkıştan sonra rastgele bir yere güvenli iniş yapılır. **Güvenli iniş** = yere indikten sonra ayaklarının üstünde, **devrilmeden** durmak.
- Yükün kütlesi ve konumu her uçuşta değişir. Konum gövdede mantıklı bir yer olmalı.
- Yakıt tüketimi, rüzgâr ve ölçme (grafikler) dahil.
- Zemin Dünya, yani normal yer çekimi ve düz zemin. İniş pisti var.
- Algoritma PPO. Simülasyon MJX.
- Eğitim kullanıcının kendi bilgisayarında yapılacak (RTX 4050 Laptop, 6 GB). Nasıl çalıştırılacağına eğitim zamanı gelince karar verilecek.
- rlpanel eklenecek.
- Repo: BalBaZeuR0/HooperCargo. README, MTA standardına uygun ve İngilizce olacak.

### Claude'un kararları (kullanıcı değiştirebilir)

| Konu | Karar | Gerekçe |
|---|---|---|
| Mesafe | İniş pisti kalkıştan **20–100 m** uzakta, rastgele yönde. Müfredat 5 m'den başlar. | Bir metrelik hopper için gerçekçi, eğitimi hızlı tutar. "Küçültülmüş kargo roketi" olarak sunulur. |
| Pist | Kalkış ve iniş pistleri aynı düz zeminde, daire şeklinde, iniş pisti yarıçapı **1,5 m** | Bacak açıklığı 0,8 m, bu yarıçap makul bir hata payı bırakır. |
| Güvenli iniş ölçütü | (a) Dört ayağın hepsi yerde ve pistin içinde, (b) dokunuştan 2 s sonra gövde eğimi < 15°, (c) dokunma anında dikey hız < 2 m/s, yatay hız < 1 m/s | (b) kullanıcının "devrilmeden" tanımı. (a) ve (c) kargonun sağlam ve doğru yerde olması için eklenen ek şart. |
| Kargo bölmesi | Hopper gövde çerçevesinde (body frame) z = 0,00 … +0,35 m arası (gövde silindiri –0,15 … +0,445 m); eksenden yanal sapma ≤ 0,03 m (gövde yarıçapı 0,06 m) | Yük gövde içinde kalır, motor ve bacak bölgesine girmez; yakıt deposu altında (z ≈ –0,07 m). |
| Yük kütlesi | 0–0,8 kg (boş hopper 1,6 kg) | Tam itkide itki/ağırlık oranı > 1,3 kalır, yani kalkış her zaman mümkün. |
| Yakıt | Başta 0,6 kg, depo kargo bölmesinin altında sabit (z ≈ –0,07 m). Tüketim hızı = itki / (Isp · g₀), **Isp = 150 s** (hesap: en ağır araç, ~20 s'lik 100 m uçuşta ortalama ~32 N itkiyle 0,42 kg = deponun %70'i). Gerekirse değerlendirmeden sonra ince ayar. | Yakıt idaresini anlamlı kılar ama görevi imkânsız yapmaz. Isp (özgül itki): yakıtın verimliliği. |
| Kütle değişiminin uygulanması | Yük ve yakıt kütlesi, hopper gövdesinin `body_mass`, `body_ipos` ve `body_inertia` alanlarına paralel eksen teoremiyle eklenir. MJX modeli her ortam için durumda (state) taşınır ve her adımda `model.replace(...)` ile güncellenir. | Ağırlık merkezi ve eylemsizlik gerçekten değişir. Bu, sabit kütleye dış kuvvet eklemekten daha doğru. |
| Rüzgâr | Ortalama rüzgâr 0–6 m/s, rastgele yön. Üstüne Ornstein–Uhlenbeck esinti (zamanla yavaş değişen rastgele dalgalanma, σ ≈ 1,5 m/s). Kuvvet = ½·ρ·Cd·A·\|v_rel\|·v_rel, havaya göre bağıl hızla, gövde basınç merkezine uygulanır. | Haktan'daki doğrusal sönümün yerine fiziksel bir sürüklenme kuvveti. |
| Gözlem (policy'nin gördüğü) | Gövde konumu ve hızı, iniş pistine göre bağıl konum, yönelim, açısal hız, TVC açısı, kalan yakıt, **yük kütlesi** (kargo manifestosunda yazar), ayakların yere değip değmediği. **Yük konumu ve rüzgâr gözlenmez.** | Gerçekte yükün ağırlığı bilinir ama tam yerleşimi ve rüzgâr bilinmez. Kontrolcü bunlara karşı sağlam olmak zorunda. |
| Eylem | Gaz (0–1) + iki TVC açısı (±20°) | Haktan'ın eylem uzayıyla aynı. Tek motor yuvarlanmayı (roll) kontrol edemez. İniş simetrik bacaklarla yapıldığı için bu sorun değil. |
| Temas | Yalnız 4 ayak ile zemin arasındaki temas açık, diğer bütün geomlar kapalı. | Haktan temasın hepsini kapatmıştı. Bizde inişin kendisi görevin parçası, ayak temasları da ucuz. |
| Motor kesme | İlk yere değmede (havalandıktan sonra) motor kesilir. Sonrası pasif: hopper ya ayakta kalır ya devrilir. | Gerçek iniş araçları da dokunuşta motoru keser; tekrar zıplayıp "hile" yapmayı önler. |
| Bölüm sonu | Güvenli iniş + 2 s yerleşme, devrilme (eğim > 60°), yere gövdeyle çarpma, sınır dışı, ya da süre aşımı (40 s) | |
| PPO uygulaması | Brax'ın kendi PPO'su (`brax.training.agents.ppo`) | MJX ortamı saf JAX olduğu için en hızlısı bu. Haktan zaten Brax kullanıyor. |
| rlpanel | Brax'ın `progress_fn`'i içinden `panel.log` / `panel.progress` çağrılır. Bağımlılık: `rlpanel @ git+https://github.com/BalBaZeuR0/RLpanel@v0.1.0` (sb3 eki yok). Testlerde `RLPANEL_DISABLE=1`. | |
| Çalışma ortamı | Kod hem JAX-CPU'da (Windows'ta doğrudan; geliştirme ve testler için) hem JAX-GPU'da (eğitim) çalışacak şekilde yazılır. | JAX'in ekran kartı desteği yerel Windows'ta yok, GPU için WSL2 Ubuntu gerekiyor. Bilgisayarda şu an yalnız docker-desktop dağıtımı var. Bu karar eğitim günü verilecek. |
| Müfredat | 3 kademe: (1) 5–20 m, hafif yük, rüzgâr yok; (2) 20–60 m, yükün tam aralığı, hafif rüzgâr; (3) 20–100 m, tam rüzgâr | Zor görevi doğrudan öğretmek PPO'da genellikle çöker. |

## 5b. Ölçek ve oranlar (kullanıcı onayladı, 2026-10-27)

Hopper Haktan'ın modelinin boyutunda kalır (yaklaşık 1 m, boş kütle 1,6 kg, en fazla 45 N itki). Bu, gerçek öğrenci/hobi TVC hopper'larıyla aynı sınıftır. Proje **küçük ölçekli teknoloji gösterici** olarak sunulur; örnek olarak SpaceX'in Falcon 9'dan önceki Grasshopper deneyleri verilebilir. Tam ölçek, 1002 sonrası bir 1001 başvurusunun konusu olur.

İnandırıcılığı mutlak boyut değil, **gerçekçi oranlar** sağlar:

| Oran | Değer | Not |
|---|---|---|
| Kalkış itki/ağırlık, en hafif (yük 0 + yakıt 0,6 = 2,2 kg) | ≈ 2,1 | |
| Kalkış itki/ağırlık, en ağır (yük 0,8 + yakıt 0,6 = 3,0 kg) | ≈ 1,5 | Gerçek roketlerin kalkış aralığı (~1,3–1,5) |
| Yük / boş kütle | 0–0,5 | |
| İniş pisti yarıçapı / bacak açıklığı | ≈ 1,9 (1,5 m / 0,8 m) | |
| Dokunma hızı sınırı | < 2 m/s dikey | Bacak boyuna (0,36 m) göre sert ama kırmayan iniş |

**Parametrik ölçek:** Model XML'i Python'dan tek bir `scale` katsayısıyla üretilir. Uzunluk L ile ölçeklenir, kütle L³ ile, itki L³ ile (itki/ağırlık sabit kalır), eylemsizlik L⁵ ile, mesafeler L ile. Rüzgâr hızı ve dokunma hızı sınırı Froude benzerliğine göre √L ile ölçeklenir (Froude benzerliği: yer çekimi altında hareket eden sistemlerde hızın boyun karekökü ile ölçeklendiği kural). Varsayılan `scale = 1`. **İsteğe bağlı, zaman kalırsa:** aynı yöntemle `scale = 2` aracı da eğitilip değerlendirilir. Bu, "yöntem ölçekten bağımsız" iddiasını sayıyla destekler.

## 6. Ölçme (1002'ye girecek çıktılar)

- **Değerlendirme seti:** Sabit tohumla (seed) 1000 uçuş. Yük, ağırlık merkezi, rüzgâr ve pist konumu tam aralıkta rastgele.
- **Ana sayılar:** güvenli iniş oranı (%), iniş hatası (pist merkezine uzaklık, m), harcanan yakıt (%), uçuş süresi (s).
- **Grafikler** (`plots/`):
  1. Yük kütlesine göre başarı oranı
  2. Ağırlık merkezi kaymasına göre başarı oranı
  3. Rüzgâr şiddetine göre başarı oranı
  4. Örnek uçuş yörüngeleri (yandan ve üstten görünüş)
  5. Eğitim eğrisi (ödül ve başarı oranı)
- **Karşılaştırma** (zaman kalırsa): Yük **değiştirilmeden** (sabit kütle) eğitilmiş bir politika aynı değerlendirme setinde test edilir. Bu, "yükü bilmek ve rastgeleleştirmek gerekli" iddiasını sayıyla gösterir.
- **Ölçek** (zaman kalırsa): `scale = 2` hopper'da aynı ana sayılar ve oranlar, `scale = 1` ile yan yana (§5b).
- **Video:** En az bir başarılı uçuş GIF'i (`assets/demo.gif`, < 10 MB).

## 7. Repo yapısı (öneri)

```
HooperCargo/
├── assets/hooper_cargo.xml      # Haktan modelinden türetilmiş şablon: kargo bölmesi, yakıt deposu, iniş pisti, geniş zemin
├── hoopercargo/model.py         # şablondan `scale` katsayısına göre XML üretir (§5b)
├── hoopercargo/
│   ├── env.py                   # MJX ortamı (saf JAX)
│   ├── payload.py               # yük ve yakıtın kütle/ağırlık merkezi/eylemsizlik hesabı
│   ├── wind.py                  # rüzgâr ve sürüklenme kuvveti
│   ├── train.py                 # Brax PPO + rlpanel + müfredat
│   └── evaluate.py              # 1000 uçuşluk değerlendirme → CSV
├── scripts/                     # grafik ve video üretimi
├── plots/  tests/  docs/
├── README.md                    # MTA şablonu, İngilizce; Haktan reposuna atıf
└── pyproject.toml
```

## 8. Doğrulama (testler)

- Yük ekleyince toplam kütle ve ağırlık merkezi, elle hesaplanan değerle aynı çıkıyor.
- Yakıt tükenince itki sıfır oluyor.
- Rüzgâr yokken ve hız sıfırken sürüklenme kuvveti sıfır.
- Sıfır gazla bırakılan hopper düşüyor. Elle verilen hover gazıyla yaklaşık havada kalıyor.
- Güvenli iniş ölçütü: dik ve yavaş konan hopper başarılı sayılıyor. Yan yatırılarak bırakılan hopper başarısız sayılıyor.
- `scale = 2` modelde kütle 8 katı, itki/ağırlık oranı `scale = 1` ile aynı çıkıyor.
- Ortam `jax.jit` ve `jax.vmap` ile derleniyor ve CPU'da küçük bir PPO "duman testi" (birkaç bin adım) hatasız koşuyor.

## 9. Haftalık takvim (tahmini)

| Gün | İş |
|---|---|
| 1–2 | Model + ortam + testler (CPU) |
| 2 | Çalışma ortamı kararı (WSL2/GPU ya da CPU), kısa deneme eğitimi |
| 3–5 | Müfredatlı PPO eğitimi, ödül ayarı |
| 6 | Değerlendirme, grafikler, video; zaman kalırsa `scale = 2` eğitimi |
| 7 | README + 1002 metni için sonuç paragrafı |

## 10. Uygulayıcıya bırakılanlar

- Ödül fonksiyonunun ağırlıkları ve şekli (yaklaşma, hız uyumu, dik kalma, yakıt cezası, iniş bonusu)
- PPO hiperparametreleri, paralel ortam sayısı (GPU belleğine göre)
- Yakıt deposunun tam konumu ve Isp ince ayarı (§5'teki hedefe göre)
- Görselleştirme aracı (MuJoCo renderer ile GIF)
