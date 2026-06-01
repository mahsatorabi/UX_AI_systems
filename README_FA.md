# راهنمای فارسی — مخزن مقاله UX × هوش مصنوعی مولد

این پوشه (`UX`) همان محتوایی است که باید در مخزن گیت‌هاب منتشر شود:  
[https://github.com/mahsatorabi/UX_AI_systems](https://github.com/mahsatorabi/UX_AI_systems)

## چه چیزهایی برای مقاله آماده است؟

| بخش | مسیر | کاربرد در مقاله |
|-----|------|------------------|
| **شکل‌های RQ1** | `outputs/figures/rq1_fig_*.png` یا `.pdf` | روند انتشار و رشد تجمعی |
| **شکل‌های RQ2** | `outputs/figures/rq2_fig_*.png` | موضوعات NMF، شبکه کلیدواژه، تکامل موضوعی |
| **شکل‌های RQ3** | `outputs/figures/rq3_fig_*.png` | واژه‌های نوظهور، عبارات burst، رشد موضوعات |
| **جداول** | `outputs/rq*.csv` و `rq1_growth_phases.json` | ساخت جداول Results |
| **روش‌شناسی** | `METHODOLOGY.md` | بخش Methods |
| **فهرست شکل/جدول** | `FIGURES_AND_TABLES.md` | نقشهٔ دقیق فایل → شماره شکل در مقاله |

## آپلود به گیت‌هاب (یک‌بار)

در PowerShell، از داخل همین پوشه:

```powershell
cd "d:\Uni of Birjand\articles\UX"
git init
git add .
git commit -m "Add reproducibility package: code, data, figures, and outputs for UX×GenAI bibliometric study"
git branch -M main
git remote add origin https://github.com/mahsatorabi/UX_AI_systems.git
git push -u origin main
```

اگر مخزن از قبل README دارد و تاریخچه دارد:

```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## اجرای مجدد تحلیل

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
```

## لینک در مقاله (Data availability)

متن پیشنهادی:

> Data and analysis code are openly available at  
> https://github.com/mahsatorabi/UX_AI_systems  
> (Scopus export, preprocessed corpus, scripts, figures, and supplementary tables).

## نکتهٔ حقوقی Scopus

فایل `data.csv` خروجی Scopus است. در مقاله ذکر کنید که داده تحت قوانین Elsevier است و مخزن فقط برای **بازتولیدپذیری** همان مطالعه است.
