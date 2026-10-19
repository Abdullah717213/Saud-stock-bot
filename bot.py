import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# إعدادات البوت - منهج فيصل
# ==========================================

WATCHLIST_FILE = "watchlist.csv"
PERIOD = "1y"
INTERVAL = "1d"

RSI_MIN = 23
RSI_MAX = 27

SUPPORT_WINDOW = 20
RESISTANCE_WINDOW = 60

MIN_DATA_ROWS = 210


# ==========================================
# أدوات مساعدة
# ==========================================

def number(value):
    try:
        result = float(value)
        if np.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return None


def fmt(value):
    value = number(value)
    return "غير متوفر" if value is None else f"{value:.4f}"


def discover_stocks():
url = "https://api.nasdaq.com/api/screener/stocks"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/"
}
symbols = set()
for exchange in ["NASDAQ", "NYSE", "AMEX"]:
    try:
        response = requests.get(
            url,
            params={
                "tableonly": "true",
                "limit": "5000",
                "offset": "0",
                "exchange": exchange,
                "download": "true"
            },
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json().get("data", {})
        rows = data.get("rows", [])
        for row in rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            price_text = str(row.get("lastsale", ""))
            try:
                price = float(
                    price_text.replace("$", "").replace(",", "")
                )
            except (ValueError, TypeError):
                continue
            if symbol and 1 <= price <= 6:
                symbols.add(symbol)
        print(
            f"تم اكتشاف {len(symbols)} سهمًا حتى الآن"
        )
    except Exception as exc:
        print(f"تعذر جلب أسهم {exchange}: {exc}")
result = sorted(symbols)
print(f"إجمالي الأسهم المكتشفة بسعر 1–6 دولارات: {len(result)}")
return result


def download_daily(symbol):
    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            print(f"لا توجد بيانات للسهم {symbol}")
            return None

        # معالجة الأعمدة متعددة المستويات في yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]

        df.columns = [str(c).title() for c in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required):
            print(f"بيانات OHLCV غير مكتملة للسهم {symbol}")
            return None

        df = df[required].copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        if len(df) < 60:
            print(f"بيانات تاريخية غير كافية للسهم {symbol}")
            return None

        return df

    except Exception as exc:
        print(f"تعذر تحميل {symbol}: {exc}")
        return None


# ==========================================
# المؤشرات الفنية
# ==========================================

def add_indicators(df):
    df = df.copy()
    close = df["Close"]

    # المتوسطات المتحركة
    for period in [20, 30, 50, 200]:
        df[f"EMA{period}"] = close.ewm(
            span=period,
            adjust=False
        ).mean()
        df[f"SMA{period}"] = close.rolling(
            period
        ).mean()

    # RSI - طريقة Wilder
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # معالجة حالة عدم وجود خسائر
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]

    # متوسط حجم التداول
    df["VolumeAvg20"] = df["Volume"].rolling(20).mean()

    # الدعم السابق
    df["Support20"] = df["Low"].shift(1).rolling(20).min()
    df["NearestSupport"] = np.nan

    # أقرب قاع محلي أسفل السعر
    for i in range(20, len(df)):
        price = df["Close"].iloc[i]
        previous_lows = df["Low"].iloc[i - 20:i]

        local_troughs = (
            (previous_lows < previous_lows.shift(1)) &
            (previous_lows <= previous_lows.shift(-1))
        )

        levels = previous_lows[
            local_troughs & (previous_lows < price)
        ]

        if not levels.empty:
            df.iloc[
                i,
                df.columns.get_loc("NearestSupport")
            ] = levels.max()

    # المقاومة السابقة
    df["Resistance60"] = df["High"].shift(1).rolling(60).max()
    df["NearestResistance"] = np.nan

    # أقرب قمة محلية فوق السعر
    for i in range(60, len(df)):
        price = df["Close"].iloc[i]
        previous_highs = df["High"].iloc[i - 60:i]

        local_peaks = (
            (previous_highs > previous_highs.shift(1)) &
            (previous_highs >= previous_highs.shift(-1))
        )

        levels = previous_highs[
            local_peaks & (previous_highs > price)
        ]

        if not levels.empty:
            df.iloc[
                i,
                df.columns.get_loc("NearestResistance")
            ] = levels.min()

    return df
# ==========================================
# تحليل الشموع
# ==========================================

def detect_candle(df):
    if len(df) < 3:
        return "بيانات غير كافية"

    last = df.iloc[-1]
    previous = df.iloc[-2]

    o = number(last["Open"])
    h = number(last["High"])
    l = number(last["Low"])
    c = number(last["Close"])

    po = number(previous["Open"])
    pc = number(previous["Close"])

    if None in [o, h, l, c, po, pc]:
        return "بيانات غير مكتملة"

    body = abs(c - o)
    candle_range = h - l

    if candle_range <= 0:
        return "شمعة غير واضحة"

    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    # ابتلاع شرائي
    if (
        pc < po
        and c > o
        and o <= pc
        and c >= po
    ):
        return "ابتلاع شرائي محتمل"

# مطرقة محتملة مع إغلاق في الجزء العلوي من الشمعة
    if (
        lower_wick >= max(body * 2, candle_range * 0.30)
        and upper_wick <= candle_range * 0.15
        and c > l + candle_range * 0.60
    ):
        return "مطرقة محتملة"
    # شمعة صاعدة قوية
    if (
        c > o
        and body >= candle_range * 0.60
    ):
        return "شمعة صاعدة قوية"

    # شمعة هابطة
    if c < o and body >= candle_range * 0.60:
        return "شمعة هابطة قوية"

    return "لا توجد شمعة انعكاس واضحة"


# ==========================================
# فحص سحب السيولة واستعادة الدعم
# ==========================================

def detect_liquidity_sweep(df):
    """
    فحص تقريبي:
    هبوط أدنى من قيعان سابقة ثم إغلاق فوق مستوى القاع السابق.

    لا يثبت هذا الفحص وجود سحب سيولة مؤسسي.
    """

    if len(df) < 22:
        return False, None

    last = df.iloc[-1]

    previous_lows = df["Low"].iloc[-21:-1]

    if previous_lows.empty:
        return False, None

    previous_support = number(previous_lows.min())
    last_low = number(last["Low"])
    last_close = number(last["Close"])

    if None in [previous_support, last_low, last_close]:
        return False, previous_support

    swept = last_low < previous_support
    reclaimed = last_close > previous_support

    return swept and reclaimed, previous_support


# ==========================================
# فحص نموذج W
# ==========================================

def detect_w_pattern(df):
    """
    فحص تقريبي لقاعين متقاربين.
    لا يعتمد على نموذج إحصائي مؤكد.
    """

    if len(df) < 40:
        return {
            "possible": False,
            "confirmed": False,
            "neckline": None,
            "reason": "بيانات غير كافية"
        }

    recent = df.tail(60).copy()

    lows = recent["Low"].to_numpy()
    highs = recent["High"].to_numpy()
    closes = recent["Close"].to_numpy()

    candidates = []

    # استخراج القيعان المحلية
    for i in range(2, len(lows) - 2):
        if (
            lows[i] <= lows[i - 1]
            and lows[i] <= lows[i - 2]
            and lows[i] <= lows[i + 1]
            and lows[i] <= lows[i + 2]
        ):
            candidates.append(i)

    if len(candidates) < 2:
        return {
            "possible": False,
            "confirmed": False,
            "neckline": None,
            "reason": "لم يظهر قاعان محليان واضحان"
        }

    # البحث عن آخر قاعين متقاربين
    for j in range(len(candidates) - 1, 0, -1):
        second_idx = candidates[j]

        for k in range(j - 1, -1, -1):
            first_idx = candidates[k]

            distance = second_idx - first_idx

            if distance < 5 or distance > 35:
                continue

            first_low = lows[first_idx]
            second_low = lows[second_idx]

            if first_low <= 0 or second_low <= 0:
                continue

            difference = abs(first_low - second_low) / (
                (first_low + second_low) / 2
            )

            if difference > 0.04:
                continue

            neckline = float(
                np.max(highs[first_idx:second_idx + 1])
            )

            current_close = float(closes[-1])

            return {
                "possible": True,
                "confirmed": current_close > neckline,
                "neckline": neckline,
                "reason": (
                    "اختراق خط العنق"
                    if current_close > neckline
                    else "لم يتأكد تجاوز خط العنق"
                )
            }

    return {
        "possible": False,
        "confirmed": False,
        "neckline": None,
        "reason": "لم يظهر قاعان متقاربان بما يكفي"
    }


# ==========================================
# معلومات الشركة
# ==========================================

def get_company_data(symbol):
    result = {
        "float": None,
        "short": None,
        "short_percent": None
    }

    try:
        info = yf.Ticker(symbol).info

        result["float"] = number(info.get("floatShares"))
        result["short"] = number(info.get("sharesShort"))

        short_percent = number(info.get("shortPercentOfFloat"))

        if short_percent is not None:
            result["short_percent"] = short_percent * 100

    except Exception:
        pass

    return result


# ==========================================
# تصنيف الفرصة وإدارة المخاطر
# ==========================================

def classify_stock(
    price,
    rsi,
    support,
    resistance,
    macd,
    signal,
    histogram,
    previous_histogram,
    volume,
    average_volume,
    candle,
    sweep,
    w_pattern,
    recovered_averages
):
    reasons = []
    score = 0

    near_support = (
        support is not None
        and support > 0
        and abs(price - support) / support <= 0.03
    )

    if near_support:
        score += 1
        reasons.append("السعر قريب من الدعم التقريبي")

    if rsi is not None and RSI_MIN <= rsi <= RSI_MAX:
        score += 1
        reasons.append("RSI ضمن نطاق 23–27")

    if sweep:
        score += 2
        reasons.append("كسر قاع سابق ثم استعادة المستوى")

    macd_improving = (
        histogram is not None
        and previous_histogram is not None
        and histogram > previous_histogram
    )

    if macd_improving:
        score += 1
        reasons.append("تحسن في زخم MACD")

    if (
        volume is not None
        and average_volume is not None
        and average_volume > 0
        and volume > average_volume
        and price > 0
        and candle in [
            "ابتلاع شرائي محتمل",
            "مطرقة محتملة",
            "شمعة صاعدة قوية"
        ]
    ):
        score += 1
        reasons.append("حجم أعلى من المتوسط مع شمعة صاعدة")

    if candle in [
        "ابتلاع شرائي محتمل",
        "مطرقة محتملة",
        "شمعة صاعدة قوية"
    ]:
        score += 1
        reasons.append("ظهرت شمعة صاعدة محتملة")

    if w_pattern["confirmed"]:
        score += 2
        reasons.append("اختراق محتمل لخط عنق نموذج W")

    if recovered_averages:
        score += 1
        reasons.append("السعر أعلى من بعض المتوسطات المتحركة")

# لا نعتبر السهم مرشحاً للدخول بمجرد انخفاض RSI
    if support is not None and price < support:
        status = "🔴 تجنب الدخول حاليًا"
        reasons.append("السعر أسفل الدعم التقريبي")

    elif (
        score >= 5
        and sweep
        and candle in [
            "ابتلاع شرائي محتمل",
            "مطرقة محتملة",
            "شمعة صاعدة قوية"
        ]
    ):
        status = "🟢 مرشح للدراسة والدخول المشروط"
        reasons.append(
            "اجتمعت إشارة سحب السيولة مع شمعة إيجابية"
        )

    elif (
        score >= 4
        and w_pattern["confirmed"]
        and candle in [
            "ابتلاع شرائي محتمل",
            "مطرقة محتملة",
            "شمعة صاعدة قوية"
        ]
    ):
        status = "🟢 مرشح للدراسة بعد اختراق نموذج W"
        reasons.append(
            "اختراق نموذج W مع ظهور شمعة إيجابية"
        )

    elif (
        score >= 2
        or near_support
    ):
        status = "🟡 تحت المراقبة"

    else:
        status = "🔴 لا توجد فرصة واضحة حاليًا"

    return status, score, reasons


# ==========================================
# التحليل الرئيسي
# ==========================================

def analyze_stock(symbol):
    # التحقق من Float قبل التحليل الفني
    company = get_company_data(symbol)
    float_shares = company["float"]

    if float_shares is None:
        print(f"❌ {symbol}: بيانات Float غير متوفرة")
        return

    if float_shares >= 5_000_000:
        print(
            f"❌ {symbol}: Float يبلغ "
            f"{float_shares:,.0f} سهم، تم استبعاده"
        )
        return

    print(
        f"✅ {symbol}: Float أقل من 5 ملايين سهم"
    )
    print("\n" + "=" * 48)
    print(f"فحص السهم: {symbol}")
    print("=" * 48)

    raw = download_daily(symbol)

    if raw is None:
        return

    df = add_indicators(raw)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = number(current["Close"])
    previous_close = number(previous["Close"])

    if price is None or price <= 0:
        print("السعر غير متوفر")
        return

    daily_change = (
        (price - previous_close) / previous_close * 100
        if previous_close and previous_close > 0
        else None
    )

    rsi = number(current["RSI"])

    macd = number(current["MACD"])
    signal = number(current["Signal"])
    histogram = number(current["Histogram"])
    previous_histogram = number(previous["Histogram"])

    volume = number(current["Volume"])
    average_volume = number(current["VolumeAvg20"])

    support = number(current["NearestSupport"])
    resistance = number(current["NearestResistance"])

    print(f"السعر الحالي: ${price:.4f}")

    if daily_change is not None:
        print(f"التغير اليومي: {daily_change:+.2f}%")

    print(f"RSI: {fmt(rsi)}")

    print("\nالمتوسطات:")

    averages = {}

    for kind in ["EMA", "SMA"]:
        for period in [20, 30, 50, 200]:
            name = f"{kind}{period}"
            value = number(current[name])
            averages[name] = value
            print(f"{name}: {fmt(value)}")

    recovered_averages = [
        name
        for name, value in averages.items()
        if value is not None and price > value
    ]

    print("\nالدعم والمقاومة:")

    print(f"الدعم التقريبي: {fmt(support)}")
    print(f"المقاومة التقريبية: {fmt(resistance)}")

    lowest_60 = number(df["Low"].tail(60).min())

    print(f"أدنى سعر خلال 60 جلسة: {fmt(lowest_60)}")

    print("\nحجم التداول:")

    print(f"حجم اليوم: {fmt(volume)}")
    print(f"متوسط الحجم: {fmt(average_volume)}")

    print("\nMACD:")

    print(f"MACD: {fmt(macd)}")
    print(f"Signal: {fmt(signal)}")
    print(f"Histogram: {fmt(histogram)}")

    macd_improving = (
        histogram is not None
        and previous_histogram is not None
        and histogram > previous_histogram
    )

    print(
        "تحسن الزخم: "
        + ("نعم" if macd_improving else "لا")
    )

    bullish_cross = (
        macd is not None
        and signal is not None
        and macd > signal
        and number(previous["MACD"]) is not None
        and number(previous["Signal"]) is not None
        and number(previous["MACD"]) <= number(previous["Signal"])
    )

    print(
        "تقاطع إيجابي جديد: "
        + ("نعم" if bullish_cross else "لا")
    )

    candle = detect_candle(df)

    sweep, previous_support = detect_liquidity_sweep(df)

    w_pattern = detect_w_pattern(df)

    print("\nفحص سحب السيولة:")

    if sweep:
        print("ظهرت إشارة تقريبية لكسر قاع سابق ثم استعادة المستوى")
        print(f"المستوى المستعاد: {fmt(previous_support)}")
    else:
        print("لم تتأكد إشارة سحب السيولة واستعادة المستوى")

    print("\nالشموع:")
    print(candle)

    print("\nنموذج W:")

    if w_pattern["possible"]:
        print("نموذج محتمل")
        print(f"خط العنق التقريبي: {fmt(w_pattern['neckline'])}")
        print(w_pattern["reason"])
    else:
        print(w_pattern["reason"])

    # التغير خلال 20 جلسة
    change_20 = None

    if len(df) >= 21:
        old_price = number(df["Close"].iloc[-21])

        if old_price and old_price > 0:
            change_20 = (price / old_price - 1) * 100

    print("\nمخاطر مطاردة السعر:")

    if change_20 is not None:
        print(f"التغير خلال 20 جلسة: {change_20:+.2f}%")

        if change_20 >= 20:
            print("تحذير: لا تطارد ارتفاعًا قويًا دون إعادة اختبار")
    else:
        print("بيانات غير كافية")

    print("\nمستويات فنية تقديرية:")
    stop = None
    target = None
    entry_low = None
    entry_high = None

    if support is not None and support > 0:
        stop = support * 0.98

        if support <= price <= support * 1.03:
            entry_low = support
            entry_high = support * 1.02

    # البحث عن أقرب هدف فني أعلى من السعر
    candidates = []

    if resistance is not None and resistance > price:
        candidates.append(resistance)

    for name in ["EMA20", "SMA20", "EMA30", "SMA30", "EMA50", "SMA50"]:
        value = number(current[name])
        if value is not None and value > price:
            candidates.append(value)

    if candidates:
        target = min(candidates)

    print(
        f"منطقة دخول للمراقبة: "
        f"{fmt(entry_low)} - {fmt(entry_high)}"
        if entry_low is not None
        else "لا توجد منطقة دخول مناسبة محسوبة"
    )

    print(f"وقف فني تقديري: {fmt(stop)}")
    print(f"الهدف الفني الأول: {fmt(target)}")

    if entry_low is not None and stop is not None:
        risk = entry_low - stop
        reward = target - entry_low if target is not None else None

        if risk > 0 and reward is not None and reward > 0:
            print(
                f"العائد إلى المخاطرة التقريبي: "
                f"1:{reward / risk:.2f}"
            )

    status, score, reasons = classify_stock(
        price=price,
        rsi=rsi,
        support=support,
        resistance=resistance,
        macd=macd,
        signal=signal,
        histogram=histogram,
        previous_histogram=previous_histogram,
        volume=volume,
        average_volume=average_volume,
        candle=candle,
        sweep=sweep,
        w_pattern=w_pattern,
        recovered_averages=recovered_averages
    )

    print("\n" + "=" * 48)
    print("التصنيف النهائي:")
    print(status)
    print(f"درجة الشروط المتحققة: {score}")

    print("\nأسباب التصنيف:")

    if reasons:
        for reason in reasons:
            print(f"- {reason}")
    else:
        print("- لم تتحقق شروط فنية كافية")

    if support is not None and price < support:
        print(
            "\nتنبيه: كسر الدعم المحسوب لا يثبت وحده "
            "استمرار الهبوط أو حدوث سحب سيولة."
        )

    print("\nبيانات الشركة:")


    print(f"Float: {fmt(company['float'])}")
    print(f"Shares Short: {fmt(company['short'])}")
    print(
        f"Short % of Float: {fmt(company['short_percent'])}%"
        if company["short_percent"] is not None
        else "Short % of Float: غير متوفر"
    )

    print(
        "\nتنبيه: التحليل آلي وتعليمي، "
        "والبيانات اليومية لا تؤكد وحدها توقيت الدخول."
    )


# ==========================================
# تشغيل البوت
# ==========================================

def main():
    print("بدء ماسح الأسهم - منهج فيصل")
    print("الفاصل الزمني: يومي")
    print("البيانات من Yahoo Finance")

    symbols = discover_stocks()

    if not symbols:
        print("قائمة الأسهم فارغة أو غير صالحة.")
        return

    print(f"عدد الأسهم: {len(symbols)}")

    for symbol in symbols:
        try:
            analyze_stock(symbol)
        except Exception as exc:
            print(f"حدث خطأ أثناء تحليل {symbol}: {exc}")

    print("\nانتهى فحص جميع الأسهم.")


if __name__ == "__main__":
    main()
