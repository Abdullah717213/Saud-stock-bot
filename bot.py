import csv
import math
import os
import time
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# إعدادات الفحص - مبنية على منهج فيصل
# ==========================================

RSI_MIN = 23
RSI_MAX = 27

SUPPORT_LOOKBACK = 20
LOW_LOOKBACK = 60
VOLUME_LOOKBACK = 20

# لا تعتبر هذه الأوزان توصية شراء.
# هي طريقة لترتيب الأسهم حسب اكتمال شروط الفحص.
MIN_WATCH_SCORE = 5


def number(value, default=None):
    """تحويل القيمة إلى رقم آمن."""
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return default


def load_watchlist():
    """قراءة قائمة الأسهم."""
    if not os.path.exists("watchlist.csv"):
        print("خطأ: ملف watchlist.csv غير موجود.")
        return []

    with open(
        "watchlist.csv",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        symbols = []

        for row in reader:
            symbol = row.get("symbol", "").strip().upper()
            if symbol:
                symbols.append(symbol)

        return list(dict.fromkeys(symbols))


def download_daily(symbol):
    """تحميل بيانات يومية للتحليل الفني."""
    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        return None

    # معالجة اختلاف شكل الأعمدة في yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in required):
        return None

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if len(df) < 60:
        return None

    return df


def add_indicators(df):
    """حساب المؤشرات الفنية."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # المتوسطات المتحركة
    for period in [20, 30, 50, 200]:
        df[f"SMA{period}"] = close.rolling(period).mean()
        df[f"EMA{period}"] = close.ewm(
            span=period,
            adjust=False
        ).mean()

    # RSI 14
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

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # متوسط الفوليوم
    df["AVG_VOLUME"] = volume.rolling(
        VOLUME_LOOKBACK
    ).mean()

    # متوسط نطاق الشمعة
    df["RANGE"] = high - low

    return df


def detect_candle(df):
    """فحص أولي للمطرقة والابتلاع الشرائي."""
    if len(df) < 2:
        return []

    current = df.iloc[-1]
    previous = df.iloc[-2]

    o = number(current["Open"], 0)
    h = number(current["High"], 0)
    l = number(current["Low"], 0)
    c = number(current["Close"], 0)

    body = abs(c - o)
    full_range = h - l

    if full_range <= 0:
        return []

    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    patterns = []

    # مطرقة تقريبية: ذيل سفلي طويل وإغلاق قرب أعلى الشمعة
    if (
        lower_shadow >= max(body * 2, full_range * 0.4)
        and upper_shadow <= full_range * 0.3
        and c > l + full_range * 0.6
    ):
        patterns.append("مطرقة محتملة")

    # ابتلاع شرائي
    prev_o = number(previous["Open"], 0)
    prev_c = number(previous["Close"], 0)

    if (
        prev_c < prev_o
        and c > o
        and o <= prev_c
        and c >= prev_o
    ):
        patterns.append("ابتلاع شرائي محتمل")

    # رفض كسر داخل الشمعة
    if lower_shadow > body * 1.5 and c > o:
        patterns.append("ذيل سفلي مع إغلاق إيجابي")

    return patterns


def detect_w_pattern(df):
    """
    بحث تقريبي عن قاعين متقاربين.
    يحتاج مراجعة الرسم؛ ليس كشفًا مضمونًا لنموذج W.
    """
    if len(df) < 40:
        return False, None

    recent = df.tail(40)
    lows = recent["Low"].tolist()
    highs = recent["High"].tolist()

    pivots = []

    for i in range(2, len(lows) - 2):
        if (
            lows[i] <= lows[i - 1]
            and lows[i] <= lows[i - 2]
            and lows[i] <= lows[i + 1]
            and lows[i] <= lows[i + 2]
        ):
            pivots.append(i)

    if len(pivots) < 2:
        return False, None

    first = pivots[-2]
    second = pivots[-1]

    if second - first < 5:
        return False, None

    low1 = lows[first]
    low2 = lows[second]

    if low1 <= 0:
        return False, None

    difference = abs(low1 - low2) / low1

    # القاعان متقاربان نسبيًا
    if difference > 0.04:
        return False, None

    neckline = max(highs[first:second + 1])

    # نعرض النموذج حتى لو لم يخترق خط العنق،
    # لكن نوضح أن التأكيد لم يحصل بعد.
    return True, neckline


def get_company_data(symbol):
    """
    محاولة جلب الفلووت والشورت والأخبار.
    قد لا تتوفر هذه البيانات لكل سهم.
    """
    result = {
        "float": None,
        "short": None,
        "short_pct": None,
        "news": []
    }

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()

        result["float"] = info.get("floatShares")
        result["short"] = info.get("sharesShort")
        result["short_pct"] = info.get("shortPercentOfFloat")

        news_items = ticker.news or []

        for item in news_items[:5]:
            content = item.get("content", item)

            title = content.get("title", "")

            if not title:
                continue

            result["news"].append(title)

    except Exception:
        pass

    return result


def analyze_stock(symbol):
    print("\n" + "=" * 45)
    print(f"فحص السهم: {symbol}")
    print("=" * 45)

    df = download_daily(symbol)

    if df is None:
        print("النتيجة: تعذر الحصول على بيانات كافية.")
        return

    df = add_indicators(df)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = number(current["Close"])
    previous_close = number(previous["Close"])

    if price is None or previous_close is None:
        print("النتيجة: بيانات السعر غير مكتملة.")
        return

    daily_change = (
        (price - previous_close) / previous_close
    ) * 100 if previous_close else 0

    rsi = number(current["RSI"])
    macd = number(current["MACD"])
    signal = number(current["MACD_SIGNAL"])
    histogram = number(current["MACD_HIST"])
    previous_hist = number(previous["MACD_HIST"])

    avg_volume = number(current["AVG_VOLUME"], 0)
    latest_volume = number(current["Volume"], 0)

    # الدعم والقاع: نستثني الشمعة الحالية عند تحديد القاع السابق
    previous_lows = df["Low"].iloc[-21:-1]
    support20 = number(previous_lows.min())

    low60 = number(df["Low"].tail(60).min())

    # مقاومة تقريبية من القمم السابقة
    previous_highs = df["High"].iloc[-21:-1]
    resistance20 = number(previous_highs.max())

    # الفحص الأولي لقرب السعر من الدعم
    near_support = (
        support20 is not None
        and support20 > 0
        and price <= support20 * 1.05
    )

    # سحب السيولة: كسر القاع السابق ثم الإغلاق فوقه
    sweep_reclaim = (
        support20 is not None
        and number(current["Low"]) < support20
        and price > support20
    )

    # هل تحسن MACD؟
    macd_improving = (
        histogram is not None
        and previous_hist is not None
        and histogram > previous_hist
    )

    macd_bullish_cross = (
        macd is not None
        and signal is not None
        and macd > signal
        and number(previous["MACD"])
        is not None
        and number(previous["MACD_SIGNAL"])
        is not None
        and number(previous["MACD"])
        <= number(previous["MACD_SIGNAL"])
    )

    # الفوليوم مع شمعة صاعدة
    bullish_candle = price > number(current["Open"], price)

    volume_confirmation = (
        avg_volume > 0
        and latest_volume > avg_volume
        and bullish_candle
    )

    candles = detect_candle(df)

# فحص استعادة المتوسطات EMA وSMA
    averages = {}

    for kind in ["EMA", "SMA"]:
        for period in [20, 30, 50, 200]:
            name = f"{kind}{period}"
            value = number(current[name])
            averages[name] = value

    recovered_averages = [
        name for name, value in averages.items()
        if value is not None and price > value
    ]

    # هل ارتفع السهم بقوة خلال آخر 20 جلسة؟
    close20 = number(df["Close"].iloc[-21])

    rise20 = (
        ((price / close20) - 1) * 100
        if close20 and close20 > 0
        else 0
    )

    chasing_warning = rise20 >= 50

    # البحث التقريبي عن W
    has_w, neckline = detect_w_pattern(df)

    neckline_break = (
        has_w
        and neckline is not None
        and price > neckline
    )

    # دعم فني مبدئي لإبطال فكرة الارتداد
    stop_level = support20 * 0.99 if support20 else None

    # حساب النقاط الفنية
    checks = {
        "قرب الدعم": near_support,
        "RSI بين 23 و27": (
            rsi is not None and RSI_MIN <= rsi <= RSI_MAX
        ),
        "سحب سيولة واستعادة الدعم": sweep_reclaim,
        "تحسن MACD": macd_improving or macd_bullish_cross,
        "فوليوم أعلى من المتوسط مع شمعة صاعدة": volume_confirmation,
        "شمعة ارتداد محتملة": bool(candles),
        "استعادة متوسط متحرك": bool(recovered_averages),
    }

    score = sum(checks.values())

    # النتائج
    print(f"السعر الحالي: {price:.4f}")
    print(f"التغير اليومي: {daily_change:.2f}%")
    print(f"RSI: {rsi:.2f}" if rsi is not None else "RSI: غير متاح")

    print("\nالمتوسطات:")
    for period in [20, 30, 50, 200]:
        ema = number(current[f"EMA{period}"])
        sma = number(current[f"SMA{period}"])

        print(
            f"EMA{period}: {ema:.4f} | SMA{period}: {sma:.4f}"
            if ema is not None and sma is not None
            else f"MA{period}: غير متاح"
        )

    print("\nالدعم والمقاومة:")
    print(f"الدعم التقريبي: {support20:.4f}" if support20 else "الدعم: غير متاح")
    print(f"أدنى سعر خلال 60 جلسة: {low60:.4f}" if low60 else "القاع: غير متاح")
    print(f"المقاومة التقريبية: {resistance20:.4f}" if resistance20 else "المقاومة: غير متاحة")

    print("\nالفوليوم:")
    print(f"حجم اليوم: {latest_volume:,.0f}")
    print(f"متوسط الحجم: {avg_volume:,.0f}")

    print("\nMACD:")
    print(f"MACD: {macd:.4f}" if macd is not None else "MACD: غير متاح")
    print(f"Signal: {signal:.4f}" if signal is not None else "Signal: غير متاح")
    print(f"تحسن الزخم: {'نعم' if macd_improving else 'لا'}")
    print(f"تقاطع إيجابي جديد: {'نعم' if macd_bullish_cross else 'لا'}")

    print("\nفحص شروط منهج فيصل:")
    for name, passed in checks.items():
        print(f"{'✓' if passed else '✗'} {name}")

    print("\nالشموع:")
    print(", ".join(candles) if candles else "لم تظهر شمعة ارتداد من الأنماط البسيطة")

    print("\nنموذج W:")
    if has_w:
        print(f"نموذج محتمل، خط العنق التقريبي: {neckline:.4f}")
        print(
            "تجاوز خط العنق"
            if neckline_break
            else "لم يتأكد تجاوز خط العنق"
        )
    else:
        print("لم يتم رصد نموذج W واضح حسابيًا")

    print("\nمخاطر مطاردة السعر:")
    print(f"الارتفاع خلال 20 جلسة: {rise20:.2f}%")

    if chasing_warning:
        print("تحذير: ارتفاع قوي؛ لا تطارد السهم، وانتظر إعادة اختبار أو سحب سيولة.")

    print("\nوقف فني مبدئي:")
    if stop_level is not None:
        print(f"مستوى متابعة تقريبي: {stop_level:.4f}")
        print("هذا ليس أمر وقف مضمونًا، ويجب التحقق من الدعم الحقيقي على الرسم.")
    else:
        print("غير متاح")

    print("\nالنتيجة الأولية:")

    if chasing_warning:
        print("انتظار إعادة اختبار أو سحب سيولة؛ لا تطارد الصعود.")
    elif (
        near_support
        and score >= MIN_WATCH_SCORE
        and (sweep_reclaim or candles or macd_improving)
    ):
        print("مرشح للمراقبة المتقدمة؛ راجع الشارت والثبات قبل أي قرار.")
    elif near_support:
        print("قريب من الدعم، لكن انتظر تأكيد فني إضافي.")
    else:
        print("ليس مرشح ارتكاز مكتمل الشروط حاليًا.")

    # البيانات الخارجية: الفلووت والشورت والأخبار
    print("\nبيانات الشركة والأخبار:")
    company = get_company_data(symbol)

    if company["float"] is not None:
        print(f"Float: {company['float']:,}")
    else:
        print("Float: غير متاح من مصدر البيانات")

    if company["short"] is not None:
        print(f"Shares Short: {company['short']:,}")
    else:
        print("Shares Short: غير متاحة")

    if company["short_pct"] is not None:
        print(f"Short % of Float: {company['short_pct']:.2%}")
    else:
        print("نسبة الشورت إلى الفلووت: غير متاحة")

    if company["news"]:
        for title in company["news"]:
            print(f"- {title}")
    else:
        print("لم تتوفر عناوين أخبار من مصدر البيانات.")

    print("\nتنبيه: هذا فحص آلي أولي وليس توصية شراء أو بيع.")


def main():
    symbols = load_watchlist()

    if not symbols:
        print("أضف رموز الأسهم إلى watchlist.csv أولًا.")
        return

    print("بدء فحص الأسهم وفق شروط منهج فيصل")
    print(f"عدد الأسهم: {len(symbols)}")

    for symbol in symbols:
        try:
            analyze_stock(symbol)
        except Exception as error:
            print(f"{symbol}: حدث خطأ أثناء الفحص: {error}")

        time.sleep(1)


if __name__ == "__main__":
    main()
