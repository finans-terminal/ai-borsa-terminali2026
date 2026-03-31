
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import streamlit.components.v1 as components

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Finans Merkezi | Doğuş Can Şen", layout="wide")

# --- 1. ÜST KAYAN BANT (TICKER) ---
@st.cache_data(ttl=300)
def get_ticker_data():
    symbols = {
        "BIST 100": "XU100.IS", "USD/TRY": "USDTRY=X", "EUR/TRY": "EURTRY=X", 
        "ALTIN": "GC=F", "NASDAQ": "^IXIC", "S&P 500": "^GSPC", "TESLA": "TSLA", 
        "APPLE": "AAPL", "NVIDIA": "NVDA", "BITCOIN": "BTC-USD"
    }
    text = ""
    for name, sym in symbols.items():
        try:
            d = yf.Ticker(sym).history(period="2d")
            c = d['Close'].iloc[-1]
            ch = ((c - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
            col = "#00ff00" if ch >= 0 else "#ff0000"
            arrow = "▲" if ch >= 0 else "▼"
            text += f"&nbsp;&nbsp;&nbsp;&nbsp; **{name}:** {round(c,2)} <span style='color:{col};'>{arrow} %{round(ch,2)}</span> &nbsp;&nbsp;&nbsp;&nbsp; |"
        except: continue
    return text

st.markdown(f'<marquee style="color: white; font-size: 16px; background: #1e2130; padding: 10px; border-radius: 5px;">{get_ticker_data()}</marquee>', unsafe_allow_html=True)


# --- 2. SOL PANEL (AYARLAR) ---
st.sidebar.header("🕹️ Kontrol & Enstrüman")
hisse_listesi = {
    "THYAO.IS": "THY (BIST)", "TUPRS.IS": "Tüpraş (BIST)", "EREGL.IS": "Erdemir (BIST)", 
    "SASA.IS": "Sasa (BIST)", "KCHOL.IS": "Koç Holding (BIST)", "ASELS.IS": "Aselsan (BIST)",
    "AAPL": "Apple (USA)", "TSLA": "Tesla (USA)", "NVDA": "Nvidia (USA)",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"
}
hisse_key = st.sidebar.selectbox("Enstrüman Seçin", list(hisse_listesi.keys()), format_func=lambda x: hisse_listesi[x])
interval = st.sidebar.selectbox("Zaman Dilimi", ["1d", "1h", "15m", "5m"], index=0)
confidence_level = st.sidebar.slider("AI Alım Güveni (%)", 50, 95, 80) / 100

st.sidebar.divider()
st.sidebar.subheader("📺 Finans TV")
st.sidebar.link_button("🌐 Bloomberg HT Canlı Yayını", "https://www.youtube.com/watch?v=hHSmBJk6w0c")

with st.sidebar.expander("❓ Robot Nasıl Çalışır?"):
    st.info("Bu robot; RSI, EMA ve Hacim verilerini Random Forest algoritması ile işleyerek gelecekteki fiyat yönünü tahmin eder.")

# --- 3. VERİ VE AI SİSTEMİ ---
@st.cache_data(ttl=300) # 5 dakikada bir tazeler, boşlukları kapatmaya çalışır
def get_full_data(symbol, i):
    # Ayrık mumları engellemek için periodu sabitliyoruz
    p = "1mo" if i == "1h" else "1y"
    
    # Veriyi çekiyoruz
    df = yf.download(symbol, period=p, interval=i, auto_adjust=True)
    
    if not df.empty:
        # Sütun isimlerini temizle
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # AYRIK MUM TAMİRİ: Verisi olmayan (NaN) satırları siliyoruz
        df.dropna(inplace=True)
        
        # Eğer borsa kapalıysa son geçerli veriyi gösterir
        return df
df = get_full_data(hisse_key, interval)

if len(df) > 30:
    # Göstergeler
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df_m = df.dropna().copy()
    X = df_m[['RSI', 'EMA20', 'Volume']]
    y = df_m['Target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:-1], y[:-1])
    df_m['AI_Prob'] = model.predict_proba(X)[:, 1]
    
    # Özet Kartlar
    son_fiyat = df_m['Close'].iloc[-1]
    atr_v = df_m['ATR'].iloc[-1]
    stop_l = son_fiyat - (atr_v * 2)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anlık Fiyat", f"{round(son_fiyat, 2)}")
    c2.metric("Robot Durumu", "AL" if df_m['AI_Prob'].iloc[-1] >= confidence_level else "BEKLE")
    c3.metric("AI Güven", f"%{round(df_m['AI_Prob'].iloc[-1]*100, 1)}")
    c4.metric("Stop-Loss", f"{round(stop_l, 2)}")

    # Tablar
    tab1, tab2, tab3 = st.tabs(["📊 Robot Grafik", "📋 İşlem Geçmişi", "📰 Haberler"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_m.index, open=df_m['Open'], high=df_m['High'], low=df_m['Low'], close=df_m['Close'], name="Fiyat"))
        buys = df_m[df_m['AI_Prob'] >= confidence_level]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name="AL Sinyali"))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📋 Son İşlemler")
        history = df_m[df_m['AI_Prob'] >= confidence_level].tail(10).copy()
        if not history.empty:
            history['Zaman'] = history.index.strftime('%d-%m-%Y %H:%M')
            st.table(history[['Zaman', 'Close', 'AI_Prob']].rename(columns={'Close': 'Fiyat', 'AI_Prob': 'AI Güveni'}))

    with tab3:
        st.subheader("🔔 Gündem")
        ticker = yf.Ticker(hisse_key)
        try:
            for n in ticker.news[:5]:
                st.write(f"**{n['title']}**")
                st.caption(f"Kaynak: {n.get('publisher', 'Haber')}")
                st.link_button("Detay", n['link'])
                st.divider()
        except: st.write("Haberler şu an yüklenemedi.")



# --- 5. SAĞ ALT KÖŞE KARTVİZİT ---
st.markdown("""
<style>
.fixed-footer {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #1e2130;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #00ff00;
    color: white;
    z-index: 9999;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.8);
    width: 260px;
}
</style>
<div class="fixed-footer">
    <div style="color: #00ff00; font-weight: bold; font-size: 16px; margin-bottom: 5px;">Geliştirici: Doğuş Can Şen</div>
    <div style="font-size: 13px; line-height: 1.5;">
        🎓 <b>Mehmet Akif Ersoy Üniversitesi</b><br>
        💼 Ekonomi ve Finans Bölümü<br>
        📧 <b>Email:</b> doguscan@email.com<br>
        📞 <b>Tel:</b> +90 5xx xxx xx xx
    </div>
</div>
""", unsafe_allow_html=True)
