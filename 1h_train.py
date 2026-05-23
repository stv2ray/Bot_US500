import pandas as pd
import numpy as np
import ta
import joblib
import os
import warnings
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout,
    BatchNormalization, MultiHeadAttention, Flatten, Add, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN APPLE SILICON (M4 - GPU OPTIMIZADO)
# ═══════════════════════════════════════════════════════════════
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(2)
print("TensorFlow versión:", tf.__version__)
print("Dispositivos:", tf.config.list_physical_devices())

# ═══════════════════════════════════════════════════════════════
# PARÁMETROS DE LA ESTRATEGIA (VERSIÓN PRODUCCIÓN ICMARKETS)
# ═══════════════════════════════════════════════════════════════
SYMBOL = "US500"
DATA_DIR = "históricos"
MODEL_DIR = f"modelos/{SYMBOL}_produccion"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LENGTH = 60
HORIZON = 8                
TARGET_ATR_MULT = 2.0      
SL_ATR_MULT = 1.5          
RISK_PER_TRADE = 0.01
LOT_VALUE_PER_POINT = 100  
INITIAL_CAPITAL = 10000
SPREAD_POINTS = 0.5
COMMISSION_PER_LOT = 5.0   
LSTM_PERCENTIL = 85        
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"

# ═══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS Y LIMPIEZA INICIAL
# ═══════════════════════════════════════════════════════════════
print("Cargando datos compatibles con MT5...")
df_us500 = pd.read_csv(os.path.join(DATA_DIR, "US500_H1.csv"), index_col=0, parse_dates=True)
df_gold  = pd.read_csv(os.path.join(DATA_DIR, "XAUUSD_H1.csv"), index_col=0, parse_dates=True)
df_oil   = pd.read_csv(os.path.join(DATA_DIR, "XTIUSD_H1.csv"), index_col=0, parse_dates=True)

for df_temp in [df_us500, df_gold, df_oil]:
    if not isinstance(df_temp.index, pd.DatetimeIndex):
        df_temp.index = pd.to_datetime(df_temp.index, errors='coerce')
    if df_temp.index.tz is not None:
        df_temp.index = df_temp.index.tz_localize(None)

if 'tick_volume' in df_us500.columns and 'volume' not in df_us500.columns:
    df_us500.rename(columns={'tick_volume': 'volume'}, inplace=True)
if 'tick_volume' in df_gold.columns and 'volume' not in df_gold.columns:
    df_gold.rename(columns={'tick_volume': 'volume'}, inplace=True)
if 'tick_volume' in df_oil.columns and 'volume' not in df_oil.columns:
    df_oil.rename(columns={'tick_volume': 'volume'}, inplace=True)

df_gold_ren = df_gold[['open','high','low','close','volume']].rename(columns=lambda x: f'gold_{x}')
df_oil_ren  = df_oil[['open','high','low','close','volume']].rename(columns=lambda x: f'oil_{x}')

df = df_us500.join(df_gold_ren, how='left').join(df_oil_ren, how='left')

cols_numeric = ['gold_open', 'gold_high', 'gold_low', 'gold_close', 'gold_volume',
                'oil_open', 'oil_high', 'oil_low', 'oil_close', 'oil_volume']
for col in cols_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=cols_numeric, inplace=True)

# ═══════════════════════════════════════════════════════════════
# 2. INGENIERÍA DE FEATURES 
# ═══════════════════════════════════════════════════════════════
# CÁLCULOS LOGARÍTMICOS PARA ESTABILIZAR ESCALAS
df['returns'] = np.log(df['close'] / df['close'].shift(1))
df['gold_ret'] = np.log(df['gold_close'] / df['gold_close'].shift(1))
df['oil_ret']  = np.log(df['oil_close'] / df['oil_close'].shift(1))

df['hl_ratio'] = (df['high'] - df['low']) / df['close']
df['co_ratio'] = (df['close'] - df['open']) / df['close']
df['EMA_50'] = ta.trend.ema_indicator(df['close'], 50)
df['EMA_200'] = ta.trend.ema_indicator(df['close'], 200)
df['EMA_dist'] = (df['close'] - df['EMA_50']) / df['EMA_50'] # Distancia en % en vez de valor crudo
df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx() / 100 # Normalizado [0,1]
df['RSI'] = ta.momentum.rsi(df['close'], 14) / 100 # Normalizado [0,1]
df['MACD'] = ta.trend.macd_diff(df['close']) / df['close'] # MACD escalado al precio
df['ATR_pct'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14) / df['close'] # ATR en %
df['BB_width'] = (ta.volatility.bollinger_hband(df['close'], 20) - ta.volatility.bollinger_lband(df['close'], 20)) / df['close']
df['volatility'] = df['returns'].rolling(20).std()

# CÁLCULO SEGURO DEL VOLUMEN (Evitar división por 0)
rolling_vol_us500 = df['volume'].rolling(20).mean()
rolling_vol_gold = df['gold_volume'].rolling(20).mean()
rolling_vol_oil = df['oil_volume'].rolling(20).mean()

df['volume_ratio'] = np.where(rolling_vol_us500 > 0, df['volume'] / rolling_vol_us500, 1.0)
df['gold_vol_ratio'] = np.where(rolling_vol_gold > 0, df['gold_volume'] / rolling_vol_gold, 1.0)
df['oil_vol_ratio'] = np.where(rolling_vol_oil > 0, df['oil_volume'] / rolling_vol_oil, 1.0)

# ═══════════════════════════════════════════════════════════════
# 3. TRIPLE BARRIER 
# ═══════════════════════════════════════════════════════════════
print("Calculando Triple Barrier con spread...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Guardar ATR original crudo (no en porcentaje) SOLO para el SL y TP reales
df['ATR_RAW'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
df.dropna(inplace=True)

def triple_barrier_label(df, horizon=HORIZON, tp_mult=TARGET_ATR_MULT,
                         sl_mult=SL_ATR_MULT, spread=SPREAD_POINTS):
    target = pd.Series(0, index=df.index)
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    atr    = df['ATR_RAW'].values
    for i in range(len(df) - horizon):
        entry = closes[i] + spread
        tp_price = entry + tp_mult * atr[i]
        sl_price = entry - sl_mult * atr[i]
        touched_tp, touched_sl = False, False
        for j in range(1, horizon+1):
            if lows[i+j] <= sl_price:
                touched_sl = True
                break
            if highs[i+j] >= tp_price:
                touched_tp = True
                break
        if touched_tp and not touched_sl:
            target.iloc[i] = 1
    return target

df['target'] = triple_barrier_label(df)
df = df.iloc[:-HORIZON].copy()
print(f"Target positivo (Oportunidades reales): {df['target'].mean()*100:.1f}%")

# FEATURES (Usamos los valores transformados, no los crudos)
FEATURES = [
    'returns', 'hl_ratio', 'co_ratio', 'EMA_dist', 
    'ADX', 'RSI', 'MACD', 'ATR_pct', 'BB_width', 'volume_ratio', 'volatility',
    'gold_ret', 'gold_vol_ratio',
    'oil_ret', 'oil_vol_ratio'
]

# ═══════════════════════════════════════════════════════════════
# 4. DIVISIÓN TEMPORAL Y ESCALADO (StandardScaler)
# ═══════════════════════════════════════════════════════════════
train = df[df.index <= TRAIN_END_DATE].copy()
test  = df[df.index >= TEST_START_DATE].copy()
print(f"Train: {len(train)} velas, Test: {len(test)} velas")

# AQUI ESTA LA CURA PRINCIPAL: StandardScaler centra la media en 0 y achata los outliers
scaler = StandardScaler()
scaler.fit(train[FEATURES].values)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

def crear_secuencias(data, scaler, seq_len=SEQ_LENGTH):
    scaled = scaler.transform(data[FEATURES].values)
    # Clip manual para asegurar que ningún outlier loco sobreviva la estandarización
    scaled = np.clip(scaled, -5, 5) 
    X, y = [], []
    targets = data['target'].values
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

X_train, y_train = crear_secuencias(train, scaler)
X_val, y_val = crear_secuencias(test, scaler)
print(f"Forma de entrada a la red -> X_train: {X_train.shape}, X_val: {X_val.shape}")

# ═══════════════════════════════════════════════════════════════
# 5. ARQUITECTURA DEL MODELO (Reforzada)
# ═══════════════════════════════════════════════════════════════
input_layer = Input(shape=(SEQ_LENGTH, len(FEATURES)))

# LayerNormalization es infinitamente superior a BatchNormalization para LSTMs financieras
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
x = LayerNormalization()(x)
x = Dropout(0.2)(x)

x = LSTM(64, return_sequences=True)(x)
x = LayerNormalization()(x)
x = Dropout(0.2)(x)

# Atención aligerada
attention = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x = Add()([x, attention])
x = Flatten()(x)

x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)

# Clipnorm y LR bajo
optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)

model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=[
                  'accuracy', 
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall'),
                  tf.keras.metrics.AUC(name='auc')
              ])
model.summary()

early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=25, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

peso_0 = 1.0
peso_1 = (1 - df['target'].mean()) / df['target'].mean()
class_weight = {0: peso_0, 1: peso_1}
print(f"Class weights: {class_weight}")

# ═══════════════════════════════════════════════════════════════
# 6. ENTRENAMIENTO + CALIBRACIÓN
# ═══════════════════════════════════════════════════════════════
print("Entrenando modelo de Producción (con Class Weights)...")
model.fit(X_train, y_train, epochs=300, batch_size=256,
          validation_data=(X_val, y_val),
          class_weight=class_weight,
          callbacks=[early_stop, reduce_lr], verbose=1)
model.save(os.path.join(MODEL_DIR, 'model.keras'))

print("Calibrando probabilidades sobre conjunto de validación...")
y_pred_val = model.predict(X_val, verbose=0).flatten()
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_pred_val, y_val)
joblib.dump(iso_reg, os.path.join(MODEL_DIR, 'calibrador.pkl'))

# ═══════════════════════════════════════════════════════════════
# 7. BACKTEST REALISTA CON COMISIONES Y UMBRAL 85%
# ═══════════════════════════════════════════════════════════════
print("Ejecutando backtest de Producción...")
test_aligned = test.iloc[SEQ_LENGTH:].copy()
X_test_seq = []
for i in range(SEQ_LENGTH, len(test)):
    # OJO AQUI: Aplicamos la misma transformación al test y cortamos extremos
    scaled_row = scaler.transform(test[FEATURES].values[i-SEQ_LENGTH:i])
    scaled_row = np.clip(scaled_row, -5, 5)
    X_test_seq.append(scaled_row)

X_test_seq = np.array(X_test_seq)
probs = model.predict(X_test_seq, verbose=0).flatten()
probs_calibradas = iso_reg.predict(probs)
test_aligned['prob_lstm'] = probs_calibradas

capital = INITIAL_CAPITAL
equity = []
trades = []
pos_abierta = False
entry_price = sl_price = tp_price = lot_size = bars_held = 0

for idx in range(500, len(test_aligned)):
    ventana = test_aligned['prob_lstm'].iloc[idx-500:idx]
    umbral = np.percentile(ventana, LSTM_PERCENTIL)
    row = test_aligned.iloc[idx]

    if pos_abierta:
        bars_held += 1
        if row['low'] <= sl_price:
            exit_price = sl_price
            gross_pnl = (exit_price - entry_price) * lot_size * LOT_VALUE_PER_POINT
            comision_total = lot_size * COMMISSION_PER_LOT
            pnl = gross_pnl - comision_total
            capital += pnl
            trades[-1].update({'exit_time': test_aligned.index[idx],
                               'exit_price': exit_price, 'exit_reason': 'sl',
                               'pnl': pnl, 'capital_after': capital,
                               'bars_held': bars_held})
            pos_abierta = False
        elif row['high'] >= tp_price:
            exit_price = tp_price
            gross_pnl = (exit_price - entry_price) * lot_size * LOT_VALUE_PER_POINT
            comision_total = lot_size * COMMISSION_PER_LOT
            pnl = gross_pnl - comision_total
            capital += pnl
            trades[-1].update({'exit_time': test_aligned.index[idx],
                               'exit_price': exit_price, 'exit_reason': 'tp',
                               'pnl': pnl, 'capital_after': capital,
                               'bars_held': bars_held})
            pos_abierta = False
        elif bars_held >= HORIZON:
            exit_price = row['close']
            gross_pnl = (exit_price - entry_price) * lot_size * LOT_VALUE_PER_POINT
            comision_total = lot_size * COMMISSION_PER_LOT
            pnl = gross_pnl - comision_total
            capital += pnl
            trades[-1].update({'exit_time': test_aligned.index[idx],
                               'exit_price': exit_price, 'exit_reason': 'time_exit',
                               'pnl': pnl, 'capital_after': capital,
                               'bars_held': bars_held})
            pos_abierta = False
        continue

    # 2. Entrada (Filtro Técnico ORIGINAL)
    if (row['EMA_50'] > row['EMA_200'] and row['close'] > row['EMA_50'] and
        (row['ADX'] * 100) > 25 and (row['RSI'] * 100) < 65 and
        row['prob_lstm'] >= umbral):

        atr = row['ATR_RAW']
        sl = SL_ATR_MULT * atr
        tp = TARGET_ATR_MULT * atr
        if sl <= 0:
            continue
        
        riesgo_usd = capital * RISK_PER_TRADE
        lot_size = riesgo_usd / (sl * LOT_VALUE_PER_POINT)
        lot_size = max(0.01, np.floor(lot_size * 100) / 100)

        entry_price = row['close'] + SPREAD_POINTS
        sl_price = entry_price - sl
        tp_price = entry_price + tp

        trades.append({
            'entry_time': test_aligned.index[idx],
            'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
            'lotes': lot_size, 'capital_before': capital
        })
        pos_abierta = True
        bars_held = 0

    equity.append({'time': test_aligned.index[idx], 'equity': capital})

if pos_abierta:
    exit_price = test_aligned.iloc[-1]['close']
    gross_pnl = (exit_price - entry_price) * lot_size * LOT_VALUE_PER_POINT
    comision_total = lot_size * COMMISSION_PER_LOT
    pnl = gross_pnl - comision_total
    capital += pnl
    trades[-1].update({'exit_time': test_aligned.index[-1],
                       'exit_price': exit_price, 'exit_reason': 'close_final',
                       'pnl': pnl, 'capital_after': capital})

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity).set_index('time')

# ═══════════════════════════════════════════════════════════════
# 8. MÉTRICAS FINALES Y REPORTE
# ═══════════════════════════════════════════════════════════════
if len(trades_df) == 0:
    print("\n⚠️ Sin operaciones. El mercado no dio oportunidades bajo estos parámetros.")
else:
    win = trades_df[trades_df['pnl'] > 0]
    loss = trades_df[trades_df['pnl'] < 0]
    total = len(trades_df)
    win_rate = len(win) / total * 100
    profit_factor = win['pnl'].sum() / abs(loss['pnl'].sum()) if len(loss) > 0 else np.inf
    retorno = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    dd = (equity_df['equity'].cummax() - equity_df['equity']).max() / INITIAL_CAPITAL * 100
    daily_ret = equity_df['equity'].pct_change().dropna()
    sharpe_annual = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() != 0 else 0

    print("\n" + "="*50)
    print(" RESULTADOS APEX FUSION (VERSIÓN PRODUCCIÓN ICMARKETS)")
    print("="*50)
    print(f"Capital final:    ${capital:,.2f}")
    print(f"Retorno Neto:     {retorno:.2f}%")
    print(f"Drawdown máximo:  {dd:.2f}%")
    print(f"Operaciones:      {total}")
    print(f"Tasa de aciertos: {win_rate:.1f}%")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Ganancia media:   ${win['pnl'].mean():,.2f}")
    if len(loss) > 0:
        print(f"Pérdida media:    ${loss['pnl'].mean():,.2f}")
    print(f"Sharpe Anual:     {sharpe_annual:.2f}")
    print("="*50)

    trades_df.to_csv(os.path.join(MODEL_DIR, 'trades.csv'), index=False)
    equity_df.to_csv(os.path.join(MODEL_DIR, 'equity.csv'))
    print(f"Resultados guardados en {MODEL_DIR}/")

print("Pipeline completado.")
