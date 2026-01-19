"""
Streamlit 기반 GUI 애플리케이션
주식/코인 가격 생성기 웹 인터페이스
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
from go_stock import StockPriceGenerator
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="주식/코인 가격 생성기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'generated_prices' not in st.session_state:
    st.session_state.generated_prices = None
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {'current_epoch': 0, 'total_epochs': 0, 'loss': 0.0}

# 제목 및 설명
st.title("📈 LLM 기반 주식/코인 가격 생성기")
st.markdown("""
이 애플리케이션은 Transformer 모델을 사용하여 과거 가격 데이터와 뉴스 정보를 결합하여 
미래 가격을 예측하고 생성합니다.
""")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 종목 선택
    asset_type = st.radio("자산 유형", ["암호화폐", "주식"], index=0)
    
    if asset_type == "암호화폐":
        symbol = st.text_input(
            "심볼 입력",
            value="BTC-USD",
            help="예: BTC-USD, ETH-USD, XRP-USD"
        )
        popular_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
    else:
        symbol = st.text_input(
            "심볼 입력",
            value="AAPL",
            help="예: AAPL, TSLA, GOOGL, MSFT"
        )
        popular_symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
    
    # 인기 종목 빠른 선택
    st.markdown("**인기 종목:**")
    cols = st.columns(2)
    for i, pop_symbol in enumerate(popular_symbols[:4]):
        if cols[i % 2].button(pop_symbol, key=f"btn_{pop_symbol}"):
            symbol = pop_symbol
            st.rerun()
    
    st.divider()
    
    # 데이터 수집 설정
    st.subheader("📊 데이터 설정")
    period = st.selectbox(
        "기간",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
        help="데이터 수집 기간"
    )
    
    interval = st.selectbox(
        "시간 간격",
        options=["1h", "1d", "1wk", "1mo"],
        index=1,  # 기본값을 1d로 변경 (더 안정적)
        help="데이터 수집 간격 (1h: 최대 60일, 1d: 제한 없음)"
    )
    
    st.divider()
    
    # 학습 설정
    st.subheader("🤖 모델 학습 설정")
    epochs = st.slider("Epochs (학습 횟수)", 10, 200, 50, 10)
    batch_size = st.slider("Batch Size", 8, 64, 32, 8)
    learning_rate = st.selectbox(
        "Learning Rate",
        options=[0.0001, 0.001, 0.01],
        index=1,
        format_func=lambda x: f"{x:.4f}"
    )
    
    st.divider()
    
    # 생성 설정
    st.subheader("🎯 가격 생성 설정")
    prediction_steps = st.slider("예측 단계 수", 1, 50, 10, 1)
    
    st.divider()
    
    # 모델 관리
    st.subheader("💾 모델 관리")
    if st.button("모델 초기화", width='stretch'):
        st.session_state.generator = StockPriceGenerator()
        st.session_state.model_trained = False
        st.success("모델이 초기화되었습니다!")
    
    if st.button("모델 로드", width='stretch'):
        model_path = st.text_input("모델 경로", value="price_generator_model.pt")
        try:
            st.session_state.generator = StockPriceGenerator(model_path=model_path)
            st.session_state.model_trained = True
            st.success(f"모델이 로드되었습니다: {model_path}")
        except Exception as e:
            st.error(f"모델 로드 실패: {str(e)}")

# 메인 컨텐츠 영역
tab1, tab2, tab3, tab4 = st.tabs(["📥 데이터 수집", "🎓 모델 학습", "🔮 가격 생성", "📊 결과 분석"])

# 탭 1: 데이터 수집
with tab1:
    st.header("데이터 수집")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("📥 데이터 수집 시작", type="primary", width='stretch'):
            if not symbol:
                st.error("심볼을 입력해주세요!")
            else:
                with st.spinner(f"{symbol} 데이터 수집 중..."):
                    try:
                        # 생성기 초기화
                        if st.session_state.generator is None:
                            st.session_state.generator = StockPriceGenerator()
                        
                        # 가격 데이터 수집
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("가격 데이터 수집 중...")
                        price_data = st.session_state.generator.collect_price_data(
                            symbol, period=period, interval=interval
                        )
                        progress_bar.progress(50)
                        
                        # 뉴스 데이터 수집 (all: yfinance + API 키 있는 소스 전부, 날짜순·중복 제거)
                        status_text.text("뉴스 데이터 수집 중...")
                        news_data = st.session_state.generator.collect_news_data(symbol, days=365, news_source='all')
                        progress_bar.progress(100)
                        
                        # 뉴스 데이터가 없으면 더미 데이터 생성
                        if len(news_data) == 0 and not price_data.empty and 'datetime' in price_data.columns:
                            st.warning("뉴스 데이터가 없습니다. 가격 데이터만 사용합니다.")
                            news_data = [
                                {
                                    'timestamp': price_data.iloc[i]['datetime'],
                                    'title': '',
                                    'content': '',
                                    'sentiment': 0
                                }
                                for i in range(len(price_data))
                            ]
                        
                        st.session_state.price_data = price_data
                        st.session_state.news_data = news_data
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ 데이터 수집 완료!")
                        st.info(f"가격 데이터: {len(price_data)}개\n뉴스 데이터: {len(news_data)}개")
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"데이터 수집 실패")
                        st.error(error_msg)
                        st.info("💡 **팁**:\n- 1h 간격은 최대 60일까지만 지원됩니다\n- 더 긴 기간을 원하면 '1d' (일봉) 간격을 사용해보세요\n- 심볼이 올바른지 확인해주세요")
    
    with col2:
        if st.session_state.price_data is not None:
            st.metric("수집된 데이터", len(st.session_state.price_data))
            if st.session_state.news_data:
                st.metric("뉴스 데이터", len(st.session_state.news_data))
    
    # 데이터 미리보기
    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
        st.subheader("데이터 미리보기")
        
        # datetime 컬럼 확인
        if 'datetime' not in st.session_state.price_data.columns:
            st.error("데이터에 'datetime' 컬럼이 없습니다. 데이터 수집을 다시 시도해주세요.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    st.session_state.price_data.head(10),
                    width='stretch'
                )
            
            with col2:
                # 기본 통계
                if 'close' in st.session_state.price_data.columns:
                    price_stats = st.session_state.price_data['close'].describe()
                    st.dataframe(price_stats, width='stretch')
            
            # 가격 차트
            if 'close' in st.session_state.price_data.columns and 'datetime' in st.session_state.price_data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.price_data['datetime'],
                    y=st.session_state.price_data['close'],
                    mode='lines',
                    name='종가',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title=f"{symbol} 가격 차트",
                    xaxis_title="날짜",
                    yaxis_title="가격 (USD)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch')

# 탭 2: 모델 학습
with tab2:
    st.header("모델 학습")
    
    if st.session_state.price_data is None:
        st.warning("⚠️ 먼저 데이터를 수집해주세요!")
    else:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button("🎓 학습 시작", type="primary", width='stretch'):
                with st.spinner("모델 학습 중..."):
                    try:
                        # 진행 상황 표시를 위한 컨테이너
                        progress_container = st.container()
                        
                        # 생성기 초기화
                        if st.session_state.generator is None:
                            st.session_state.generator = StockPriceGenerator()
                        
                        # 학습 진행 상황 표시를 위한 커스텀 콜백 구현 필요
                        # 여기서는 간단하게 학습만 진행
                        st.session_state.generator.train(
                            st.session_state.price_data,
                            st.session_state.news_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=learning_rate
                        )
                        
                        st.session_state.model_trained = True
                        st.success(f"✅ 학습 완료! ({epochs} epochs)")
                        
                    except Exception as e:
                        st.error(f"학습 실패: {str(e)}")
        
        with col2:
            if st.button("💾 모델 저장", width='stretch'):
                if st.session_state.model_trained:
                    try:
                        model_path = f"model_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                        st.session_state.generator.save_model(model_path)
                        st.success(f"모델이 저장되었습니다: {model_path}")
                    except Exception as e:
                        st.error(f"저장 실패: {str(e)}")
                else:
                    st.warning("먼저 모델을 학습시켜주세요!")
        
        with col3:
            if st.session_state.model_trained:
                st.success("✅ 학습됨")
            else:
                st.info("❌ 미학습")
        
        # 학습 상태 정보
        if st.session_state.model_trained:
            st.info(f"""
            **학습 완료 정보:**
            - Epochs: {epochs}
            - Batch Size: {batch_size}
            - Learning Rate: {learning_rate}
            - 데이터 크기: {len(st.session_state.price_data)}
            """)

# 탭 3: 가격 생성
with tab3:
    st.header("가격 생성")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ 먼저 모델을 학습시켜주세요!")
    elif st.session_state.price_data is None:
        st.warning("⚠️ 먼저 데이터를 수집해주세요!")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔮 가격 생성", type="primary", width='stretch'):
                with st.spinner(f"{prediction_steps}단계 가격 생성 중..."):
                    try:
                        # 최근 데이터 사용
                        recent_data = st.session_state.price_data.tail(100)
                        recent_news = st.session_state.news_data[-10:] if st.session_state.news_data else []
                        
                        generated = st.session_state.generator.generate_price(
                            recent_data,
                            recent_news,
                            steps=prediction_steps
                        )
                        
                        st.session_state.generated_prices = generated
                        
                        st.success(f"✅ {prediction_steps}단계 가격 생성 완료!")
                        
                        # 생성된 가격 표시
                        col_a, col_b, col_c = st.columns(3)
                        if len(generated) > 0:
                            col_a.metric("현재 가격", f"${recent_data['close'].iloc[-1]:.2f}")
                            col_b.metric("예측 가격 (1단계)", f"${generated[0]:.2f}")
                            if len(generated) > 1:
                                change = ((generated[-1] - generated[0]) / generated[0]) * 100
                                col_c.metric("예상 변화율", f"{change:.2f}%")
                        
                    except Exception as e:
                        st.error(f"가격 생성 실패: {str(e)}")
        
        # 생성 결과 시각화
        if st.session_state.generated_prices is not None:
            st.subheader("생성 결과")
            
            # 데이터 유효성 검사
            if st.session_state.price_data.empty or 'datetime' not in st.session_state.price_data.columns:
                st.error("유효한 가격 데이터가 없습니다.")
            else:
                # 과거 + 생성 가격 결합
                historical_prices = st.session_state.price_data['close'].tail(50).values
                last_date = pd.to_datetime(st.session_state.price_data['datetime'].iloc[-1])
                future_dates = pd.date_range(
                    start=last_date + timedelta(hours=1 if interval == '1h' else 1),
                    periods=len(st.session_state.generated_prices),
                    freq='H' if interval == '1h' else 'D'
                )
                
                historical_dates = pd.to_datetime(st.session_state.price_data['datetime'].tail(50))
                
                fig = make_subplots(rows=1, cols=1)
                
                # 과거 데이터
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_prices,
                    mode='lines',
                    name='과거 가격',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # 생성된 데이터
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=st.session_state.generated_prices,
                    mode='lines+markers',
                    name='생성된 가격',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # 연결점
                fig.add_trace(go.Scatter(
                    x=[last_date, future_dates[0]],
                    y=[historical_prices[-1], st.session_state.generated_prices[0]],
                    mode='lines',
                    name='연결',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"{symbol} 가격 예측",
                    xaxis_title="날짜",
                    yaxis_title="가격 (USD)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(x=0, y=1)
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # 생성된 가격 테이블
                with st.expander("생성된 가격 상세"):
                    df_generated = pd.DataFrame({
                        'Step': range(1, len(st.session_state.generated_prices) + 1),
                        'Date': future_dates,
                        'Price': st.session_state.generated_prices,
                        'Change': np.concatenate([
                            [0],
                            np.diff(st.session_state.generated_prices)
                        ]),
                        'Change %': np.concatenate([
                            [0],
                            (np.diff(st.session_state.generated_prices) / st.session_state.generated_prices[:-1]) * 100
                        ])
                    })
                    st.dataframe(df_generated, width='stretch')

# 탭 4: 결과 분석
with tab4:
    st.header("결과 분석")
    
    if st.session_state.generated_prices is None:
        st.info("가격을 생성한 후 결과를 분석할 수 있습니다.")
    elif st.session_state.price_data is None or st.session_state.price_data.empty:
        st.warning("가격 데이터가 없습니다.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        generated = st.session_state.generated_prices
        if 'close' in st.session_state.price_data.columns:
            current_price = st.session_state.price_data['close'].iloc[-1]
        else:
            current_price = 0
            st.warning("가격 데이터에 'close' 컬럼이 없습니다.")
        
        with col1:
            st.metric("현재 가격", f"${current_price:.2f}")
        
        with col2:
            st.metric("예측 시작 가격", f"${generated[0]:.2f}")
        
        with col3:
            st.metric("예측 종료 가격", f"${generated[-1]:.2f}")
        
        with col4:
            total_change = ((generated[-1] - current_price) / current_price) * 100
            st.metric("총 변화율", f"{total_change:.2f}%")
        
        # 통계 분석
        st.subheader("통계 분석")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("**예측 가격 통계**")
            stats_df = pd.DataFrame({
                '통계량': ['평균', '최대값', '최소값', '표준편차', '변동계수'],
                '값': [
                    f"${np.mean(generated):.2f}",
                    f"${np.max(generated):.2f}",
                    f"${np.min(generated):.2f}",
                    f"${np.std(generated):.2f}",
                    f"{(np.std(generated) / np.mean(generated) * 100):.2f}%"
                ]
            })
            st.dataframe(stats_df, width='stretch', hide_index=True)
        
        with analysis_col2:
            st.markdown("**가격 변화 추이**")
            changes = np.diff(generated)
            changes_df = pd.DataFrame({
                'Step': range(1, len(changes) + 1),
                '변화량': changes,
                '변화율(%)': (changes / generated[:-1]) * 100
            })
            st.dataframe(changes_df, width='stretch')
        
        # 변동성 분석
        st.subheader("변동성 분석")
        volatility = np.std(generated) / np.mean(generated) * 100
        st.metric("예측 변동성 (CV)", f"{volatility:.2f}%")
        
        # 히스토그램
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=generated,
            nbinsx=20,
            name='예측 가격 분포'
        ))
        fig_hist.update_layout(
            title="예측 가격 분포",
            xaxis_title="가격",
            yaxis_title="빈도",
            height=300
        )
        st.plotly_chart(fig_hist, width='stretch')

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>LLM 기반 주식/코인 가격 생성기 | 
    ⚠️ 이 결과는 참고용이며 실제 투자 결정에 사용하기 전 전문가 자문을 받으세요.</p>
</div>
""", unsafe_allow_html=True)
