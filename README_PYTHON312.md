# Python 3.12 사용 가이드

Python 3.12가 설치되어 있습니다. Python 3.12를 사용하는 방법:

## 방법 1: py launcher 사용 (권장)

```bash
# Python 3.12로 Streamlit 실행
py -3.12 -m streamlit run gui.py

# 또는 배치 파일 사용
run_with_python312.bat
```

## 방법 2: Python 3.12를 기본값으로 설정

### Windows 환경 변수 설정:

1. `Win + R` → `sysdm.cpl` 입력 → Enter
2. "고급" 탭 → "환경 변수" 클릭
3. "사용자 변수" 또는 "시스템 변수"에서 `Path` 선택 → "편집"
4. Python 3.12 경로를 Python 3.9 위로 이동 (우선순위 설정)

Python 3.12 경로 (예시):
```
C:\Users\pstcw\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\
```

## 방법 3: 가상환경 생성 (프로젝트별)

```bash
# Python 3.12로 가상환경 생성
py -3.12 -m venv venv312

# 가상환경 활성화
venv312\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# Streamlit 실행
streamlit run gui.py
```

## 확인

```bash
# Python 버전 확인
py -3.12 --version

# Python 3.12로 패키지 설치
py -3.12 -m pip install -r requirements.txt
```
