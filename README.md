# Ritta's Website

This is a static website hosted on GitHub Pages.

## Project Structure

- `static_site/` - Contains the main website files
  - `index.html` - Home page
  - `analysis.html` - Analysis page
  - `css/` - Stylesheets
  - `js/` - JavaScript files
  - `data/` - JSON data files

## Setup

1. The site is automatically deployed to GitHub Pages
2. Visit https://ritta1125.github.io/ritta1125/ to view the site

## Features

- Responsive design
- Interactive chat interface
- Policy recommendations display
- Dynamic content loading

# 폐교 위험 지역과 다문화 가정 분포 간의 상관관계 분석

## 프로젝트 개요
본 프로젝트는 전국 단위의 공간 데이터를 기반으로 폐교 위험도와 다문화 가정 분포 간의 관계를 분석하고, 이를 통해 지역 기반 교육 정책, 학교 통폐합 정책, 다문화 지원 정책의 데이터 기반 의사결정을 지원합니다.

## 주요 연구 질문
1. 폐교 발생률과 학급 수 감소는 어떤 지역에서 두드러지는가?
2. 다문화 가정은 어떤 지역(도시/농촌/특정 시군구)에 집중되어 있는가?
3. 폐교 위기 지역과 다문화 가정 밀집 지역 간의 공간적 상관관계는 존재하는가?
4. 다문화 학생 비율이 높은 학교는 폐교 또는 통폐합 위험에 더 노출되는가?
5. 지역 간 교육 인프라 격차는 어떻게 분포되어 있는가?
6. 정책 결정자가 우선 지원해야 할 지역은 어디인가?

## 프로젝트 구조
```
.
├── data/                    # 데이터 저장 디렉토리
│   ├── raw/                # 원본 데이터
│   ├── processed/          # 처리된 데이터
│   └── geospatial/         # 지리정보 데이터
├── notebooks/              # Jupyter 노트북
├── src/                    # 소스 코드
│   ├── data_collection/    # 데이터 수집 스크립트
│   ├── data_processing/    # 데이터 처리 스크립트
│   ├── analysis/           # 분석 스크립트
│   └── visualization/      # 시각화 스크립트
├── docs/                   # 문서
└── reports/                # 분석 보고서
```

## 설치 방법
1. Python 3.8 이상 설치
2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```
3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 데이터 수집
- 교육통계서비스(KESS)
- 학교알리미
- 교육부/지역 교육청 공시
- 통계청, KOSIS
- 국토지리정보원, VWorld

## 분석 방법
1. 데이터 전처리 및 통합
2. 폐교위험지수 및 다문화비중지수 생성
3. 공간 클러스터링 분석
4. 상관관계 분석
5. 시각화 및 대시보드 구축

## 기대효과
- 정책 설계 지원
- 교육 형평성 강화
- 데이터 기반 행정 가능
- 시민 접근성 향상 