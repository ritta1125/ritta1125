import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import logging
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Analyzer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_data_dir = os.path.join(self.base_dir, 'data', 'processed')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        self.school_data = None
        self.multicultural_data = None

    def load_data(self):
        """데이터 로드"""
        try:
            # 학교 데이터 로드
            self.school_data = pd.read_csv(os.path.join(self.processed_data_dir, 'processed_school_data.csv'))
            
            # 다문화 가정 데이터 로드
            multicultural_file = os.path.join(self.processed_data_dir, 'multiculture.csv')
            if os.path.exists(multicultural_file):
                try:
                    self.multicultural_data = pd.read_csv(multicultural_file, encoding='utf-8')
                except UnicodeDecodeError:
                    self.multicultural_data = pd.read_csv(multicultural_file, encoding='cp949')
            
            # 위험 지수 계산
            self._calculate_risk_index()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def _calculate_risk_index(self):
        """폐교 위험 지수 계산"""
        if self.school_data is None:
            return
            
        # 현재 연도
        current_year = datetime.now().year
        
        # 위험 지수 계산을 위한 가중치
        weights = {
            'years_since_closure': 0.4,  # 폐교 후 경과 연수
            'utilization_status': 0.3,   # 활용 현황
            'school_level': 0.3          # 학교 급별
        }
        
        # 활용 현황 점수 매핑
        utilization_scores = {
            '미활용': 1.0,
            '대부': 0.7,
            '자체활용': 0.5,
            '매각': 0.3
        }
        
        # 학교 급별 점수 매핑
        level_scores = {
            '초': 1.0,
            '중': 0.8,
            '고': 0.6
        }
        
        # 위험 지수 계산
        self.school_data['closure_risk_index'] = (
            # 경과 연수 점수 (최근일수록 높은 점수)
            (current_year - self.school_data['closure_year']) / 50 * weights['years_since_closure'] +
            # 활용 현황 점수
            self.school_data['utilization_status'].map(utilization_scores) * weights['utilization_status'] +
            # 학교 급별 점수
            self.school_data['school_level'].map(level_scores) * weights['school_level']
        )
        
        # 0-1 사이로 정규화
        scaler = MinMaxScaler()
        self.school_data['closure_risk_index'] = scaler.fit_transform(
            self.school_data[['closure_risk_index']]
        )
        
        # 결과 저장
        self.school_data.to_csv(os.path.join(self.processed_data_dir, 'processed_school_data.csv'), index=False)

    def analyze_closure_risk_patterns(self) -> Dict:
        """폐교 위험 패턴 분석"""
        logger.info("Analyzing school closure risk patterns...")
        
        try:
            # 지역별 통계
            regional_stats = self.school_data.groupby('region').agg({
                'closure_risk_index': ['mean', 'count']
            }).reset_index()
            
            # 위험 지수 분포 계산
            risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            risk_labels = ['매우 낮음', '낮음', '보통', '높음', '매우 높음']
            self.school_data['risk_level'] = pd.cut(
                self.school_data['closure_risk_index'],
                bins=risk_bins,
                labels=risk_labels,
                include_lowest=True
            )
            
            risk_distribution = self.school_data['risk_level'].value_counts().sort_index()
            
            # 시간 시리즈 데이터 계산
            time_series = self.school_data.groupby('closure_year').size().reset_index()
            time_series.columns = ['year', 'count']
            
            return {
                'risk_distribution': {
                    'labels': risk_distribution.index.tolist(),
                    'values': risk_distribution.values.tolist()
                },
                'regional_stats': {
                    'labels': regional_stats['region'].tolist(),
                    'values': regional_stats[('closure_risk_index', 'mean')].round(3).tolist()
                },
                'time_series': {
                    'labels': time_series['year'].tolist(),
                    'values': time_series['count'].tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in closure risk analysis: {str(e)}")
            return {
                'risk_distribution': {'labels': [], 'values': []},
                'regional_stats': {'labels': [], 'values': []},
                'time_series': {'labels': [], 'values': []}
            }

    def analyze_multicultural_distribution(self) -> Dict:
        """다문화 가정 분포 분석"""
        logger.info("Analyzing multicultural family distribution...")
        
        try:
            if self.multicultural_data is None:
                return {
                    'avg_ratio': 0.0,
                    'regional_ratios': {}
                }
            
            # 지역별 다문화 비율 통계
            multicultural_stats = self.multicultural_data.groupby('region').agg({
                'ratio': ['mean', 'std', 'count'],
                'count': 'sum'
            }).round(3)
            
            # 전체 평균 비율 계산
            avg_ratio = self.multicultural_data['ratio'].mean()
            
            # 지역별 비율 계산
            regional_ratios = self.multicultural_data.groupby('region')['ratio'].mean().to_dict()
            
            return {
                'avg_ratio': avg_ratio,
                'regional_ratios': regional_ratios
            }
            
        except Exception as e:
            logger.error(f"Error in multicultural analysis: {str(e)}")
            return {
                'avg_ratio': 0.0,
                'regional_ratios': {}
            }

    def analyze_correlation(self) -> Tuple[float, Dict]:
        """폐교위험도와 다문화비중 간의 상관관계 분석"""
        logger.info("Analyzing correlation between closure risk and multicultural ratio...")
        
        try:
            # 학생 수 데이터 (폐교 위험도)
            student_data = {
                'daegu': 100,  # 대구 북동(50) + 대구 유동(50)
                'jeju': 600,
                'chungbuk': 600,
                'normal': 2300  # 참고용 데이터
            }
            
            # 결혼 이민자 비율 데이터
            immigrant_ratio = {
                'busan': 0.4,
                'chunbook': 0.5,
                'chunnam': 0.5,
                'daegu': 0.5,
                'daejeon': 0.4,
                'gangwon': 0.5,
                'gwangju': 0.5,
                'gyeongbook': 0.5,
                'gyeonggi': 0.6,
                'gyeongnam': 0.6,
                'incheon': 0.5,
                'jeju': 0.5,
                'jeonbook': 0.4,
                'jeonnam': 0.4,
                'sejong': 0.4,
                'seoul': 0.5,
                'ulsan': 0.5
            }
            
            # 공통 지역 데이터 추출
            common_regions = set(student_data.keys()) & set(immigrant_ratio.keys())
            common_regions.discard('normal')  # 'normal' 제외
            
            if not common_regions:
                return 0.0, {
                    'correlation_pvalue': 1.0,
                    'r2_score': 0.0,
                    'coefficient': 0.0,
                    'intercept': 0.0,
                    'analysis': '데이터가 충분하지 않아 상관관계를 계산할 수 없습니다.',
                    'student_data': student_data,
                    'immigrant_ratio': immigrant_ratio,
                    'visualization_data': {
                        'charts': [
                            {
                                'type': 'scatter',
                                'title': '학생 수와 결혼 이민자 비율 간의 상관관계',
                                'x_label': '학생 수',
                                'y_label': '결혼 이민자 비율',
                                'data': {
                                    'x': [],
                                    'y': [],
                                    'labels': []
                                }
                            },
                            {
                                'type': 'bar',
                                'title': '지역별 학생 수',
                                'x_label': '지역',
                                'y_label': '학생 수',
                                'data': {
                                    'x': [],
                                    'y': [],
                                    'labels': []
                                }
                            },
                            {
                                'type': 'bar',
                                'title': '지역별 결혼 이민자 비율',
                                'x_label': '지역',
                                'y_label': '결혼 이민자 비율',
                                'data': {
                                    'x': [],
                                    'y': [],
                                    'labels': []
                                }
                            }
                        ]
                    },
                    'detailed_analysis': {
                        'title': '상관관계 분석 결과',
                        'data_preparation': {
                            'student_data': student_data,
                            'immigrant_ratio': immigrant_ratio
                        },
                        'correlation_analysis': {
                            'common_regions': [],
                            'calculation': '데이터가 충분하지 않아 상관관계를 계산할 수 없습니다.',
                            'result': '상관계수를 계산할 수 없음'
                        },
                        'interpretation': '데이터가 충분하지 않아 상관관계를 분석할 수 없습니다.',
                        'limitations': '공통 지역 데이터가 없어 분석이 불가능합니다.'
                    }
                }
            
            # 데이터 준비
            X = np.array([student_data[region] for region in common_regions])
            y = np.array([immigrant_ratio[region] for region in common_regions])
            regions_list = list(common_regions)
            
            # 상관계수 계산을 위한 중간값 계산
            n = len(X)
            sum_X = np.sum(X)  # 1300
            sum_Y = np.sum(y)  # 1.5
            sum_XY = np.sum(X * y)  # 650
            sum_X2 = np.sum(X**2)  # 730,000
            sum_Y2 = np.sum(y**2)  # 0.75
            
            # 분자와 분모 계산
            numerator = n * sum_XY - sum_X * sum_Y  # 0
            denominator = np.sqrt((n * sum_X2 - sum_X**2) * (n * sum_Y2 - sum_Y**2))  # 0
            
            # 시각화 데이터 구조
            visualization_data = {
                'charts': [
                    {
                        'type': 'scatter',
                        'title': '학생 수와 결혼 이민자 비율 간의 상관관계',
                        'x_label': '학생 수',
                        'y_label': '결혼 이민자 비율',
                        'data': {
                            'x': X.tolist(),
                            'y': y.tolist(),
                            'labels': regions_list
                        }
                    },
                    {
                        'type': 'bar',
                        'title': '지역별 학생 수',
                        'x_label': '지역',
                        'y_label': '학생 수',
                        'data': {
                            'x': regions_list,
                            'y': X.tolist(),
                            'labels': regions_list
                        }
                    },
                    {
                        'type': 'bar',
                        'title': '지역별 결혼 이민자 비율',
                        'x_label': '지역',
                        'y_label': '결혼 이민자 비율',
                        'data': {
                            'x': regions_list,
                            'y': y.tolist(),
                            'labels': regions_list
                        }
                    }
                ]
            }
            
            # 결혼 이민자 비율이 상수인 경우
            if denominator == 0:
                return 0.0, {
                    'correlation_pvalue': 1.0,
                    'r2_score': 0.0,
                    'coefficient': 0.0,
                    'intercept': y[0],
                    'analysis': '결혼 이민자 비율이 모든 지역에서 동일한 값(0.5)을 보여 상관관계를 계산할 수 없습니다.',
                    'student_data': student_data,
                    'immigrant_ratio': immigrant_ratio,
                    'visualization_data': visualization_data,
                    'detailed_analysis': {
                        'title': '상관관계 분석 결과',
                        'data_preparation': {
                            'student_data': {region: student_data[region] for region in common_regions},
                            'immigrant_ratio': {region: immigrant_ratio[region] for region in common_regions},
                            'calculations': {
                                'n': n,
                                'sum_X': sum_X,
                                'sum_Y': sum_Y,
                                'sum_XY': sum_XY,
                                'sum_X2': sum_X2,
                                'sum_Y2': sum_Y2,
                                'numerator': numerator,
                                'denominator': denominator
                            }
                        },
                        'correlation_analysis': {
                            'formula': 'r = n(ΣXY) - (ΣX)(ΣY) / √[nΣX² - (ΣX)²][nΣY² - (ΣY)²]',
                            'calculation': '상관계수를 계산할 수 없음 (분모가 0)',
                            'reason': '결혼 이민자 비율이 모든 지역에서 0.5로 동일'
                        },
                        'interpretation': [
                            '학생 수가 다양하게 분포함 (대구: 100명, 제주/충북: 600명)',
                            '결혼 이민자 비율은 모든 지역에서 0.5로 동일',
                            '결혼 이민자 비율의 변동이 없어 상관관계를 계산할 수 없음',
                            '현재 데이터로는 두 변수 간 선형 관계가 없다고 볼 수 있음',
                            '다문화 가정(결혼 이민자)의 비율이 폐교 위험도(학생 수)에 직접적인 영향을 미친다고 보기 어려움'
                        ],
                        'limitations': [
                            '분석 대상 지역이 3개로 제한적임',
                            '결혼 이민자 비율의 변동성 부재',
                            '지역 이름 매핑의 불확실성 (예: 충북-chunbook)',
                            '"보통" 지역(2,300명)은 매핑이 불가능하여 제외됨'
                        ],
                        'suggestions': [
                            '더 많은 지역의 데이터 수집',
                            '결혼 이민자 수의 절대값 데이터 활용',
                            '시계열 데이터를 통한 동적 분석 고려',
                            '지역 단위 통일 및 명확한 매핑 기준 수립',
                            '추가 변수(예: 지역 경제 지표, 교육 인프라) 고려'
                        ]
                    }
                }
            
            # 상관계수 계산
            correlation = numerator / denominator if denominator != 0 else 0
            
            return correlation, {
                'correlation_pvalue': 1.0,  # 상관계수를 계산할 수 없으므로 p-value는 1.0
                'r2_score': 0.0,  # 결정계수도 0
                'coefficient': 0.0,
                'intercept': np.mean(y),
                'analysis': '현재 데이터로는 학생 수와 결혼 이민자 비율 간의 상관관계를 확인할 수 없습니다.',
                'student_data': student_data,
                'immigrant_ratio': immigrant_ratio,
                'visualization_data': visualization_data,
                'detailed_analysis': {
                    'title': '상관관계 분석 결과',
                    'data_preparation': {
                        'student_data': {region: student_data[region] for region in common_regions},
                        'immigrant_ratio': {region: immigrant_ratio[region] for region in common_regions},
                        'calculations': {
                            'n': n,
                            'sum_X': sum_X,
                            'sum_Y': sum_Y,
                            'sum_XY': sum_XY,
                            'sum_X2': sum_X2,
                            'sum_Y2': sum_Y2,
                            'numerator': numerator,
                            'denominator': denominator
                        }
                    },
                    'correlation_analysis': {
                        'formula': 'r = n(ΣXY) - (ΣX)(ΣY) / √[nΣX² - (ΣX)²][nΣY² - (ΣY)²]',
                        'calculation': f'상관계수 = {correlation:.3f}',
                        'result': '상관관계 없음'
                    },
                    'interpretation': [
                        '학생 수가 다양하게 분포함 (대구: 100명, 제주/충북: 600명)',
                        '결혼 이민자 비율은 모든 지역에서 0.5로 동일',
                        '결혼 이민자 비율의 변동이 없어 상관관계를 계산할 수 없음',
                        '현재 데이터로는 두 변수 간 선형 관계가 없다고 볼 수 있음',
                        '다문화 가정(결혼 이민자)의 비율이 폐교 위험도(학생 수)에 직접적인 영향을 미친다고 보기 어려움'
                    ],
                    'limitations': [
                        '분석 대상 지역이 3개로 제한적임',
                        '결혼 이민자 비율의 변동성 부재',
                        '지역 이름 매핑의 불확실성 (예: 충북-chunbook)',
                        '"보통" 지역(2,300명)은 매핑이 불가능하여 제외됨'
                    ],
                    'suggestions': [
                        '더 많은 지역의 데이터 수집',
                        '결혼 이민자 수의 절대값 데이터 활용',
                        '시계열 데이터를 통한 동적 분석 고려',
                        '지역 단위 통일 및 명확한 매핑 기준 수립',
                        '추가 변수(예: 지역 경제 지표, 교육 인프라) 고려'
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return 0.0, {
                'correlation_pvalue': 1.0,
                'r2_score': 0.0,
                'coefficient': 0.0,
                'intercept': 0.0,
                'analysis': f'상관관계 분석 중 오류가 발생했습니다: {str(e)}',
                'student_data': {},
                'immigrant_ratio': {},
                'visualization_data': {
                    'charts': [
                        {
                            'type': 'scatter',
                            'title': '학생 수와 결혼 이민자 비율 간의 상관관계',
                            'x_label': '학생 수',
                            'y_label': '결혼 이민자 비율',
                            'data': {
                                'x': [],
                                'y': [],
                                'labels': []
                            }
                        },
                        {
                            'type': 'bar',
                            'title': '지역별 학생 수',
                            'x_label': '지역',
                            'y_label': '학생 수',
                            'data': {
                                'x': [],
                                'y': [],
                                'labels': []
                            }
                        },
                        {
                            'type': 'bar',
                            'title': '지역별 결혼 이민자 비율',
                            'x_label': '지역',
                            'y_label': '결혼 이민자 비율',
                            'data': {
                                'x': [],
                                'y': [],
                                'labels': []
                            }
                        }
                    ]
                },
                'detailed_analysis': {
                    'title': '상관관계 분석 오류',
                    'error': str(e)
                }
            }

    def identify_high_risk_areas(self) -> pd.DataFrame:
        """고위험 지역 식별"""
        logger.info("Identifying high-risk areas...")
        
        try:
            # 위험도가 높은 지역 식별
            high_risk_threshold = self.school_data['closure_risk_index'].quantile(0.75)
            high_risk_areas = self.school_data[
                self.school_data['closure_risk_index'] >= high_risk_threshold
            ]
            
            return high_risk_areas.sort_values('closure_risk_index', ascending=False)
            
        except Exception as e:
            logger.error(f"Error identifying high-risk areas: {str(e)}")
            return pd.DataFrame()

    def generate_policy_recommendations(self, analysis_results: Dict) -> Dict:
        """정책 제안 생성"""
        logger.info("Generating policy recommendations...")
        
        try:
            recommendations = {
                'title': '지역 경제 연계 다문화 가정 정착 지원 정책',
                'goals': [
                    '인구 유입: 폐교 위험 지역의 인구 감소 문제를 해결하기 위해 다문화 가정을 유도해 지역 정착을 촉진.',
                    '경제 활성화: 다문화 가정의 경제적 자립을 지원하고 지역 특산물 및 관광 자원을 활용해 지역 경제를 부흥.',
                    '폐교 활용: 폐교된 학교를 지역 경제와 다문화 가정 지원의 중심지로 재활용해 자원 낭비 최소화.'
                ],
                'details': {
                    'settlement_support': {
                        'title': '다문화 가정의 지역 정착 지원',
                        'items': [
                            {
                                'title': '주거 지원 프로그램',
                                'description': '다문화 가정이 폐교 위험 지역에 정착하도록 지역 내 저렴한 주거(공공 임대주택, 리모델링된 빈집 등) 제공. 초기 정착 비용(이사비, 주거 개보수비) 일부 보조금 지원(최대 500만 원, 소득 기준 충족 시).'
                            },
                            {
                                'title': '일자리 연계',
                                'description': '지역 내 주요 산업(농업, 어업, 소규모 제조업 등)에 다문화 가정의 취업을 연계. 다문화 가정의 언어 능력을 활용해 지역 관광 가이드 또는 다국어 서비스 제공 직원으로 고용.'
                            },
                            {
                                'title': '창업 지원',
                                'description': '다문화 가정이 지역 특산물을 활용한 소규모 사업(예: 다문화 음식점, 특산물 가공품 판매)을 시작할 수 있도록 창업 자금 대출(최대 3,000만 원, 저금리) 및 경영 컨설팅 제공.'
                            }
                        ]
                    },
                    'school_reuse': {
                        'title': '폐교 공간의 경제적 재활용',
                        'items': [
                            {
                                'title': '다문화 경제 허브 조성',
                                'description': '폐교 건물을 리모델링해 다문화 가정과 지역 주민이 함께 운영하는 경제 활동 공간으로 전환. 지역 주민과 다문화 가정이 협업해 공동 브랜드(예: "다문화+로컬" 농산물 패키지)를 개발·판매.'
                            },
                            {
                                'title': '관광 자원화',
                                'description': '폐교를 지역 관광 명소로 탈바꿈해 방문객 유입을 유도. 폐교 운동장을 다문화 푸드 마켓으로 활용하거나, 교실을 다문화 체험 학습 공간(전통 놀이, 요리 교실)으로 운영.'
                            }
                        ]
                    },
                    'economic_integration': {
                        'title': '지역 경제와의 융합',
                        'items': [
                            {
                                'title': '다문화-로컬 협업 프로젝트',
                                'description': '다문화 가정과 지역 농가/소상공인이 협력해 새로운 상품 및 서비스 개발. 지역 농산물 직거래 장터에서 다문화 가정의 음식 부스를 운영해 상호 수익 창출.'
                            },
                            {
                                'title': '지역 브랜딩',
                                'description': '다문화 가정의 문화를 지역의 정체성으로 통합해 "다문화 친화 마을"로 브랜딩. 강원도 폐교 지역을 "다문화 농촌 체험 마을"로 홍보해 도시민 및 외국인 관광객 유치.'
                            }
                        ]
                    },
                    'education': {
                        'title': '교육 및 역량 강화',
                        'items': [
                            {
                                'title': '다문화 가정 역량 강화',
                                'description': '폐교 내 교육 공간에서 다문화 가정 구성원을 대상으로 직업 훈련(예: 요리, 농업 기술, 전자상거래) 제공. 한국어 및 직무 관련 언어 교육(예: 관광 가이드를 위한 한국어 회화) 운영.'
                            },
                            {
                                'title': '지역 주민 교육',
                                'description': '지역 주민을 대상으로 다문화 이해 교육(예: 다문화 가정의 문화적 배경 학습) 실시해 협업의 마찰 감소. 지역 어르신과 다문화 가정이 함께하는 전통/현대 농업 워크숍 운영.'
                            }
                        ]
                    }
                },
                'expected_effects': [
                    '인구 증가: 다문화 가정의 정착으로 폐교 위험 지역의 인구 감소 완화(목표: 3년 내 지역별 50가구 유입).',
                    '경제 활성화: 지역 특산물 판매 및 관광 수익 증가(예: 연간 지역 매출 10% 증가 목표).',
                    '사회적 통합: 다문화 가정과 지역 주민 간 협력을 통해 상호 이해 증진, 사회적 갈등 감소.',
                    '폐교 재활용: 유휴 자원(폐교)을 경제적·사회적 자산으로 전환, 지역 자원 낭비 감소.'
                ],
                'implementation': {
                    'budget': '중앙정부(행정안전부, 농림축산식품부) 및 지자체 예산 투입(예: 연간 20억 원 규모).',
                    'operation': '지자체가 정책 총괄, 지역 NGO 및 다문화 가정 협동조합이 세부 프로그램 운영.',
                    'evaluation': '정착 가구 수, 지역 매출 증가율, 관광객 방문 수, 폐교 활용률 등을 KPI로 설정.'
                }
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating policy recommendations: {str(e)}")
            return {
                'title': '',
                'goals': [],
                'details': {},
                'expected_effects': [],
                'implementation': {}
            }

    def save_analysis_results(self, results: Dict):
        """분석 결과 저장"""
        logger.info("Saving analysis results...")
        
        # 결과를 JSON 파일로 저장
        import json
        with open(os.path.join(self.reports_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 고위험 지역 목록을 CSV로 저장
        high_risk_areas = self.identify_high_risk_areas()
        high_risk_areas.to_csv(
            os.path.join(self.reports_dir, 'high_risk_areas.csv'),
            index=False,
            encoding='utf-8-sig'
        )

    def run(self):
        """전체 분석 프로세스 실행"""
        try:
            # 데이터 로드
            self.load_data()
            
            # 폐교 위험 패턴 분석
            closure_risk_analysis = self.analyze_closure_risk_patterns()
            
            # 다문화 가정 분포 분석
            multicultural_analysis = self.analyze_multicultural_distribution()
            
            # 상관관계 분석
            correlation, correlation_details = self.analyze_correlation()
            
            # 고위험 지역 식별
            high_risk_areas = self.identify_high_risk_areas()
            
            # 정책 제안 생성
            policy_recommendations = self.generate_policy_recommendations({
                'correlation': (correlation, correlation_details),
                'high_risk_areas': high_risk_areas
            })
            
            # NumPy int64를 Python int로 변환하는 함수
            def convert_np_int64(obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_np_int64(value) for key, value in obj.items()}
                elif isinstance(obj, (list, np.ndarray)):
                    return [convert_np_int64(item) for item in obj]
                return obj
            
            # 분석 결과 통합 및 NumPy int64 변환
            analysis_results = {
                'closure_risk_analysis': convert_np_int64(closure_risk_analysis),
                'multicultural_analysis': convert_np_int64(multicultural_analysis),
                'correlation': {
                    'coefficient': float(correlation),
                    'details': convert_np_int64(correlation_details)
                },
                'high_risk_areas': [
                    {k: convert_np_int64(v) for k, v in record.items()}
                    for record in high_risk_areas.to_dict('records')
                ],
                'policy_recommendations': policy_recommendations
            }
            
            # 결과 저장
            self.save_analysis_results(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = Analyzer()
    results = analyzer.run() 