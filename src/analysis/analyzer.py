import pandas as pd
import numpy as np
from scipy import stats
import geopandas as gpd
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
            if self.multicultural_data is None:
                return 0.0, {
                    'correlation_pvalue': 1.0,
                    'r2_score': 0.0,
                    'coefficient': 0.0,
                    'intercept': 0.0
                }
            
            # 지역별 평균 위험 지수와 다문화 비율 계산
            risk_by_region = self.school_data.groupby('region')['closure_risk_index'].mean()
            multicultural_by_region = self.multicultural_data.groupby('region')['ratio'].mean()
            
            # 두 데이터프레임을 병합
            correlation_data = pd.merge(
                risk_by_region.reset_index(),
                multicultural_by_region.reset_index(),
                on='region',
                how='inner'
            )
            
            # 피어슨 상관계수
            correlation = stats.pearsonr(
                correlation_data['closure_risk_index'],
                correlation_data['ratio']
            )
            
            # 선형 회귀 분석
            X = correlation_data[['ratio']]
            y = correlation_data['closure_risk_index']
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            
            return correlation[0], {
                'correlation_pvalue': correlation[1],
                'r2_score': r2,
                'coefficient': model.coef_[0],
                'intercept': model.intercept_
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return 0.0, {
                'correlation_pvalue': 1.0,
                'r2_score': 0.0,
                'coefficient': 0.0,
                'intercept': 0.0
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
                'priority_areas': [],
                'support_measures': [],
                'monitoring_suggestions': []
            }
            
            # 고위험 지역 식별
            high_risk_areas = self.identify_high_risk_areas()
            if not high_risk_areas.empty:
                recommendations['priority_areas'] = high_risk_areas[
                    ['region', 'school_name', 'closure_risk_index']
                ].head(10).to_dict('records')
            
            # 지원 방안 제안
            recommendations['support_measures'].extend([
                "학급 수 및 학생 수 감소율이 높은 지역에 대한 추가 지원",
                "폐교 위험이 높은 학교에 대한 특별 관리",
                "지역별 교육 인프라 현황 정기 점검"
            ])
            
            # 모니터링 제안
            recommendations['monitoring_suggestions'].extend([
                "학급 수 및 학생 수 감소율의 지속적인 모니터링",
                "지역별 교육 인프라 현황 정기 점검",
                "폐교 위험 지수의 변화 추적"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating policy recommendations: {str(e)}")
            return {
                'priority_areas': [],
                'support_measures': [],
                'monitoring_suggestions': []
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
            
            # 분석 결과 통합
            analysis_results = {
                'closure_risk_analysis': closure_risk_analysis,
                'multicultural_analysis': multicultural_analysis,
                'correlation': {
                    'coefficient': correlation,
                    'details': correlation_details
                },
                'high_risk_areas': high_risk_areas.to_dict('records'),
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