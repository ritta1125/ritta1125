import pandas as pd
import numpy as np
from scipy import stats
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import logging
from typing import Dict, Tuple

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

    def load_data(self):
        """처리된 데이터 로드"""
        logger.info("Loading processed data...")
        self.school_data = pd.read_csv(
            os.path.join(self.processed_data_dir, 'processed_school_data.csv')
        )
        self.geo_data = gpd.read_file(
            os.path.join(self.processed_data_dir, 'processed_geospatial_data.geojson')
        )

    def analyze_closure_risk_patterns(self) -> Dict:
        """폐교 위험 패턴 분석"""
        logger.info("Analyzing school closure risk patterns...")
        
        # 지역별 폐교위험도 통계
        regional_stats = self.school_data.groupby('region').agg({
            'closure_risk_index': ['mean', 'std', 'count'],
            'class_decrease_rate': 'mean',
            'student_decrease_rate': 'mean'
        }).round(3)
        
        # 도시/농촌별 폐교위험도 비교
        urban_rural_stats = self.school_data.groupby('urban_rural').agg({
            'closure_risk_index': ['mean', 'std'],
            'multicultural_ratio': 'mean'
        }).round(3)
        
        return {
            'regional_stats': regional_stats,
            'urban_rural_stats': urban_rural_stats
        }

    def analyze_multicultural_distribution(self) -> Dict:
        """다문화 가정 분포 분석"""
        logger.info("Analyzing multicultural family distribution...")
        
        # 지역별 다문화 비율 통계
        multicultural_stats = self.school_data.groupby('region').agg({
            'multicultural_ratio': ['mean', 'std', 'count'],
            'multicultural_students': 'sum'
        }).round(3)
        
        # 도시/농촌별 다문화 비율 비교
        urban_rural_multicultural = self.school_data.groupby('urban_rural').agg({
            'multicultural_ratio': ['mean', 'std'],
            'multicultural_students': 'sum'
        }).round(3)
        
        return {
            'multicultural_stats': multicultural_stats,
            'urban_rural_multicultural': urban_rural_multicultural
        }

    def analyze_correlation(self) -> Tuple[float, Dict]:
        """폐교위험도와 다문화비중 간의 상관관계 분석"""
        logger.info("Analyzing correlation between closure risk and multicultural ratio...")
        
        # 피어슨 상관계수
        correlation = stats.pearsonr(
            self.school_data['closure_risk_index'],
            self.school_data['multicultural_index']
        )
        
        # 선형 회귀 분석
        X = self.school_data[['multicultural_index']]
        y = self.school_data['closure_risk_index']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        return correlation[0], {
            'correlation_pvalue': correlation[1],
            'r2_score': r2,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_
        }

    def identify_high_risk_areas(self) -> pd.DataFrame:
        """고위험 지역 식별"""
        logger.info("Identifying high-risk areas...")
        
        # 위험도와 다문화비중이 모두 높은 지역 식별
        high_risk_threshold = self.school_data['closure_risk_index'].quantile(0.75)
        high_multicultural_threshold = self.school_data['multicultural_ratio'].quantile(0.75)
        
        high_risk_areas = self.school_data[
            (self.school_data['closure_risk_index'] >= high_risk_threshold) &
            (self.school_data['multicultural_ratio'] >= high_multicultural_threshold)
        ]
        
        return high_risk_areas.sort_values('closure_risk_index', ascending=False)

    def generate_policy_recommendations(self, analysis_results: Dict) -> Dict:
        """정책 제안 생성"""
        logger.info("Generating policy recommendations...")
        
        recommendations = {
            'priority_areas': [],
            'support_measures': [],
            'monitoring_suggestions': []
        }
        
        # 고위험 지역 식별
        high_risk_areas = self.identify_high_risk_areas()
        recommendations['priority_areas'] = high_risk_areas[
            ['region', 'school_name', 'closure_risk_index', 'multicultural_ratio']
        ].head(10).to_dict('records')
        
        # 지원 방안 제안
        if analysis_results['correlation'][0] > 0.3:
            recommendations['support_measures'].append(
                "다문화 학생 지원 프로그램 강화 (폐교위험도와 다문화비중이 양의 상관관계를 보임)"
            )
        
        # 모니터링 제안
        recommendations['monitoring_suggestions'].extend([
            "학급 수 및 학생 수 감소율의 지속적인 모니터링",
            "다문화 학생 비율 변화 추적",
            "지역별 교육 인프라 현황 정기 점검"
        ])
        
        return recommendations

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
            self.load_data()
            
            # 각종 분석 수행
            closure_risk_analysis = self.analyze_closure_risk_patterns()
            multicultural_analysis = self.analyze_multicultural_distribution()
            correlation_analysis = self.analyze_correlation()
            
            # 분석 결과 통합
            analysis_results = {
                'closure_risk_analysis': closure_risk_analysis,
                'multicultural_analysis': multicultural_analysis,
                'correlation': correlation_analysis,
                'policy_recommendations': self.generate_policy_recommendations({
                    'correlation': correlation_analysis
                })
            }
            
            # 결과 저장
            self.save_analysis_results(analysis_results)
            logger.info("Analysis completed successfully")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = Analyzer()
    results = analyzer.run() 