import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import logging
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_data_dir = os.path.join(self.base_dir, 'data', 'processed')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        self.school_data = None
        self.multicultural_data = None

    def load_data(self):
        """데이터 로드"""
        logger.info("Loading processed data...")
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
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_risk_map(self):
        """위험도 분포 지도 생성"""
        try:
            # 지역별 평균 위험 지수 계산
            regional_risk = self.school_data.groupby('region')['closure_risk_index'].mean().reset_index()
            
            # 위험도 등급 계산
            regional_risk['risk_level'] = pd.qcut(
                regional_risk['closure_risk_index'],
                q=5,
                labels=['매우 낮음', '낮음', '보통', '높음', '매우 높음']
            )
            
            # 결과 저장
            regional_risk.to_csv(
                os.path.join(self.reports_dir, 'regional_risk.csv'),
                index=False,
                encoding='utf-8-sig'
            )
            
        except Exception as e:
            logger.error(f"Error creating risk map: {str(e)}")
            raise

    def create_visualization_data(self):
        """시각화 데이터 생성"""
        try:
            # 지역별 통계
            regional_stats = self.school_data.groupby('region').agg({
                'closure_risk_index': ['mean', 'count']
            }).reset_index()
            
            # 시계열 데이터
            time_series = self.school_data.groupby('closure_year').size().reset_index()
            time_series.columns = ['year', 'count']
            
            # 다문화 데이터 처리
            multicultural_stats = {}
            if self.multicultural_data is not None:
                multicultural_stats = self.multicultural_data.groupby('region')['ratio'].mean().to_dict()
            
            # 결과 저장
            visualization_data = {
                'regional_stats': {
                    'labels': regional_stats['region'].tolist(),
                    'values': regional_stats[('closure_risk_index', 'mean')].round(3).tolist()
                },
                'time_series': {
                    'labels': time_series['year'].tolist(),
                    'values': time_series['count'].tolist()
                },
                'multicultural_stats': multicultural_stats
            }
            
            with open(os.path.join(self.reports_dir, 'visualization_data.json'), 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating visualization data: {str(e)}")
            raise

    def run(self):
        """전체 시각화 프로세스 실행"""
        try:
            self.load_data()
            self.create_risk_map()
            self.create_visualization_data()
            
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            raise

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.run() 