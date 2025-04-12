import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import geopandas as gpd
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.processed_data_dir = os.path.join(self.base_dir, 'data', 'processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_data(self):
        """원본 데이터 로드"""
        logger.info("Loading raw data...")
        # TODO: 실제 데이터 파일명으로 수정 필요
        self.school_data = pd.read_csv(os.path.join(self.raw_data_dir, 'school_data.csv'))
        self.facilities_data = pd.read_csv(os.path.join(self.raw_data_dir, 'facilities_data.csv'))
        self.closed_schools = pd.read_csv(os.path.join(self.raw_data_dir, 'closed_schools.csv'))
        self.multicultural_data = pd.read_csv(os.path.join(self.raw_data_dir, 'multicultural_data.csv'))
        self.geo_data = gpd.read_file(os.path.join(self.raw_data_dir, 'geospatial_data.geojson'))

    def calculate_school_closure_risk_index(self):
        """폐교위험지수 계산"""
        logger.info("Calculating school closure risk index...")
        # 가중치 설정
        weights = {
            'class_decrease_rate': 0.3,
            'student_decrease_rate': 0.3,
            'nearby_closed_schools': 0.2,
            'facility_condition': 0.2
        }
        
        # 각 지표 계산
        self.school_data['class_decrease_rate'] = (
            (self.school_data['current_classes'] - self.school_data['previous_classes']) 
            / self.school_data['previous_classes']
        )
        
        self.school_data['student_decrease_rate'] = (
            (self.school_data['current_students'] - self.school_data['previous_students']) 
            / self.school_data['previous_students']
        )
        
        # 표준화
        scaler = StandardScaler()
        features = ['class_decrease_rate', 'student_decrease_rate', 'nearby_closed_schools', 'facility_condition']
        self.school_data[features] = scaler.fit_transform(self.school_data[features])
        
        # 폐교위험지수 계산
        self.school_data['closure_risk_index'] = (
            self.school_data['class_decrease_rate'] * weights['class_decrease_rate'] +
            self.school_data['student_decrease_rate'] * weights['student_decrease_rate'] +
            self.school_data['nearby_closed_schools'] * weights['nearby_closed_schools'] +
            self.school_data['facility_condition'] * weights['facility_condition']
        )

    def calculate_multicultural_index(self):
        """다문화비중지수 계산"""
        logger.info("Calculating multicultural index...")
        self.school_data['multicultural_ratio'] = (
            self.school_data['multicultural_students'] / self.school_data['total_students']
        )
        
        # 표준화
        scaler = StandardScaler()
        self.school_data['multicultural_index'] = scaler.fit_transform(
            self.school_data[['multicultural_ratio']]
        )

    def perform_spatial_clustering(self):
        """공간 클러스터링 수행"""
        logger.info("Performing spatial clustering...")
        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=5, random_state=42)
        self.school_data['risk_cluster'] = kmeans.fit_predict(
            self.school_data[['closure_risk_index', 'multicultural_index']]
        )
        
        # DBSCAN 클러스터링 (이상치 탐지)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.school_data['anomaly_cluster'] = dbscan.fit_predict(
            self.school_data[['closure_risk_index', 'multicultural_index']]
        )

    def save_processed_data(self):
        """처리된 데이터 저장"""
        logger.info("Saving processed data...")
        self.school_data.to_csv(
            os.path.join(self.processed_data_dir, 'processed_school_data.csv'),
            index=False,
            encoding='utf-8-sig'
        )
        
        # GeoJSON으로 저장
        gdf = gpd.GeoDataFrame(
            self.school_data,
            geometry=self.geo_data.geometry
        )
        gdf.to_file(
            os.path.join(self.processed_data_dir, 'processed_geospatial_data.geojson'),
            driver='GeoJSON'
        )

    def run(self):
        """전체 데이터 처리 프로세스 실행"""
        try:
            self.load_data()
            self.calculate_school_closure_risk_index()
            self.calculate_multicultural_index()
            self.perform_spatial_clustering()
            self.save_processed_data()
            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run() 