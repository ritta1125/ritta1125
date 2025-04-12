import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
import os
import logging

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
        
        # 스타일 설정
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def load_data(self):
        """처리된 데이터 로드"""
        logger.info("Loading processed data...")
        self.school_data = pd.read_csv(
            os.path.join(self.processed_data_dir, 'processed_school_data.csv')
        )
        self.geo_data = gpd.read_file(
            os.path.join(self.processed_data_dir, 'processed_geospatial_data.geojson')
        )

    def create_risk_distribution_map(self):
        """폐교위험도 분포 지도 생성"""
        logger.info("Creating risk distribution map...")
        m = folium.Map(location=[36.5, 127.5], zoom_start=7)
        
        # 위험도에 따른 색상 설정
        colors = {
            0: 'green',
            1: 'lightgreen',
            2: 'yellow',
            3: 'orange',
            4: 'red'
        }
        
        # 마커 클러스터 생성
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in self.geo_data.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=f"학교: {row['school_name']}<br>위험도: {row['closure_risk_index']:.2f}",
                color=colors[row['risk_cluster']],
                fill=True,
                fill_color=colors[row['risk_cluster']]
            ).add_to(marker_cluster)
        
        # 지도 저장
        m.save(os.path.join(self.reports_dir, 'risk_distribution_map.html'))

    def create_multicultural_distribution_map(self):
        """다문화 가정 분포 지도 생성"""
        logger.info("Creating multicultural distribution map...")
        m = folium.Map(location=[36.5, 127.5], zoom_start=7)
        
        # 다문화 비율에 따른 색상 설정
        colors = {
            0: 'blue',
            1: 'lightblue',
            2: 'purple',
            3: 'pink',
            4: 'red'
        }
        
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in self.geo_data.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=f"학교: {row['school_name']}<br>다문화 비율: {row['multicultural_ratio']:.2%}",
                color=colors[row['risk_cluster']],
                fill=True,
                fill_color=colors[row['risk_cluster']]
            ).add_to(marker_cluster)
        
        m.save(os.path.join(self.reports_dir, 'multicultural_distribution_map.html'))

    def create_correlation_plot(self):
        """상관관계 시각화"""
        logger.info("Creating correlation plots...")
        plt.figure(figsize=(12, 6))
        
        # 산점도
        plt.subplot(1, 2, 1)
        sns.scatterplot(
            data=self.school_data,
            x='closure_risk_index',
            y='multicultural_index',
            hue='risk_cluster',
            palette='husl'
        )
        plt.title('폐교위험도와 다문화비중의 관계')
        
        # 상관관계 히트맵
        plt.subplot(1, 2, 2)
        correlation = self.school_data[['closure_risk_index', 'multicultural_index', 'class_decrease_rate', 'student_decrease_rate']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('변수 간 상관관계')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, 'correlation_analysis.png'))

    def create_time_series_plot(self):
        """시계열 분석 시각화"""
        logger.info("Creating time series plots...")
        fig = go.Figure()
        
        # 폐교 수 추이
        fig.add_trace(go.Scatter(
            x=self.school_data['year'],
            y=self.school_data['closed_schools'],
            mode='lines+markers',
            name='폐교 수'
        ))
        
        # 다문화 학생 수 추이
        fig.add_trace(go.Scatter(
            x=self.school_data['year'],
            y=self.school_data['multicultural_students'],
            mode='lines+markers',
            name='다문화 학생 수'
        ))
        
        fig.update_layout(
            title='연도별 폐교 수와 다문화 학생 수 추이',
            xaxis_title='연도',
            yaxis_title='수',
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(self.reports_dir, 'time_series_analysis.html'))

    def create_regional_comparison_plot(self):
        """지역별 비교 분석 시각화"""
        logger.info("Creating regional comparison plots...")
        plt.figure(figsize=(15, 8))
        
        # 지역별 폐교위험도와 다문화비중 비교
        sns.boxplot(
            data=self.school_data,
            x='region',
            y='closure_risk_index',
            hue='urban_rural',
            palette='husl'
        )
        plt.title('지역별 폐교위험도 분포')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, 'regional_comparison.png'))

    def run(self):
        """전체 시각화 프로세스 실행"""
        try:
            self.load_data()
            self.create_risk_distribution_map()
            self.create_multicultural_distribution_map()
            self.create_correlation_plot()
            self.create_time_series_plot()
            self.create_regional_comparison_plot()
            logger.info("Visualization completed successfully")
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            raise

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.run() 