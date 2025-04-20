from flask import Flask, render_template, jsonify, request, send_from_directory
from src.analysis.analyzer import Analyzer
from src.visualization.visualizer import Visualizer
from src.data_processing.data_processor import DataProcessor
import os
import pandas as pd
import glob
import plotly.express as px
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import folium
from folium import plugins
import json
import logging
import anthropic
from dotenv import load_dotenv
import time
from folium.plugins import HeatMap
import random

# Load environment variables
load_dotenv()

# Check if API key exists
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    logger.warning("ANTHROPIC_API_KEY not found in environment variables. Chat functionality will be disabled.")

# Initialize Claude client only if API key exists
claude_client = None
if api_key:
    claude_client = anthropic.Anthropic(api_key=api_key)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.debug = True  # Enable debug mode

# Initialize data processor, analyzers, and visualizer
data_processor = DataProcessor()
analyzer = Analyzer()
visualizer = Visualizer()

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure static folder exists
os.makedirs(app.static_folder, exist_ok=True)

# Korea region coordinates
KOREA_REGIONS = {
    '서울': {'lat': 37.5665, 'lon': 126.9780},
    '부산': {'lat': 35.1796, 'lon': 129.0756},
    '대구': {'lat': 35.8714, 'lon': 128.6014},
    '인천': {'lat': 37.4563, 'lon': 126.7052},
    '광주': {'lat': 35.1595, 'lon': 126.8526},
    '대전': {'lat': 36.3504, 'lon': 127.3845},
    '울산': {'lat': 35.5384, 'lon': 129.3114},
    '세종': {'lat': 36.4800, 'lon': 127.2890},
    '경기': {'lat': 37.2750, 'lon': 127.0000},
    '강원': {'lat': 37.8228, 'lon': 128.3445},
    '충북': {'lat': 36.8000, 'lon': 127.7000},
    '충남': {'lat': 36.5000, 'lon': 126.8000},
    '전북': {'lat': 35.8200, 'lon': 127.1089},
    '전남': {'lat': 34.8160, 'lon': 126.9910},
    '경북': {'lat': 36.4919, 'lon': 128.6000},
    '경남': {'lat': 35.4606, 'lon': 128.2132},
    '제주': {'lat': 33.4890, 'lon': 126.5000}
}

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/analysis')
def get_analysis():
    """분석 결과 API 엔드포인트"""
    try:
        # 분석 실행
        analysis_results = analyzer.run()
        
        # 실제 데이터 로드
        try:
            # KESS 데이터 로드
            kess_df = pd.read_csv('data/raw/kess_school_data_2023.csv')
            # 다문화 가정 데이터 로드
            multicultural_df = pd.read_csv('data/raw/multiculture.csv')
            
            # 데이터 전처리
            kess_df['학생수'] = pd.to_numeric(kess_df['학생수'], errors='coerce')
            multicultural_df['결혼이민자_비율'] = pd.to_numeric(multicultural_df['결혼이민자_비율'], errors='coerce')
            
            # 지역별 평균 학생 수와 결혼 이민자 비율 계산
            region_stats = {}
            for region in KOREA_REGIONS.keys():
                # KESS 데이터에서 지역별 학생 수 평균 계산
                region_students = kess_df[kess_df['지역'] == region]['학생수'].mean()
                # 다문화 가정 데이터에서 지역별 결혼 이민자 비율 평균 계산
                region_ratio = multicultural_df[multicultural_df['지역'] == region]['결혼이민자_비율'].mean()
                
                if not pd.isna(region_students) and not pd.isna(region_ratio):
                    region_stats[region] = {
                        'students': region_students,
                        'immigrant_ratio': region_ratio
                    }
            
            # 상관관계 분석을 위한 데이터 준비
            correlation_data = {
                'student_data': {region: stats['students'] for region, stats in region_stats.items()},
                'immigrant_ratio': {region: stats['immigrant_ratio'] for region, stats in region_stats.items()}
            }
            
        except Exception as e:
            logger.warning(f"Error loading actual data: {str(e)}")
            # 실제 데이터 로드 실패 시 예시 데이터 사용
            correlation_data = {
                'student_data': {
                    '서울': 350,
                    '부산': 250,
                    '대구': 200,
                    '인천': 300,
                    '광주': 150,
                    '대전': 180,
                    '울산': 120,
                    '세종': 100,
                    '경기': 400,
                    '강원': 100,
                    '충북': 150,
                    '충남': 200,
                    '전북': 180,
                    '전남': 150,
                    '경북': 200,
                    '경남': 250,
                    '제주': 100
                },
                'immigrant_ratio': {
                    '서울': 25,
                    '부산': 20,
                    '대구': 15,
                    '인천': 22,
                    '광주': 12,
                    '대전': 18,
                    '울산': 10,
                    '세종': 8,
                    '경기': 23,
                    '강원': 5,
                    '충북': 10,
                    '충남': 15,
                    '전북': 12,
                    '전남': 10,
                    '경북': 15,
                    '경남': 18,
                    '제주': 8
                }
            }
        
        # 상관계수 계산
        student_values = list(correlation_data['student_data'].values())
        immigrant_values = list(correlation_data['immigrant_ratio'].values())
        correlation_coefficient = np.corrcoef(student_values, immigrant_values)[0, 1]
        
        # 대시보드에 필요한 데이터 구조 생성
        dashboard_data = {
            'risk_distribution': {
                'labels': ['낮음', '중간', '높음'],
                'values': [30, 50, 20]  # 예시 데이터
            },
            'regional_stats': {
                'labels': list(correlation_data['student_data'].keys()),
                'values': student_values
            },
            'correlation': {
                'coefficient': correlation_coefficient,
                'details': {
                    'visualization_data': {
                        'charts': [
                            {
                                'type': 'scatter',
                                'title': '학생 수와 결혼 이민자 비율 간의 상관관계',
                                'x_label': '학생 수',
                                'y_label': '결혼 이민자 비율',
                                'data': {
                                    'x': student_values,
                                    'y': immigrant_values,
                                    'labels': list(correlation_data['student_data'].keys())
                                }
                            },
                            {
                                'type': 'bar',
                                'title': '지역별 학생 수',
                                'x_label': '지역',
                                'y_label': '학생 수',
                                'data': {
                                    'x': list(correlation_data['student_data'].keys()),
                                    'y': student_values,
                                    'labels': list(correlation_data['student_data'].keys())
                                }
                            },
                            {
                                'type': 'bar',
                                'title': '지역별 결혼 이민자 비율',
                                'x_label': '지역',
                                'y_label': '결혼 이민자 비율',
                                'data': {
                                    'x': list(correlation_data['immigrant_ratio'].keys()),
                                    'y': immigrant_values,
                                    'labels': list(correlation_data['immigrant_ratio'].keys())
                                }
                            }
                        ]
                    },
                    'detailed_analysis': {
                        'interpretation': [
                            f'학생 수와 결혼 이민자 비율 간의 상관계수는 {correlation_coefficient:.3f}로, {"강한 양의" if correlation_coefficient > 0.7 else "약한"} 상관관계가 있습니다.',
                            '학생 수가 증가할수록 결혼 이민자 비율도 증가하는 경향이 나타납니다.',
                            '다문화 가정(결혼 이민자)의 비율이 높은 지역일수록 학생 수가 많아 폐교 위험이 낮아질 가능성이 있습니다.',
                            '반대로 학생 수가 적은 지역은 결혼 이민자 비율도 낮아 폐교 위험이 높아질 수 있습니다.'
                        ],
                        'limitations': [
                            '데이터가 제한되어 상관관계 분석의 신뢰도가 낮을 수 있습니다.',
                            '현재 데이터는 특정 시점만 포함하므로 시계열 분석이 불가능합니다.',
                            '학생 수와 결혼 이민자 비율의 상관관계만 확인되었으며, 폐교 수와의 직접적 관계는 추론에 의존합니다.'
                        ],
                        'suggestions': [
                            '각 지역의 연도별 학생 수와 결혼 이민자 비율 데이터를 수집하여 시계열 분석을 수행하세요.',
                            '지역별 폐교 수 데이터를 포함하여 다문화 가정 → 학생 수 → 폐교 수로 이어지는 관계를 더 명확히 분석하세요.',
                            '지역 수를 늘려 상관관계 분석의 신뢰도를 높이세요.',
                            '상관계수 외에 선형 회귀 분석을 통해 결혼 이민자 비율이 학생 수에 미치는 영향을 정량적으로 평가하세요.'
                        ]
                    }
                }
            },
            'time_series': {
                'labels': ['2018', '2019', '2020', '2021', '2022', '2023'],
                'values': [10, 15, 20, 18, 22, 25]  # 예시 데이터
            }
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Error in analysis API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization')
def get_visualization():
    """시각화 결과 API"""
    try:
        # 시각화 실행
        visualizer.run()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
    try:
        # 분석 실행
        analysis_results = analyzer.run()
        
        # 대시보드에 필요한 데이터 추출
        correlation_data = analysis_results.get('correlation', {})
        if isinstance(correlation_data, tuple):
            correlation_coefficient, correlation_details = correlation_data
        else:
            correlation_coefficient, correlation_details = correlation_data.get('coefficient', 0.0), correlation_data.get('details', {})
        
        dashboard_data = {
            'correlation': {
                'coefficient': correlation_coefficient,
                'details': {
                    'visualization_data': {
                        'charts': [
                            {
                                'type': 'scatter',
                                'title': '학생 수와 결혼 이민자 비율 간의 상관관계',
                                'x_label': '학생 수',
                                'y_label': '결혼 이민자 비율',
                                'data': {
                                    'x': correlation_details.get('student_data', {}).values(),
                                    'y': correlation_details.get('immigrant_ratio', {}).values(),
                                    'labels': list(correlation_details.get('student_data', {}).keys())
                                }
                            },
                            {
                                'type': 'bar',
                                'title': '지역별 학생 수',
                                'x_label': '지역',
                                'y_label': '학생 수',
                                'data': {
                                    'x': list(correlation_details.get('student_data', {}).keys()),
                                    'y': list(correlation_details.get('student_data', {}).values()),
                                    'labels': list(correlation_details.get('student_data', {}).keys())
                                }
                            },
                            {
                                'type': 'bar',
                                'title': '지역별 결혼 이민자 비율',
                                'x_label': '지역',
                                'y_label': '결혼 이민자 비율',
                                'data': {
                                    'x': list(correlation_details.get('immigrant_ratio', {}).keys()),
                                    'y': list(correlation_details.get('immigrant_ratio', {}).values()),
                                    'labels': list(correlation_details.get('immigrant_ratio', {}).keys())
                                }
                            }
                        ]
                    },
                    'detailed_analysis': correlation_details.get('detailed_analysis', {
                        'interpretation': [],
                        'limitations': [],
                        'suggestions': []
                    })
                }
            }
        }
        
        return render_template('dashboard.html', **dashboard_data)
    except Exception as e:
        logger.error(f"Error in dashboard view: {str(e)}")
        # 오류 발생 시 기본 데이터 구조 제공
        return render_template('dashboard.html', correlation={
            'coefficient': 0.0,
            'details': {
                'visualization_data': {
                    'charts': []
                },
                'detailed_analysis': {
                    'interpretation': [],
                    'limitations': [],
                    'suggestions': []
                }
            }
        })

@app.route('/map')
def map_view():
    """지도 시각화 페이지"""
    return render_template('map.html')

@app.route('/analysis')
def analysis_view():
    """상세 분석 페이지"""
    try:
        # 분석 실행
        analysis_results = analyzer.run()
        
        # 정책 제안 생성
        policy = analyzer.generate_policy_recommendations(analysis_results)
        
        return render_template('analysis.html', policy=policy)
    except Exception as e:
        logger.error(f"Error in analysis view: {str(e)}")
        return render_template('analysis.html', policy={
            'title': '정책 제안',
            'goals': [],
            'details': {},
            'expected_effects': [],
            'implementation': {}
        })

@app.route('/kess_data')
def kess_data():
    try:
        # Read KESS data
        kess_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'kess_school_data_2023.csv')
        if os.path.exists(kess_data_path):
            df = pd.read_csv(kess_data_path)
            data = df.to_dict('records')
            return render_template('kess_data.html', data=data)
        else:
            return render_template('kess_data.html', data=[], error="KESS 데이터 파일을 찾을 수 없습니다.")
    except Exception as e:
        return render_template('kess_data.html', data=[], error=f"데이터 로딩 중 오류 발생: {str(e)}")

@app.route('/school_closure_risk')
def school_closure_risk():
    # Get selected region from query parameters
    selected_region = request.args.get('region', 'all')
    
    # Initialize data structures
    region_risk = {}
    closure_risk_data = []
    
    # Map Korean region names to CSV filenames
    region_to_file = {
        '강원': 'gangwon.csv',
        '경기': 'gyeonggi.csv',
        '경남': 'gyeongnam.csv',
        '경북': 'gyeongbook.csv',
        '광주': 'gwangju.csv',
        '대구': 'daegu.csv',
        '대전': 'daejeon.csv',
        '부산': 'busan.csv',
        '서울': 'seoul.csv',
        '세종': 'sejong.csv',
        '울산': 'ulsan.csv',
        '인천': 'incheon.csv',
        '전남': 'jeonnam.csv',
        '전북': 'jeonbook.csv',
        '제주': 'jeju.csv',
        '충남': 'chunnam.csv',
        '충북': 'chunbook.csv'
    }
    
    try:
        # If a specific region is selected, only load that region's data
        if selected_region != 'all':
            if selected_region not in region_to_file:
                return render_template('school_closure_risk.html', 
                                    error=f"Invalid region: {selected_region}",
                                    selected_region=selected_region,
                                    closure_risk_data=[])
            
            filename = region_to_file[selected_region]
            filepath = os.path.join('data', 'raw', 'location_gone', filename)
            
            if not os.path.exists(filepath):
                return render_template('school_closure_risk.html', 
                                    error=f"Data file not found for {selected_region}",
                                    selected_region=selected_region,
                                    closure_risk_data=[])
            
            # Try reading with UTF-8 first, fall back to CP949 if needed
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='cp949')
            
            # Ensure 폐교연도 is numeric and handle any conversion errors
            df['폐교연도'] = pd.to_numeric(df['폐교연도'], errors='coerce')
            df = df.dropna(subset=['폐교연도'])  # Remove rows with invalid years
            
            # Process the data
            for _, row in df.iterrows():
                school_data = {
                    'school_name': row['폐교명'],
                    'closure_year': int(row['폐교연도']),  # Convert to integer
                    'local_office': row['지역교육청'],
                    'school_level': row['급별'],
                    'utilization_status': row['활용현황'],
                    'address': row['주소']
                }
                closure_risk_data.append(school_data)
            
            # Calculate region statistics
            region_risk[selected_region] = {
                'total_schools': len(closure_risk_data),
                'avg_year': df['폐교연도'].mean(),
                'recent_closures': len(df[df['폐교연도'] >= df['폐교연도'].max() - 5]),
                'utilized_schools': len(df[df['활용현황'] != '미활용'])
            }
        else:
            # Load all regions' data
            for region, filename in region_to_file.items():
                filepath = os.path.join('data', 'raw', 'location_gone', filename)
                
                if not os.path.exists(filepath):
                    continue
                
                # Try reading with UTF-8 first, fall back to CP949 if needed
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='cp949')
                
                # Ensure 폐교연도 is numeric and handle any conversion errors
                df['폐교연도'] = pd.to_numeric(df['폐교연도'], errors='coerce')
                df = df.dropna(subset=['폐교연도'])  # Remove rows with invalid years
                
                # Process the data
                for _, row in df.iterrows():
                    school_data = {
                        'school_name': row['폐교명'],
                        'closure_year': int(row['폐교연도']),  # Convert to integer
                        'local_office': row['지역교육청'],
                        'school_level': row['급별'],
                        'utilization_status': row['활용현황'],
                        'address': row['주소']
                    }
                    closure_risk_data.append(school_data)
                
                # Calculate region statistics
                region_risk[region] = {
                    'total_schools': len(df),
                    'avg_year': df['폐교연도'].mean(),
                    'recent_closures': len(df[df['폐교연도'] >= df['폐교연도'].max() - 5]),
                    'utilized_schools': len(df[df['활용현황'] != '미활용'])
                }
        
        return render_template('school_closure_risk.html',
                            selected_region=selected_region,
                            region_risk=region_risk,
                            closure_risk_data=closure_risk_data)
    
    except Exception as e:
        return render_template('school_closure_risk.html',
                            error=f"Error processing data: {str(e)}",
                            selected_region=selected_region,
                            closure_risk_data=[])

@app.route('/api/kess_data/<int:year>')
def get_kess_data(year):
    try:
        df = pd.read_csv(f'data/raw/kess_school_data_{year}.csv')
        # Calculate students per class
        df['students_per_class'] = df['student_count'] / df['class_count']
        
        return jsonify({
            'schools': df.to_dict('records'),
            'students_per_class': df['students_per_class'].tolist(),
            'student_count': df['student_count'].tolist()
        })
    except FileNotFoundError:
        return jsonify({'error': 'Data not found for the specified year'}), 404
    except Exception as e:
        logger.error(f"Error in get_kess_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/closure_risk_map')
def closure_risk_map():
    try:
        # 지역별 데이터 초기화
        region_data = {}
        heatmap_data = []
        
        # 지역명 매핑
        region_mapping = {
            '서울': 'seoul.csv',
            '부산': 'busan.csv',
            '대구': 'daegu.csv',
            '인천': 'incheon.csv',
            '광주': 'gwangju.csv',
            '대전': 'daejeon.csv',
            '울산': 'ulsan.csv',
            '세종': 'sejong.csv',
            '경기': 'gyeonggi.csv',
            '강원': 'gangwon.csv',
            '충북': 'chunbook.csv',
            '충남': 'chunnam.csv',
            '전북': 'jeonbook.csv',
            '전남': 'jeonnam.csv',
            '경북': 'gyeongbook.csv',
            '경남': 'gyeongnam.csv',
            '제주': 'jeju.csv'
        }
        
        # 지역별 좌표
        region_coords = {
            '서울': [37.5665, 126.9780],
            '부산': [35.1796, 129.0756],
            '대구': [35.8714, 128.6014],
            '인천': [37.4563, 126.7052],
            '광주': [35.1595, 126.8526],
            '대전': [36.3504, 127.3845],
            '울산': [35.5384, 129.3114],
            '세종': [36.4801, 127.2892],
            '경기': [37.4138, 127.5183],
            '강원': [37.8228, 128.3445],
            '충북': [36.6372, 127.4897],
            '충남': [36.5184, 126.8000],
            '전북': [35.8242, 127.1480],
            '전남': [34.8679, 126.9910],
            '경북': [36.4919, 128.8889],
            '경남': [35.4606, 128.2132],
            '제주': [33.4996, 126.5312]
        }
        
        max_risk_score = 0
        
        # 각 지역별 데이터 처리
        for region_kr, filename in region_mapping.items():
            try:
                # CSV 파일 읽기
                file_path = os.path.join('data', 'raw', 'location_gone', filename)
                if not os.path.exists(file_path):
                    logging.warning(f"File not found: {file_path}")
                    continue
                    
                # Try reading with UTF-8 first, fall back to CP949 if needed
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')
                
                # 폐교연도 처리
                def extract_year(x):
                    try:
                        # 문자열로 변환
                        x = str(x)
                        # 숫자만 추출
                        import re
                        numbers = re.findall(r'\d+', x)
                        if numbers:
                            # 첫 번째 숫자 그룹을 사용
                            return float(numbers[0])
                        return None
                    except:
                        return None

                if '폐교연도' in df.columns:
                    df['폐교연도'] = df['폐교연도'].apply(extract_year)
                    df = df.dropna(subset=['폐교연도'])
                else:
                    logging.warning(f"'폐교연도' column not found in {file_path}")
                    continue
                
                if df.empty:
                    logging.warning(f"No valid data in {file_path}")
                    continue
                
                # 지역별 통계 계산
                total_closures = len(df)
                recent_closures = len(df[df['폐교연도'] >= 2020])
                unused_schools = len(df[df['활용현황'] == '미활용'])
                
                # 지역 데이터 저장
                region_data[region_kr] = {
                    'total_closures': total_closures,
                    'recent_closures': recent_closures,
                    'unused_schools': unused_schools
                }
                
                # 위험도 점수 계산 (최근 폐교와 미활용 비율을 고려)
                risk_score = (recent_closures * 0.6 + unused_schools * 0.4) / total_closures if total_closures > 0 else 0
                max_risk_score = max(max_risk_score, risk_score)
                
                # 히트맵 데이터 추가
                if region_kr in region_coords:
                    coords = region_coords[region_kr]
                    base_lat, base_lng = coords
                    
                    # 지역 크기에 따라 포인트 수 조정
                    region_sizes = {
                        '경기': 1.2, '강원': 1.2, '경북': 1.2, '경남': 1.1,
                        '전남': 1.1, '전북': 1.0, '충남': 1.0, '충북': 1.0,
                        '제주': 0.8, '서울': 0.8, '부산': 0.8, '대구': 0.8,
                        '인천': 0.8, '광주': 0.8, '대전': 0.8, '울산': 0.8, '세종': 0.8
                    }
                    
                    size_multiplier = region_sizes.get(region_kr, 1.0)
                    num_points = int((risk_score * 50 + 10) * size_multiplier)
                    
                    # 지역 크기에 따라 랜덤 범위 조정
                    lat_range = 0.5 * size_multiplier
                    lng_range = 0.5 * size_multiplier
                    
                    for _ in range(num_points):
                        # 좌표에 랜덤성 추가 (정규분포 사용)
                        lat = base_lat + np.random.normal(0, lat_range/3)
                        lng = base_lng + np.random.normal(0, lng_range/3)
                        intensity = risk_score * 100  # 강도를 100배로 스케일업
                        heatmap_data.append([lat, lng, intensity])
                
            except Exception as e:
                logging.error(f"Error processing {region_kr} data: {str(e)}")
                continue
        
        # Folium 지도 생성
        m = folium.Map(location=[36.5, 127.5], zoom_start=7,
                      tiles='CartoDB positron')
        
        # 히트맵 데이터를 단순 좌표로 변환
        simple_points = []
        for region_kr, data in region_data.items():
            if region_kr in region_coords:
                coords = region_coords[region_kr]
                base_lat, base_lng = coords
                
                # 위험도 점수 계산
                risk_score = (data['recent_closures'] * 0.6 + data['unused_schools'] * 0.4) / data['total_closures'] if data['total_closures'] > 0 else 0
                
                # 위험도에 따라 포인트 수 조정
                num_points = int(risk_score * 50) + 5  # 최소 5개, 최대 55개 포인트
                
                # 지역 크기에 따른 분포 범위
                region_sizes = {
                    '경기': 0.8, '강원': 0.8, '경북': 0.8, '경남': 0.7,
                    '전남': 0.7, '전북': 0.6, '충남': 0.6, '충북': 0.6,
                    '제주': 0.4, '서울': 0.3, '부산': 0.3, '대구': 0.3,
                    '인천': 0.3, '광주': 0.3, '대전': 0.3, '울산': 0.3, '세종': 0.3
                }
                
                spread = region_sizes.get(region_kr, 0.5)
                
                # 포인트 생성
                for _ in range(num_points):
                    lat = base_lat + (random.random() - 0.5) * spread
                    lng = base_lng + (random.random() - 0.5) * spread
                    simple_points.append([lat, lng])
        
        # 히트맵 추가
        if simple_points:
            gradient = {
                '0.0': '#3288bd',  # 파란색 (낮은 위험)
                '0.3': '#99d594',  # 연두색
                '0.5': '#fee08b',  # 노란색
                '0.7': '#fc8d59',  # 주황색
                '1.0': '#d53e4f'   # 빨간색 (높은 위험)
            }
            
            HeatMap(
                data=simple_points,
                radius=15,
                blur=10,
                min_opacity=0.3,
                gradient=gradient,
                use_local_extrema=True
            ).add_to(m)
        
        # 지역별 마커 추가
        for region, data in region_data.items():
            if region in region_coords:
                coords = region_coords[region]
                # 위험도에 따른 색상 설정
                risk_score = (data['recent_closures'] * 0.6 + data['unused_schools'] * 0.4) / data['total_closures'] if data['total_closures'] > 0 else 0
                color = '#d53e4f' if risk_score > 0.5 else '#3288bd'  # 위험도가 높으면 빨간색, 낮으면 파란색
                
                # 마커 생성
                folium.CircleMarker(
                    location=coords,
                    radius=15,
                    popup=f"""
                    <div style="font-family: 'Malgun Gothic', sans-serif;">
                        <h4>{region}</h4>
                        <p>총 폐교: {data['total_closures']}개</p>
                        <p>최근 폐교: {data['recent_closures']}개</p>
                        <p>미활용: {data['unused_schools']}개</p>
                    </div>
                    """,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=2
                ).add_to(m)
        
        # 지도를 HTML로 변환
        map_html = m._repr_html_()
        
        return render_template('closure_risk_map.html', 
                            map_html=map_html,
                            region_data=region_data)
        
    except Exception as e:
        logging.error(f"Error in closure_risk_map: {str(e)}")
        return render_template('closure_risk_map.html', 
                            error="지도 데이터를 불러오는 중 오류가 발생했습니다.")

@app.route('/multiculture')
def multiculture_view():
    try:
        # CSV 파일 경로
        filepath = os.path.join('data', 'raw', 'multiculture.csv')
        
        # 파일이 존재하는지 확인
        if not os.path.exists(filepath):
            return render_template('multiculture.html', 
                                error="데이터 파일이 존재하지 않습니다.",
                                table_data=[],
                                map_data=[])
        
        # 여러 인코딩으로 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            return render_template('multiculture.html', 
                                error="데이터 파일을 읽을 수 없습니다. 인코딩 문제가 발생했습니다.",
                                table_data=[],
                                map_data=[])
        
        # 테이블 데이터 준비
        table_data = df.to_dict('records')
        
        # 지도 데이터 준비
        map_data = []
        for _, row in df.iterrows():
            try:
                region = row['행정구역(읍면동)별(1)']
                if region == '전국':
                    continue
                    
                # 지역별 좌표 매핑
                region_coords = {
                    '서울특별시': {'lat': 37.5665, 'lng': 126.9780},
                    '부산광역시': {'lat': 35.1796, 'lng': 129.0756},
                    '대구광역시': {'lat': 35.8714, 'lng': 128.6014},
                    '인천광역시': {'lat': 37.4563, 'lng': 126.7052},
                    '광주광역시': {'lat': 35.1595, 'lng': 126.8526},
                    '대전광역시': {'lat': 36.3504, 'lng': 127.3845},
                    '울산광역시': {'lat': 35.5384, 'lng': 129.3114},
                    '세종특별자치시': {'lat': 36.4877, 'lng': 127.2817},
                    '경기도': {'lat': 37.4138, 'lng': 127.5183},
                    '강원도': {'lat': 37.8228, 'lng': 128.1555},
                    '충청북도': {'lat': 36.6372, 'lng': 127.4890},
                    '충청남도': {'lat': 36.5184, 'lng': 126.8000},
                    '전라북도': {'lat': 35.7175, 'lng': 127.1530},
                    '전라남도': {'lat': 34.8679, 'lng': 126.9910},
                    '경상북도': {'lat': 36.4919, 'lng': 128.8889},
                    '경상남도': {'lat': 35.4606, 'lng': 128.2132},
                    '제주특별자치도': {'lat': 33.4996, 'lng': 126.5312}
                }
                
                if region in region_coords:
                    coords = region_coords[region]
                    # 다문화 가정 비율 계산 (결혼이민자 및 귀화자 등 / 전체 인구)
                    total_population = float(row['결혼이민자 및 귀화자 등'])
                    multicultural_ratio = total_population / 1000000  # 백만명당 비율로 계산
                    
                    map_data.append({
                        '지역': region,
                        'lat': coords['lat'],
                        'lng': coords['lng'],
                        '값': multicultural_ratio
                    })
            except (ValueError, TypeError, KeyError) as e:
                print(f"Error processing row: {e}")
                continue
        
        return render_template('multiculture.html', 
                            table_data=table_data,
                            map_data=map_data)
    
    except Exception as e:
        print(f"Error in multiculture_view: {str(e)}")
        return render_template('multiculture.html', 
                            error=f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}",
                            table_data=[],
                            map_data=[])

@app.route('/static/data/<path:filename>')
def serve_static(filename):
    return send_from_directory('static/data', filename)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Check if API key and client are available
        if not claude_client:
            logger.error("Chat functionality is disabled: ANTHROPIC_API_KEY not configured")
            return jsonify({'error': 'Chat functionality is currently unavailable'}), 503
            
        # Create message
        message = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )
        
        # Extract response text
        response_text = message.content[0].text if message.content else "No response generated"
        
        return jsonify({'response': response_text})
        
    except anthropic.APIError as e:
        logger.error(f"Claude API error: {str(e)}")
        return jsonify({'error': 'Error communicating with Claude API'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in chat route: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/time_series/<region>')
def get_time_series(region):
    """지역별 시계열 분석 API 엔드포인트"""
    try:
        # 지역명 매핑
        region_mapping = {
            '강원': 'gangwon.csv',
            '경기': 'gyeonggi.csv',
            '경남': 'gyeongnam.csv',
            '경북': 'gyeongbook.csv',
            '광주': 'gwangju.csv',
            '대구': 'daegu.csv',
            '대전': 'daejeon.csv',
            '부산': 'busan.csv',
            '서울': 'seoul.csv',
            '세종': 'sejong.csv',
            '울산': 'ulsan.csv',
            '인천': 'incheon.csv',
            '전남': 'jeonnam.csv',
            '전북': 'jeonbook.csv',
            '제주': 'jeju.csv',
            '충남': 'chunnam.csv',
            '충북': 'chunbook.csv'
        }
        
        if region not in region_mapping:
            return jsonify({'error': f'Invalid region: {region}'}), 400
            
        # 폐교 데이터 읽기
        file_path = os.path.join('data', 'raw', 'location_gone', region_mapping[region])
        if not os.path.exists(file_path):
            return jsonify({'error': f'Data file not found for {region}'}), 404
            
        # Try reading with UTF-8 first, fall back to CP949 if needed
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
            
        # 폐교연도 처리
        df['폐교연도'] = pd.to_numeric(df['폐교연도'], errors='coerce')
        df = df.dropna(subset=['폐교연도'])
        
        # 연도별 폐교 수 계산
        yearly_closures = df.groupby('폐교연도').size().reset_index(name='count')
        yearly_closures = yearly_closures.sort_values('폐교연도')
        
        # 다문화 가정 데이터 읽기
        multicultural_path = os.path.join('data', 'raw', 'multiculture.csv')
        if os.path.exists(multicultural_path):
            try:
                mc_df = pd.read_csv(multicultural_path, encoding='utf-8')
            except UnicodeDecodeError:
                mc_df = pd.read_csv(multicultural_path, encoding='cp949')
                
            # 지역별 다문화 가정 데이터 추출
            region_mc_data = mc_df[mc_df['행정구역(읍면동)별(1)'].str.contains(region)]
            if not region_mc_data.empty:
                multicultural_count = region_mc_data['결혼이민자 및 귀화자 등'].sum()
            else:
                multicultural_count = 0
        else:
            multicultural_count = 0
            
        # 시계열 분석 결과 생성
        time_series_data = {
            'labels': yearly_closures['폐교연도'].astype(int).tolist(),
            'values': yearly_closures['count'].tolist(),
            'multicultural_count': int(multicultural_count),
            'analysis': {
                'trends': {
                    'early_period': {
                        'years': [1982, 1990],
                        'avg_closures': yearly_closures[
                            (yearly_closures['폐교연도'] >= 1982) & 
                            (yearly_closures['폐교연도'] <= 1990)
                        ]['count'].mean()
                    },
                    'peak_period': {
                        'years': [1991, 1999],
                        'avg_closures': yearly_closures[
                            (yearly_closures['폐교연도'] >= 1991) & 
                            (yearly_closures['폐교연도'] <= 1999)
                        ]['count'].mean()
                    },
                    'decline_period': {
                        'years': [2000, 2010],
                        'avg_closures': yearly_closures[
                            (yearly_closures['폐교연도'] >= 2000) & 
                            (yearly_closures['폐교연도'] <= 2010)
                        ]['count'].mean()
                    },
                    'stable_period': {
                        'years': [2011, 2024],
                        'avg_closures': yearly_closures[
                            (yearly_closures['폐교연도'] >= 2011) & 
                            (yearly_closures['폐교연도'] <= 2024)
                        ]['count'].mean()
                    }
                },
                'correlation': {
                    'multicultural_impact': '다문화 가정 증가가 폐교 수 감소에 기여했을 가능성 있음',
                    'policy_impact': '지역 정책(학교 통폐합 완화, 지역 활성화)이 폐교 감소에 영향을 미쳤을 가능성'
                }
            }
        }
        
        return jsonify(time_series_data)
        
    except Exception as e:
        logger.error(f"Error in time series analysis for {region}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 