from flask import Flask, render_template, jsonify, request, send_from_directory
from src.analysis.analyzer import Analyzer
from src.visualization.visualizer import Visualizer
import os
import pandas as pd
import glob
import plotly.express as px
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import folium
from folium import plugins

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.debug = True  # Enable debug mode

# Initialize analyzers
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
    """분석 결과 API"""
    try:
        results = analyzer.run()
        return jsonify(results)
    except Exception as e:
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
    return render_template('dashboard.html')

@app.route('/map')
def map_view():
    """지도 시각화 페이지"""
    return render_template('map.html')

@app.route('/analysis')
def analysis_view():
    """상세 분석 페이지"""
    return render_template('analysis.html')

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
        '경북': 'gyeongbuk.csv',
        '광주': 'gwangju.csv',
        '대구': 'daegu.csv',
        '대전': 'daejeon.csv',
        '부산': 'busan.csv',
        '서울': 'seoul.csv',
        '세종': 'sejong.csv',
        '울산': 'ulsan.csv',
        '인천': 'incheon.csv',
        '전남': 'jeonnam.csv',
        '전북': 'jeonbuk.csv',
        '제주': 'jeju.csv',
        '충남': 'chungnam.csv',
        '충북': 'chungbuk.csv'
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
            
            # Process the data
            for _, row in df.iterrows():
                school_data = {
                    'school_name': row['폐교명'],
                    'closure_year': row['폐교연도'],
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
                
                # Process the data
                for _, row in df.iterrows():
                    school_data = {
                        'school_name': row['폐교명'],
                        'closure_year': row['폐교연도'],
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

@app.route('/closure_risk_map')
def closure_risk_map():
    try:
        # Initialize data structures
        region_data = {}
        
        # Map Korean region names to CSV filenames
        region_to_file = {
            '강원': 'gangwon.csv',
            '경기': 'gyeonggi.csv',
            '경남': 'gyeongnam.csv',
            '경북': 'gyeongbuk.csv',
            '광주': 'gwangju.csv',
            '대구': 'daegu.csv',
            '대전': 'daejeon.csv',
            '부산': 'busan.csv',
            '서울': 'seoul.csv',
            '세종': 'sejong.csv',
            '울산': 'ulsan.csv',
            '인천': 'incheon.csv',
            '전남': 'jeonnam.csv',
            '전북': 'jeonbuk.csv',
            '제주': 'jeju.csv',
            '충남': 'chungnam.csv',
            '충북': 'chungbuk.csv'
        }
        
        # Process data for each region
        for region, filename in region_to_file.items():
            filepath = os.path.join('data', 'raw', 'location_gone', filename)
            
            if not os.path.exists(filepath):
                continue
                
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='cp949')
            
            # Calculate risk metrics
            total_closures = len(df)
            recent_closures = len(df[df['폐교연도'] >= df['폐교연도'].max() - 5])
            unused_schools = len(df[df['활용현황'] == '미활용'])
            
            # Calculate risk score (you can adjust the formula based on your needs)
            risk_score = (recent_closures * 0.5 + unused_schools * 0.3 + total_closures * 0.2)
            
            region_data[region] = {
                'total_closures': total_closures,
                'recent_closures': recent_closures,
                'unused_schools': unused_schools,
                'risk_score': risk_score
            }
        
        # Calculate risk thresholds for color coding
        risk_scores = [data['risk_score'] for data in region_data.values()]
        if risk_scores:
            low_threshold = np.percentile(risk_scores, 33)
            high_threshold = np.percentile(risk_scores, 66)
            
            # Assign color codes
            for region in region_data:
                score = region_data[region]['risk_score']
                if score <= low_threshold:
                    region_data[region]['color'] = '#00C851'  # 낮은 위험 (green)
                elif score <= high_threshold:
                    region_data[region]['color'] = '#ffa700'  # 중간 위험 (orange)
                else:
                    region_data[region]['color'] = '#ff4444'  # 높은 위험 (red)
        
        # Create a Folium map centered on South Korea
        m = folium.Map(
            location=[36.5, 127.5],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Add markers for each region
        for region, coords in KOREA_REGIONS.items():
            if region in region_data:
                data = region_data[region]
                color = data['color']
                
                # Create popup content
                popup_content = f"""
                <div style="font-family: 'Malgun Gothic', sans-serif;">
                    <h4>{region}</h4>
                    <p>총 폐교: {data['total_closures']}개</p>
                    <p>최근 폐교: {data['recent_closures']}개</p>
                    <p>미활용: {data['unused_schools']}개</p>
                </div>
                """
                
                # Add circle marker
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=15,
                    popup=folium.Popup(popup_content, max_width=200),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=2
                ).add_to(m)
                
                # Add region label
                folium.map.Marker(
                    [coords['lat'], coords['lon']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; font-weight: bold;">{region}</div>',
                        icon_size=(100,20),
                        icon_anchor=(50,0)
                    )
                ).add_to(m)
        
        # Save the map to a temporary file
        map_path = os.path.join(app.static_folder, 'map.html')
        m.save(map_path)
        
        return render_template('closure_risk_map.html', 
                            region_data=region_data)
    
    except Exception as e:
        print(f"Error in closure_risk_map: {str(e)}")
        return render_template('closure_risk_map.html', 
                            error=f"Error processing data: {str(e)}",
                            region_data={})

@app.route('/multiculture')
def multiculture_view():
    try:
        # CSV 파일 경로 확인
        csv_path = 'data/raw/multiculture.csv'
        if not os.path.exists(csv_path):
            return render_template('multiculture.html', 
                                 error_message="데이터 파일을 찾을 수 없습니다. 관리자에게 문의하세요.",
                                 table_data=[],
                                 map_data=[])
        
        # CSV 파일 읽기
        df = pd.read_csv(csv_path, encoding='cp949', skiprows=1)  # 첫 번째 행 건너뛰기
        print("Raw data shape:", df.shape)
        print("Raw data columns:", df.columns.tolist())
        
        # 데이터 전처리
        # 빈 문자열을 NaN으로 변환
        df = df.replace('', np.nan)
        # 모든 요소가 NaN인 행 제거
        df = df.dropna(how='all')
        
        print("Processed data shape:", df.shape)
        print("Processed data columns:", df.columns.tolist())
        
        # 숫자형 컬럼 변환
        numeric_columns = df.columns[1:]  # 첫 번째 컬럼(지역)을 제외한 모든 컬럼
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 전처리된 데이터 저장
        processed_data_path = 'data/processed/processed_multiculture_data.csv'
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False, encoding='utf-8')
        
        # 테이블 데이터 준비
        table_data = df.to_dict('records')
        print("Table data sample:", table_data[:2] if table_data else "No data")
        
        # 지도 데이터 준비
        # '전체'가 포함된 컬럼 찾기
        total_column = None
        for col in df.columns:
            if '전체' in col:
                total_column = col
                break
        
        if total_column:
            map_data = df[['행정구역(읍면동)별(1)', total_column]].copy()
            map_data.columns = ['지역', '값']
            # NaN 값 제거
            map_data = map_data.dropna()
            # 값 컬럼을 숫자형으로 변환
            map_data['값'] = pd.to_numeric(map_data['값'], errors='coerce')
            # NaN 값이 있는 행 제거
            map_data = map_data.dropna()
        else:
            map_data = pd.DataFrame(columns=['지역', '값'])
        
        print("Map data sample:", map_data.head().to_dict('records') if not map_data.empty else "No map data")
        
        return render_template('multiculture.html', 
                             table_data=table_data,
                             map_data=map_data.to_dict('records'))
                             
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return render_template('multiculture.html', 
                             error_message=f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}",
                             table_data=[],
                             map_data=[])

@app.route('/static/data/<path:filename>')
def serve_static(filename):
    return send_from_directory('static/data', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 