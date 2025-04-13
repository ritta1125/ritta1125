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

# Load environment variables
load_dotenv()

# Initialize Claude client
claude_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

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
        
        # 대시보드에 필요한 데이터만 추출
        dashboard_data = {
            'risk_distribution': analysis_results['closure_risk_analysis']['risk_distribution'],
            'regional_stats': analysis_results['closure_risk_analysis']['regional_stats'],
            'correlation': {
                'coefficient': analysis_results['correlation']['coefficient'],
                'details': analysis_results['correlation']['details']
            },
            'time_series': analysis_results['closure_risk_analysis']['time_series']
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
        heatmap_data = []
        
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
            
            # Add coordinates and risk score to heatmap data
            if region in KOREA_REGIONS:
                coords = KOREA_REGIONS[region]
                # Add multiple points for the region based on risk score
                for _ in range(int(risk_score * 10)):  # Scale the number of points
                    heatmap_data.append([coords['lat'], coords['lon'], risk_score])
        
        # Create a Folium map centered on South Korea
        m = folium.Map(
            location=[36.5, 127.5],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Add heatmap layer with custom gradient
        if heatmap_data:
            plugins.HeatMap(heatmap_data, 
                          radius=25,
                          blur=15,
                          max_zoom=17,
                          gradient={0.0: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'}).add_to(m)
        
        # Add markers for each region
        for region, coords in KOREA_REGIONS.items():
            if region in region_data:
                data = region_data[region]
                # Calculate color based on risk score
                if data['risk_score'] <= 0.3:
                    color = '#00C851'  # 낮은 위험 (green)
                elif data['risk_score'] <= 0.6:
                    color = '#ffa700'  # 중간 위험 (orange)
                else:
                    color = '#ff4444'  # 높은 위험 (red)
                
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

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Call Claude API
        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return jsonify({'response': message.content[0].text})
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 