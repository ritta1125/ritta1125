// 전역 변수
let map;
let markers = [];

// 페이지 로드 시 초기화
$(document).ready(function() {
    // 지도 초기화
    if ($('#riskMap').length) {
        initMap();
    }
    
    // 데이터 로드
    loadData();
});

// 지도 초기화
function initMap() {
    map = L.map('riskMap').setView([36.5, 127.5], 7);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
}

// 데이터 로드
function loadData() {
    // 분석 데이터 로드
    $.get('/api/analysis', function(data) {
        updateDashboard(data);
    }).fail(function(error) {
        console.error('데이터 로드 실패:', error);
        showError('데이터를 불러오는데 실패했습니다.');
    });
    
    // 시각화 데이터 로드
    $.get('/api/visualization', function(data) {
        updateVisualizations(data);
    }).fail(function(error) {
        console.error('시각화 데이터 로드 실패:', error);
        showError('시각화 데이터를 불러오는데 실패했습니다.');
    });
}

// 대시보드 업데이트
function updateDashboard(data) {
    // 통계 업데이트
    updateStatistics(data);
    
    // 그래프 업데이트
    updateGraphs(data);
}

// 통계 업데이트
function updateStatistics(data) {
    // 고위험 지역 수
    if ($('#highRiskCount').length) {
        $('#highRiskCount').text(data.high_risk_areas.length);
    }
    
    // 다문화 비율
    if ($('#multiculturalRatio').length) {
        const ratio = (data.multicultural_analysis.avg_ratio * 100).toFixed(1);
        $('#multiculturalRatio').text(ratio + '%');
    }
}

// 그래프 업데이트
function updateGraphs(data) {
    // 상관관계 그래프
    if ($('#correlationPlot').length) {
        createCorrelationPlot(data.correlation);
    }
    
    // 지역별 비교 그래프
    if ($('#regionalComparison').length) {
        createRegionalComparison(data.regional_stats);
    }
    
    // 시계열 그래프
    if ($('#timeSeriesPlot').length) {
        createTimeSeriesPlot(data.time_series);
    }
}

// 시각화 업데이트
function updateVisualizations(data) {
    // 지도 마커 업데이트
    if (map) {
        updateMapMarkers(data);
    }
}

// 지도 마커 업데이트
function updateMapMarkers(data) {
    // 기존 마커 제거
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    // 새로운 마커 추가
    data.schools.forEach(school => {
        const marker = L.circleMarker(
            [school.latitude, school.longitude],
            {
                radius: 5,
                fillColor: getRiskColor(school.risk_index),
                color: '#000',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }
        );
        
        marker.bindPopup(`
            <strong>${school.name}</strong><br>
            위험도: ${school.risk_index.toFixed(2)}<br>
            다문화 비율: ${(school.multicultural_ratio * 100).toFixed(1)}%
        `);
        
        marker.addTo(map);
        markers.push(marker);
    });
}

// 위험도에 따른 색상 반환
function getRiskColor(riskIndex) {
    if (riskIndex >= 0.8) return '#ff0000';
    if (riskIndex >= 0.6) return '#ff6600';
    if (riskIndex >= 0.4) return '#ffcc00';
    if (riskIndex >= 0.2) return '#99ff00';
    return '#00ff00';
}

// 에러 메시지 표시
function showError(message) {
    // Bootstrap 알림 표시
    const alert = $(`
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('.container').prepend(alert);
    
    // 5초 후 자동으로 사라짐
    setTimeout(() => {
        alert.alert('close');
    }, 5000);
}

// 로딩 표시
function showLoading(element) {
    element.html('<div class="loading"></div>');
}

// 로딩 숨기기
function hideLoading(element) {
    element.find('.loading').remove();
} 