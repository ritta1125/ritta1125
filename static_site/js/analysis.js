document.addEventListener('DOMContentLoaded', () => {
    // Load policy recommendations from a JSON file
    fetch('data/policy_recommendations.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayPolicyRecommendations(data);
        })
        .catch(error => {
            console.error('Error loading policy recommendations:', error);
            displayError('Failed to load policy recommendations');
        });
});

function displayPolicyRecommendations(data) {
    const priorityAreas = document.getElementById('priority-areas');
    const supportMeasures = document.getElementById('support-measures');
    const monitoringSuggestions = document.getElementById('monitoring-suggestions');

    // Display priority areas
    data.priority_areas.forEach(area => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        div.innerHTML = `
            <h3>${area.title}</h3>
            <p>${area.description}</p>
        `;
        priorityAreas.appendChild(div);
    });

    // Display support measures
    data.support_measures.forEach(measure => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        div.innerHTML = `
            <h3>${measure.title}</h3>
            <p>${measure.description}</p>
        `;
        supportMeasures.appendChild(div);
    });

    // Display monitoring suggestions
    data.monitoring_suggestions.forEach(suggestion => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        div.innerHTML = `
            <h3>${suggestion.title}</h3>
            <p>${suggestion.description}</p>
        `;
        monitoringSuggestions.appendChild(div);
    });
}

function displayError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    document.querySelector('main').appendChild(errorDiv);
} 