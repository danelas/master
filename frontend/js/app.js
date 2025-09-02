// DOM Elements
const questionInput = document.getElementById('question');
const submitBtn = document.getElementById('submitBtn');
const loadingElement = document.getElementById('loading');
const resultsElement = document.getElementById('results');

// API endpoint
const API_URL = 'https://master-39uc.onrender.com/query';

// Model display names and colors
const modelInfo = {
    openai: {
        name: 'OpenAI (ChatGPT)',
        color: 'bg-green-100 text-green-800',
        borderColor: 'border-green-300',
        textColor: 'text-green-800'
    },
    anthropic: {
        name: 'Anthropic (Claude)',
        color: 'bg-blue-100 text-blue-800',
        borderColor: 'border-blue-300',
        textColor: 'text-blue-800'
    },
    vertex: {
        name: 'Google (Vertex AI)',
        color: 'bg-red-100 text-red-800',
        borderColor: 'border-red-300',
        textColor: 'text-red-800'
    },
    grok: {
        name: 'Grok (xAI)',
        color: 'bg-purple-100 text-purple-800',
        borderColor: 'border-purple-300',
        textColor: 'text-purple-800'
    }
};

// Handle form submission
async function submitQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }

    // Get selected models
    const selectedModels = Array.from(document.querySelectorAll('input[name="model"]:checked')).map(cb => cb.value);
    
    if (!question || selectedModels.length === 0) {
        alert('Please enter a question and select at least one model');
        return;
    }

    try {
        // Show loading state
        loadingElement.classList.remove('hidden');
        resultsElement.innerHTML = '';

        // Call the backend API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                models: selectedModels
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        resultsElement.innerHTML = `
            <div class="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                        </svg>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm text-red-700">
                            Error: ${error.message}. Make sure the backend server is running at http://localhost:8000
                        </p>
                    </div>
                </div>
            </div>
        `;
    } finally {
        // Reset loading state
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        loadingElement.classList.add('hidden');
    }
}

// Display results from the API
function displayResults(data) {
    resultsElement.innerHTML = '';
    
    // Add summary if available
    if (data.summary) {
        const summaryElement = document.createElement('div');
        summaryElement.className = 'bg-blue-50 border-l-4 border-blue-500 p-4 mb-6';
        summaryElement.innerHTML = `
            <div class="flex">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="ml-3">
                    <h3 class="text-sm font-medium text-blue-800">Analysis Summary</h3>
                    <p class="text-sm text-blue-700">
                        ${data.summary}
                        ${data.combined_response ? `Consensus: ${Math.round(data.combined_response.consensus_percentage)}% agreement` : ''}
                    </p>
                </div>
            </div>
        `;
        resultsElement.appendChild(summaryElement);
    }
    
    // Display combined response if available
    if (data.combined_response) {
        const combinedCard = document.createElement('div');
        combinedCard.className = 'bg-white rounded-lg shadow-md overflow-hidden mb-8 border-l-4 border-purple-500';
        
        combinedCard.innerHTML = `
            <div class="p-5">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold text-purple-800 flex items-center">
                        <svg class="h-5 w-5 mr-2 text-purple-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clip-rule="evenodd" />
                        </svg>
                        Combined Response
                    </h2>
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                        ${Math.round(data.combined_response.consensus_percentage)}% Consensus
                    </span>
                </div>
                
                <div class="prose max-w-none mb-6">
                    ${data.combined_response.combined_answer ? 
                        `<div class="whitespace-pre-wrap">${data.combined_response.combined_answer}</div>` : 
                        '<p class="text-gray-500">No combined response available.</p>'
                    }
                </div>
                
                ${data.combined_response.key_points && data.combined_response.key_points.length > 0 ? `
                <div class="mt-6">
                    <h3 class="text-sm font-medium text-gray-700 mb-2">Key Points</h3>
                    <ul class="list-disc pl-5 space-y-1">
                        ${data.combined_response.key_points.map(point => 
                            `<li class="text-sm text-gray-700">${point}</li>`
                        ).join('')}
                    </ul>
                </div>
                ` : ''}
                
                ${data.combined_response.contradictions && data.combined_response.contradictions.length > 0 ? `
                <div class="mt-6">
                    <h3 class="text-sm font-medium text-gray-700 mb-2">Areas of Disagreement</h3>
                    <div class="space-y-4">
                        ${data.combined_response.contradictions.map(contradiction => `
                            <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                                <div class="flex">
                                    <div class="flex-shrink-0">
                                        <svg class="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                        </svg>
                                    </div>
                                    <div class="ml-3">
                                        <p class="text-sm text-yellow-700">
                                            <span class="font-medium">${contradiction.issue || 'Disagreement found'}</span>
                                            ${contradiction.models ? 
                                                `<span class="text-xs text-yellow-600 block mt-1">
                                                    Models with different views: ${contradiction.models.join(', ')}
                                                </span>` 
                                                : ''
                                            }
                                        </p>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        
        resultsElement.appendChild(combinedCard);
    }

    // Add section header for individual model responses
    if (Object.keys(data.responses).length > 0) {
        const header = document.createElement('h2');
        header.className = 'text-xl font-bold text-gray-800 mb-4 mt-8 flex items-center';
        header.innerHTML = `
            <svg class="h-5 w-5 mr-2 text-gray-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
            </svg>
            Individual Model Responses
        `;
        resultsElement.appendChild(header);
    }
    
    // Add response cards for each model
    Object.entries(data.responses).forEach(([model, responseData]) => {
        const modelData = modelInfo[model] || {
            name: model,
            color: 'bg-gray-100',
            textColor: 'text-gray-800',
            borderColor: 'border-gray-300'
        };

        const statusClass = responseData.status === 'success' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800';

        const card = document.createElement('div');
        card.className = `response-card ${model} bg-white rounded-lg shadow-md overflow-hidden mb-6 border-l-4 ${modelData.borderColor} hover:shadow-lg transition-shadow duration-200`;
        
        card.innerHTML = `
            <div class="p-5">
                <div class="flex justify-between items-center mb-3">
                    <div class="flex items-center">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${modelData.color} ${modelData.textColor}">
                            ${modelData.name}
                        </span>
                        <span class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusClass}">
                            ${responseData.status}
                        </span>
                    </div>
                    <span class="text-xs text-gray-500">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="prose max-w-none">
                    <p class="whitespace-pre-wrap">${responseData.response}</p>
                </div>
            </div>
        `;
        
        resultsElement.appendChild(card);
    });
}

// Add event listener for Enter key in the textarea
questionInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitQuestion();
    }
});
