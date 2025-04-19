// Chat functionality
document.addEventListener('DOMContentLoaded', () => {
    const chatWidget = document.querySelector('.chat-widget');
    const closeChat = document.querySelector('.close-chat');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Toggle chat visibility
    closeChat.addEventListener('click', () => {
        chatWidget.style.display = 'none';
    });

    // Send message
    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            // Add user message to chat
            addMessage(message, 'user');
            userInput.value = '';

            // Call Claude API
            fetchClaudeResponse(message)
                .then(response => {
                    addMessage(response, 'bot');
                })
                .catch(error => {
                    console.error('Error:', error);
                    addMessage('Sorry, there was an error processing your request.', 'bot');
                });
        }
    }

    // Add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Fetch Claude API response
    async function fetchClaudeResponse(message) {
        // Note: You'll need to set up a serverless function or API endpoint
        // to handle the Claude API calls securely
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        return data.response;
    }
}); 