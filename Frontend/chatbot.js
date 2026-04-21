/* =============================================
   Cricket AI Chatbot — Full Page Interface
   With SSE Streaming, Markdown Rendering, Cricket UX
   ============================================= */

// Use global API_BASE if already defined (from script.js), else detect
if (typeof API_BASE === 'undefined') {
  var API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? window.location.protocol + '//' + window.location.hostname + ':8000'
    : '';
}

document.addEventListener('DOMContentLoaded', function() {
  if (typeof lucide !== 'undefined') lucide.createIcons();
  initChatbot();
  setTimeout(function() {
    var inp = document.getElementById('chatbot-input');
    if (inp) inp.focus();
  }, 100);
});

function initChatbot() {
  var messagesContainer = document.getElementById('chat-messages');
  var form = document.getElementById('chatbot-form');
  var input = document.getElementById('chatbot-input');
  var sendBtn = document.getElementById('send-btn');
  var suggestionsContainer = document.getElementById('chat-suggestions');
  var statusLabel = document.getElementById('chatbot-status-label');
  var scrollBottomBtn = document.getElementById('scroll-bottom-btn');
  var inputHint = document.getElementById('input-hint');

  if (!messagesContainer || !form || !input || !sendBtn) return;

  var isProcessing = false;
  var messageCount = 0;
  var streamMetaCounter = 0;
  var userIsScrolledUp = false;
  var streamRenderScheduled = false;

  // Categorized quick actions with icons
  var quickActionsData = [
    { icon: '\u{1F3C6}', text: 'Who won the 2011 World Cup?' },
    { icon: '\u2694\uFE0F', text: 'Compare Kohli and Ponting in World Cups' },
    { icon: '\u{1F3CF}', text: 'Tell me about the 2019 final drama' },
    { icon: '\u{1F464}', text: "Dhoni's complete World Cup career" },
    { icon: '\u{1F4CA}', text: 'Top 5 run scorers across all World Cups' },
    { icon: '\u{1F3B3}', text: 'Best bowling figures in WC history' },
    { icon: '\u{1F1EE}\u{1F1F3}', text: 'India vs Australia head-to-head in WCs' },
    { icon: '\u2B50', text: 'Most memorable World Cup moments' },
    { icon: '\u{1F3DF}\uFE0F', text: '2023 World Cup full summary' },
    { icon: '\u{1F4AF}', text: 'Who scored the most centuries in WCs?' },
  ];

  renderQuickActions();
  initScrollWatcher();

  // Close button for welcome message
  var welcomeMsg = document.getElementById('welcome-message');
  var welcomeCloseBtn = document.getElementById('welcome-close-btn');
  if (welcomeCloseBtn && welcomeMsg) {
    welcomeCloseBtn.addEventListener('click', function() {
      welcomeMsg.classList.add('minimized');
    });
  }

  // Form submission
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    if (input.value.trim() && !isProcessing) {
      sendMessage(input.value.trim());
    }
  });

  // Input validation + char counter
  input.addEventListener('input', function() {
    var hasText = input.value.trim().length > 0;
    sendBtn.disabled = !hasText || isProcessing;
    sendBtn.style.opacity = (hasText && !isProcessing) ? '1' : '0.5';
    // Char counter hint
    if (inputHint) {
      var len = input.value.length;
      if (len > 80) {
        inputHint.textContent = len + ' characters';
        inputHint.style.opacity = len > 300 ? '0.9' : '0.55';
      } else {
        inputHint.textContent = 'Press Enter to send';
        inputHint.style.opacity = '0.5';
      }
    }
  });

  // ─── Scroll to bottom watcher ───
  function initScrollWatcher() {
    if (!messagesContainer || !scrollBottomBtn) return;

    messagesContainer.addEventListener('scroll', function() {
      var distFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
      userIsScrolledUp = distFromBottom > 120;
      scrollBottomBtn.classList.toggle('visible', userIsScrolledUp);
    });

    scrollBottomBtn.addEventListener('click', function() {
      scrollToBottom(true);
    });
  }

  function scrollToBottom(force) {
    if (force || !userIsScrolledUp) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }

  function renderQuickActions() {
    if (!suggestionsContainer) return;
    suggestionsContainer.innerHTML = '';
    quickActionsData.forEach(function(action) {
      var btn = document.createElement('button');
      btn.className = 'suggestion-btn';
      btn.setAttribute('role', 'listitem');
      btn.innerHTML = '<span class="suggestion-icon">' + action.icon + '</span> ' + escapeHtml(action.text);
      btn.addEventListener('click', function() {
        if (!isProcessing) sendMessage(action.text);
      });
      suggestionsContainer.appendChild(btn);
    });
  }

  function setStatus(text) {
    if (statusLabel) statusLabel.textContent = text;
  }

  function sendMessage(text) {
    if (!text || !text.trim() || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    sendBtn.style.opacity = '0.5';
    messageCount++;

    // Minimize welcome on first message
    var welcome = document.getElementById('welcome-message');
    if (welcome && !welcome.classList.contains('minimized')) {
      welcome.classList.add('minimized');
    }

    // Hide suggestions after a few messages
    var sugWrapper = document.getElementById('suggestions-wrapper');
    if (sugWrapper && messageCount > 3) {
      sugWrapper.style.display = 'none';
    }

    addMessage('user', text.trim());
    input.value = '';
    setStatus('Searching...');

    // Try streaming first, fallback to regular
    sendStreamingMessage(text.trim());
  }

  function sendStreamingMessage(text) {
    var typingEl = showTypingIndicator();
    var botMsg = null;
    var botTextEl = null;
    var fullText = '';
    var metadata = {};
    streamMetaCounter++;
    var metaId = 'stream-meta-' + streamMetaCounter;

    fetch(API_BASE + '/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
    })
    .then(function(response) {
      if (!response.ok) {
        throw new Error('Server error (' + response.status + ')');
      }

      if (typingEl && typingEl.parentNode) typingEl.remove();

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      setStatus('Generating...');

      // Create bot message shell
      botMsg = document.createElement('div');
      botMsg.className = 'chat-msg bot';
      var time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });
      botMsg.innerHTML =
        '<div class="chat-msg-avatar">' +
          '<i data-lucide="bot" style="width:14px;height:14px;color:var(--accent)"></i>' +
        '</div>' +
        '<div class="chat-msg-bubble">' +
          '<div class="chat-msg-text"><span class="streaming-cursor"></span></div>' +
          '<div class="chat-msg-meta" id="' + metaId + '"></div>' +
          '<div class="chat-msg-time">' + time + '</div>' +
        '</div>';
      messagesContainer.appendChild(botMsg);
      botTextEl = botMsg.querySelector('.chat-msg-text');
      if (typeof lucide !== 'undefined') lucide.createIcons();

      function processChunk(result) {
        if (result.done) {
          finishStreaming();
          return;
        }

        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split('\n');
        buffer = lines.pop() || '';

        var eventType = '';
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i].trim();
          if (line.startsWith('event: ')) {
            eventType = line.substring(7);
          } else if (line.startsWith('data: ')) {
            var data = line.substring(6);
            handleSSEEvent(eventType, data);
            eventType = '';
          }
        }

        reader.read().then(processChunk).catch(function() { finishStreaming(); });
      }

      reader.read().then(processChunk).catch(function() { finishStreaming(); });
    })
    .catch(function(error) {
      if (typingEl && typingEl.parentNode) typingEl.remove();
      console.error('Stream error, falling back to regular:', error);
      sendRegularMessage(text);
    });

    function handleSSEEvent(type, data) {
      if (type === 'meta') {
        try { metadata = JSON.parse(data); } catch(e) {}
        setStatus('Writing response...');
      } else if (type === 'token') {
        try { fullText += JSON.parse(data); } catch(e) { fullText += data; }
        if (botTextEl && !streamRenderScheduled) {
          streamRenderScheduled = true;
          setTimeout(function() {
            streamRenderScheduled = false;
            if (botTextEl) {
              botTextEl.innerHTML = renderText(fullText) + '<span class="streaming-cursor"></span>';
              scrollToBottom(false);
            }
          }, 80);
        }
      } else if (type === 'done') {
        try {
          var doneData = JSON.parse(data);
          metadata.processing_time = doneData.processing_time;
        } catch(e) {}
      } else if (type === 'error') {
        try {
          var errData = JSON.parse(data);
          fullText += '\n\n\u26A0\uFE0F ' + (errData.error || 'An error occurred');
        } catch(e) {}
      }
    }

    function finishStreaming() {
      streamRenderScheduled = false;
      if (botTextEl) {
        botTextEl.innerHTML = renderText(fullText);
      }

      var metaEl = botMsg ? document.getElementById(metaId) : null;
      if (metaEl && metadata) {
        metaEl.innerHTML = buildMetaHtml(metadata);
      }

      // Add copy button to finished bot message
      if (botMsg && fullText) {
        addCopyButton(botMsg, fullText);
      }

      finishProcessing();
    }
  }

  function sendRegularMessage(text) {
    var typingEl = showTypingIndicator();
    setStatus('Processing...');

    fetch(API_BASE + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
      signal: AbortSignal.timeout ? AbortSignal.timeout(120000) : undefined
    })
    .then(function(response) {
      if (typingEl && typingEl.parentNode) typingEl.remove();
      if (!response.ok) {
        return response.json().catch(function() { return {}; }).then(function(errData) {
          throw new Error(errData.detail || 'Server error (' + response.status + ')');
        });
      }
      return response.json();
    })
    .then(function(data) {
      if (data && data.answer) {
        addMessage('bot', data.answer, data);
      }
    })
    .catch(function(error) {
      if (typingEl && typingEl.parentNode) typingEl.remove();
      var errorMsg;
      if (error.name === 'TimeoutError' || error.name === 'AbortError') {
        errorMsg = '\u23F1\uFE0F Request timed out. The server might be processing a complex query \u2014 please try again.';
      } else if (error.message && (error.message.indexOf('Failed to fetch') !== -1 || error.message.indexOf('NetworkError') !== -1)) {
        errorMsg = '\u{1F50C} Cannot connect to the server. Make sure the backend is running:\n\npython server.py';
      } else {
        errorMsg = '\u26A0\uFE0F ' + (error.message || 'An unexpected error occurred');
      }
      addMessage('bot', errorMsg, null, true);
    })
    .finally(function() {
      finishProcessing();
    });
  }

  function finishProcessing() {
    setStatus('Ready');
    isProcessing = false;
    sendBtn.disabled = false;
    sendBtn.style.opacity = '1';
    input.focus();
    scrollToBottom(true);
  }

  function addMessage(sender, text, metadata, isError) {
    var msg = document.createElement('div');
    msg.className = 'chat-msg ' + sender;

    var time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });
    var formattedText = (sender === 'bot') ? renderText(text) : escapeHtml(text);
    var bubbleClass = 'chat-msg-bubble' + (isError ? ' error-bubble' : '');

    var metaHtml = '';
    if (sender === 'bot' && metadata && !isError) {
      metaHtml = '<div class="chat-msg-meta">' + buildMetaHtml(metadata) + '</div>';
    }

    msg.innerHTML =
      '<div class="chat-msg-avatar">' +
        '<i data-lucide="' + (sender === 'user' ? 'user' : 'bot') + '" ' +
           'style="width:14px;height:14px;color:' + (sender === 'user' ? 'var(--secondary)' : 'var(--accent)') + '"></i>' +
      '</div>' +
      '<div class="' + bubbleClass + '">' +
        '<div class="chat-msg-text">' + formattedText + '</div>' +
        metaHtml +
        '<div class="chat-msg-time">' + time + '</div>' +
      '</div>';

    messagesContainer.appendChild(msg);
    if (typeof lucide !== 'undefined') lucide.createIcons();
    scrollToBottom(true);

    // Add copy button for non-error bot messages
    if (sender === 'bot' && !isError && text) {
      addCopyButton(msg, text);
    }
  }

  // ─── Copy button for bot messages ───
  function addCopyButton(msgEl, plainText) {
    var bubble = msgEl.querySelector('.chat-msg-bubble');
    if (!bubble || bubble.querySelector('.copy-btn')) return;
    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.title = 'Copy response';
    btn.setAttribute('aria-label', 'Copy response to clipboard');
    btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    btn.addEventListener('click', function() {
      navigator.clipboard.writeText(plainText).then(function() {
        btn.classList.add('copy-success');
        btn.title = 'Copied!';
        setTimeout(function() {
          btn.classList.remove('copy-success');
          btn.title = 'Copy response';
        }, 2000);
      }).catch(function() {
        // Fallback for older browsers
        var ta = document.createElement('textarea');
        ta.value = plainText;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.classList.add('copy-success');
        setTimeout(function() { btn.classList.remove('copy-success'); }, 2000);
      });
    });
    bubble.appendChild(btn);
  }

  // ─── Build metadata HTML ───
  function buildMetaHtml(metadata) {
    if (!metadata) return '';
    var parts = [];
    if (metadata.query_type) parts.push('<span class="query-type-badge">' + escapeHtml(String(metadata.query_type)) + '</span>');
    if (metadata.search_results) parts.push('Sources: ' + escapeHtml(String(metadata.search_results)));
    if (metadata.processing_time) parts.push(escapeHtml(String(metadata.processing_time)) + 's');
    return parts.join(' \u00B7 ');
  }

  // ─── Text Renderer ───
  function renderText(text) {
    if (!text) return '';
    return renderMarkdownFallback(text);
  }

  // ─── Fallback Markdown Renderer ───
  function renderMarkdownFallback(text) {
    if (!text) return '';
    var html = escapeHtml(text);

    // Tables
    html = html.replace(/^(\|.+\|)\n(\|[-:\| ]+\|)\n((?:\|.+\|\n?)*)/gm, function(match, header, sep, body) {
      var ths = header.split('|').filter(function(c) { return c.trim(); })
        .map(function(c) { return '<th>' + c.trim() + '</th>'; }).join('');
      var rows = body.trim().split('\n').map(function(row) {
        var tds = row.split('|').filter(function(c) { return c.trim(); })
          .map(function(c) { return '<td>' + c.trim() + '</td>'; }).join('');
        return '<tr>' + tds + '</tr>';
      }).join('');
      return '<table><thead><tr>' + ths + '</tr></thead><tbody>' + rows + '</tbody></table>';
    });

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Unordered lists
    html = html.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.+<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    // Clean up breaks after block elements
    html = html.replace(/<\/table><br>/g, '</table>');
    html = html.replace(/<\/ul><br>/g, '</ul>');
    html = html.replace(/<\/h[123]><br>/g, function(m) { return m.replace('<br>', ''); });
    html = html.replace(/<hr><br>/g, '<hr>');

    return html;
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function showTypingIndicator() {
    var typing = document.createElement('div');
    typing.className = 'chat-typing';
    typing.setAttribute('aria-label', 'Bot is typing');
    typing.innerHTML =
      '<div class="chat-msg-avatar">' +
        '<i data-lucide="bot" style="width:14px;height:14px;color:var(--accent)"></i>' +
      '</div>' +
      '<div class="typing-bubble">' +
        '<div class="typing-dots">' +
          '<span></span><span></span><span></span>' +
        '</div>' +
        '<span class="typing-label">Searching cricket database...</span>' +
      '</div>';

    messagesContainer.appendChild(typing);
    scrollToBottom(true);
    if (typeof lucide !== 'undefined') lucide.createIcons();
    return typing;
  }

  window.sendMessage = sendMessage;
}
