// Client-side interactions for Ampere AI Support Agent Console

document.addEventListener("DOMContentLoaded", () => {
  const convIdInput       = document.getElementById("conv-id-input");
  const btnNewConv        = document.getElementById("btn-new-conv");
  const btnRefreshConvs   = document.getElementById("btn-refresh-convs");
  const chatForm          = document.getElementById("chat-form");
  const userInput         = document.getElementById("user-input");
  const chatMessages      = document.getElementById("chat-messages");
  const btnClearChat      = document.getElementById("btn-clear-chat");
  const outcomeBadge      = document.getElementById("current-outcome-badge");
  const toggleFailure     = document.getElementById("toggle-failure");
  const convHistoryList   = document.getElementById("conv-history-list");

  // Inspector Elements
  const inspectorOutcome  = document.getElementById("inspector-outcome");
  const inspectorTicket   = document.getElementById("inspector-ticket");
  const inspectorLinks    = document.getElementById("inspector-links");
  const inspectorJson     = document.getElementById("inspector-json");

  // ─── Utilities ────────────────────────────────────────────────────────────

  function generateConvId() {
    return "conv-" + Math.floor(100000 + Math.random() * 900000);
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function timeAgo(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr.endsWith("Z") ? isoStr : isoStr + "Z");
    const diff = Math.floor((Date.now() - d) / 1000);
    if (diff < 60)    return "just now";
    if (diff < 3600)  return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    return Math.floor(diff / 86400) + "d ago";
  }

  // ─── Conversation History Sidebar ─────────────────────────────────────────

  async function loadConversationList() {
    try {
      const resp = await fetch("/api/conversations");
      if (!resp.ok) return;
      const data = await resp.json();
      renderConversationList(data.conversations || []);
    } catch (e) {
      console.warn("Could not load conversation list:", e);
    }
  }

  function renderConversationList(conversations) {
    const activeId = convIdInput.value.trim();

    if (!conversations.length) {
      convHistoryList.innerHTML = '<p class="history-empty">No past conversations yet.</p>';
      return;
    }

    convHistoryList.innerHTML = conversations.map(conv => {
      const isActive = conv.conversation_id === activeId;
      const preview  = conv.preview ? escapeHtml(conv.preview) : "<em>New conversation</em>";
      const shortId  = conv.conversation_id.length > 18
        ? conv.conversation_id.slice(0, 18) + "…"
        : conv.conversation_id;

      return `
        <div
          class="conv-history-item${isActive ? " active" : ""}"
          data-conv-id="${escapeHtml(conv.conversation_id)}"
          title="${escapeHtml(conv.conversation_id)}"
        >
          <div class="conv-item-main">
            <span class="conv-item-icon">💬</span>
            <div class="conv-item-content">
              <div class="conv-item-title">${preview}</div>
              <div class="conv-item-meta-row">
                <span class="conv-item-id">${escapeHtml(shortId)}</span>
                <span class="conv-item-pill">${conv.message_count} msg${conv.message_count !== 1 ? "s" : ""}</span>
                <span class="conv-item-time">${timeAgo(conv.last_at)}</span>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join("");

    // Attach click handlers
    convHistoryList.querySelectorAll(".conv-history-item").forEach(item => {
      item.addEventListener("click", () => {
        const id = item.getAttribute("data-conv-id");
        loadConversation(id);
      });
    });
  }

  async function loadConversation(convId) {
    convIdInput.value = convId;
    clearMessages();
    resetInspector();

    // Highlight active item immediately
    convHistoryList.querySelectorAll(".conv-history-item").forEach(el => {
      el.classList.toggle("active", el.getAttribute("data-conv-id") === convId);
    });

    try {
      const resp = await fetch(`/api/history/${encodeURIComponent(convId)}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const messages = data.messages || [];

      if (!messages.length) return;

      // Remove welcome banner before replaying messages
      const welcome = chatMessages.querySelector(".bot-welcome");
      if (welcome) welcome.remove();

      let lastData = null;
      for (const msg of messages) {
        let meta = null;
        try { meta = msg.metadata ? JSON.parse(msg.metadata) : null; } catch {}

        if (msg.role === "user") {
          appendMessage("user", msg.content);
        } else {
          const docLinks = meta?.doc_links || [];
          const outcome  = meta?.outcome   || null;
          const ticketId = meta?.ticket_id || null;
          appendMessage("bot", msg.content, docLinks, outcome, ticketId);
          // Track last bot message metadata for inspector
          if (outcome) {
            lastData = {
              conversation_id: convId,
              answer: msg.content,
              doc_links: docLinks,
              outcome,
              ticket_id: ticketId,
            };
          }
        }
      }

      if (lastData) updateInspector(lastData);

    } catch (e) {
      console.warn("Failed to load conversation history:", e);
    }
  }

  // ─── Event: New Conversation ───────────────────────────────────────────────

  btnNewConv.addEventListener("click", () => {
    convIdInput.value = generateConvId();
    clearMessages();
    resetInspector();
    // Deselect any active item in the history list
    convHistoryList.querySelectorAll(".conv-history-item").forEach(el => {
      el.classList.remove("active");
    });
  });

  // ─── Event: Manual Refresh History ────────────────────────────────────────

  btnRefreshConvs.addEventListener("click", () => {
    loadConversationList();
  });

  // ─── Event: Clear All Conversations ───────────────────────────────────────

  const btnClearAllConvs = document.getElementById("btn-clear-all-convs");
  if (btnClearAllConvs) {
    btnClearAllConvs.addEventListener("click", async () => {
      if (confirm("Delete all recent chat conversations? This cannot be undone.")) {
        try {
          await fetch("/api/conversations", { method: "DELETE" });
          convIdInput.value = generateConvId();
          clearMessages();
          resetInspector();
          await loadConversationList();
        } catch (err) {
          console.error("Failed to delete conversations:", err);
        }
      }
    });
  }

  // ─── Event: Toggle Webhook Failure ────────────────────────────────────────

  toggleFailure.addEventListener("change", async (e) => {
    try {
      await fetch("/api/simulate-webhook-failure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulate_failure: e.target.checked }),
      });
    } catch (err) {
      console.error("Failed to set failure simulation:", err);
    }
  });

  // ─── Event: Preset Buttons ────────────────────────────────────────────────

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const prompt = btn.getAttribute("data-prompt");
      if (prompt) {
        userInput.value = prompt;
        sendMessage(prompt);
      }
    });
  });

  // ─── Event: Sidebar Account Items ─────────────────────────────────────────

  document.querySelectorAll(".account-item").forEach((item) => {
    item.style.cursor = "pointer";
    const emailSpan = item.querySelector(".acc-email");
    if (emailSpan) {
      const email = emailSpan.textContent.trim();
      item.addEventListener("click", () => {
        const msg = email === "any other email"
          ? "My assistant is broken, email: unknown@domain.com"
          : `My assistant is broken, email: ${email}`;
        userInput.value = msg;
        sendMessage(msg);
      });
    }
  });

  // ─── Event: Clear Chat View ───────────────────────────────────────────────

  btnClearChat.addEventListener("click", () => {
    clearMessages();
  });

  // ─── UI Helpers ───────────────────────────────────────────────────────────

  function clearMessages() {
    chatMessages.innerHTML = `
      <div class="bot-welcome">
        <div class="bot-avatar">🤖</div>
        <div class="welcome-content">
          <h4>Welcome to Ampere Support!</h4>
          <p>I can help answer questions from our official documentation, inspect your assistant status in real time, or connect you with human support engineers.</p>
        </div>
      </div>
    `;
  }

  function resetInspector() {
    inspectorOutcome.textContent = "—";
    inspectorOutcome.className   = "field-value empty";
    inspectorTicket.textContent  = "—";
    inspectorTicket.className    = "field-value empty";
    inspectorLinks.textContent   = "—";
    inspectorLinks.className     = "field-value empty";
    inspectorJson.textContent    = "// Awaiting first request";
    outcomeBadge.textContent     = "Ready";
    outcomeBadge.className       = "outcome-badge-header outcome-idle";
  }

  // ─── Form Submission ──────────────────────────────────────────────────────

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;
    sendMessage(message);
  });

  // Auto-resize + Enter to send
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // ─── Send Message ─────────────────────────────────────────────────────────

  async function sendMessage(messageText) {
    const conversationId = convIdInput.value.trim() || generateConvId();
    convIdInput.value = conversationId;

    appendMessage("user", messageText);
    userInput.value = "";

    // Typing indicator
    const typingBubble = document.createElement("div");
    typingBubble.className = "chat-bubble bot typing-placeholder";
    typingBubble.innerHTML = `
      <div class="bubble-avatar">🤖</div>
      <div class="bubble-body">
        <div class="bubble-content" style="color: var(--text-muted); font-style: italic;">
          Agent is analyzing and querying tools...
        </div>
      </div>
    `;
    chatMessages.appendChild(typingBubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId, message: messageText }),
      });

      const data = await resp.json();
      typingBubble.remove();

      if (!resp.ok) {
        appendMessage("bot", `Error: ${data.detail || "Request failed"}`);
        return;
      }

      appendMessage("bot", data.answer, data.doc_links, data.outcome, data.ticket_id);
      updateInspector(data);

      // Refresh history list so this conversation appears / moves to top
      loadConversationList();

    } catch (err) {
      typingBubble.remove();
      appendMessage("bot", `Connection error: ${err.message}`);
    }
  }

  // ─── Append Message Bubble ────────────────────────────────────────────────

  function appendMessage(role, text, docLinks = [], outcome = null, ticketId = null) {
    const bubble  = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;

    const avatar  = role === "user" ? "👤" : "🤖";
    let citationsHtml = "";

    if (docLinks && docLinks.length > 0) {
      citationsHtml = `
        <div class="bubble-citations">
          ${docLinks.map(link => `
            <a href="${link}" target="_blank" rel="noopener" class="citation-chip">
              📖 ${link.replace("https://docs.ampere.sh", "")}
            </a>
          `).join("")}
        </div>
      `;
    }

    let metaHtml = "";
    if (role === "bot" && outcome) {
      metaHtml = `
        <div class="bubble-meta">
          <span class="bubble-tag outcome-${outcome}">${outcome.replace("_", " ")}</span>
          ${ticketId ? `<span class="bubble-tag" style="background: rgba(99,102,241,0.2); color: #a5b4fc;">Ticket: ${ticketId}</span>` : ""}
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="bubble-avatar">${avatar}</div>
      <div class="bubble-body">
        <div class="bubble-content">${escapeHtml(text)}</div>
        ${citationsHtml}
        ${metaHtml}
      </div>
    `;

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ─── Update Inspector Panel ───────────────────────────────────────────────

  function updateInspector(data) {
    inspectorOutcome.textContent = data.outcome;
    inspectorOutcome.className   = `field-value outcome-${data.outcome}`;

    outcomeBadge.textContent = data.outcome.replace("_", " ");
    outcomeBadge.className   = `outcome-badge-header outcome-${data.outcome}`;

    if (data.ticket_id) {
      inspectorTicket.textContent = data.ticket_id;
      inspectorTicket.className   = "field-value";
    } else {
      inspectorTicket.textContent = "None";
      inspectorTicket.className   = "field-value empty";
    }

    if (data.doc_links && data.doc_links.length > 0) {
      inspectorLinks.innerHTML  = data.doc_links.map(l =>
        `<div><a href="${l}" target="_blank" style="color:var(--accent-cyan);">${l}</a></div>`
      ).join("");
      inspectorLinks.className  = "field-value";
    } else {
      inspectorLinks.textContent = "None";
      inspectorLinks.className   = "field-value empty";
    }

    inspectorJson.textContent = JSON.stringify(data, null, 2);
  }

  // ─── Bootstrap ───────────────────────────────────────────────────────────

  // Load conversation history on startup
  loadConversationList();
});
