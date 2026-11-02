/**
 * LUNG - Frontend Application Logic
 * Handles file drag-and-drop, image previewing, submission loading animations,
 * and the floating Academic AI Chatbot widget.
 */

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initFileUpload();
    initFormLoading();
    initChatbot();
});

/* --------------------------------------------------------------------------
   1. Mobile Navigation
   -------------------------------------------------------------------------- */
function initNavigation() {
    const mobileToggle = document.getElementById("mobileToggle");
    const navLinks = document.getElementById("navLinks");

    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener("click", () => {
            navLinks.classList.toggle("open");
        });
    }
}

/* --------------------------------------------------------------------------
   2. Drag-and-Drop & Image Preview
   -------------------------------------------------------------------------- */
function initFileUpload() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const dropContent = document.getElementById("dropZoneContent");
    const previewArea = document.getElementById("previewArea");
    const previewImg = document.getElementById("imagePreview");
    const previewName = document.getElementById("previewName");
    const previewSize = document.getElementById("previewSize");
    const clearBtn = document.getElementById("clearPreviewBtn");

    if (!dropZone || !fileInput) return;

    // Drag and Drop Events
    ["dragenter", "dragover"].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    if (clearBtn) {
        clearBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.value = "";
            previewImg.src = "#";
            previewArea.style.display = "none";
            dropContent.style.display = "block";
        });
    }

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/") && !file.name.endsWith(".dcm")) {
            alert("Please select a valid medical image file (PNG, JPG, JPEG, WEBP, DCM, TIFF).");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name;
            const sizeInKb = (file.size / 1024).toFixed(1);
            previewSize.textContent = `${sizeInKb} KB`;

            dropContent.style.display = "none";
            previewArea.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
}

/* --------------------------------------------------------------------------
   3. Form Submission Loading States
   -------------------------------------------------------------------------- */
function initFormLoading() {
    // Upload Prediction Form
    const uploadForm = document.getElementById("uploadForm");
    const submitBtn = document.getElementById("submitBtn");
    const btnSpinner = document.getElementById("btnSpinner");

    if (uploadForm && submitBtn) {
        uploadForm.addEventListener("submit", () => {
            const btnText = submitBtn.querySelector(".btn-text");
            if (btnText) btnText.textContent = "Processing CT & Computing Grad-CAM...";
            if (btnSpinner) btnSpinner.style.display = "inline-block";
            submitBtn.disabled = true;
            submitBtn.style.opacity = "0.75";
            uploadForm.submit();
        });
    }

    // Counterfactual What-If Form
    const cfForm = document.getElementById("cfForm");
    const runCfBtn = document.getElementById("runCfBtn");
    const cfSpinner = document.getElementById("cfSpinner");

    if (cfForm && runCfBtn) {
        cfForm.addEventListener("submit", () => {
            const btnText = runCfBtn.querySelector(".btn-text");
            if (btnText) btnText.textContent = "Perturbing Region & Re-Evaluating...";
            if (cfSpinner) cfSpinner.style.display = "inline-block";
            runCfBtn.disabled = true;
            runCfBtn.style.opacity = "0.75";
            cfForm.submit();
        });
    }
}

/* --------------------------------------------------------------------------
   4. Floating Academic AI Chatbot
   -------------------------------------------------------------------------- */
function initChatbot() {
    const toggleBtn = document.getElementById("chatbotToggleBtn");
    const closeBtn = document.getElementById("chatbotCloseBtn");
    const chatWindow = document.getElementById("chatbotWindow");
    const chatForm = document.getElementById("chatbotForm");
    const chatInput = document.getElementById("chatbotInput");
    const chatMessages = document.getElementById("chatbotMessages");
    const chatSuggestions = document.getElementById("chatbotSuggestions");

    if (!toggleBtn || !chatWindow) return;

    // Toggle Chat Window
    toggleBtn.addEventListener("click", () => {
        const isHidden = chatWindow.style.display === "none" || chatWindow.style.display === "";
        chatWindow.style.display = isHidden ? "flex" : "none";
        if (isHidden) {
            chatInput.focus();
            if (chatSuggestions && chatSuggestions.children.length === 0) {
                loadSuggestions();
            }
        }
    });

    if (closeBtn) {
        closeBtn.addEventListener("click", () => {
            chatWindow.style.display = "none";
        });
    }

    // Submit user message
    if (chatForm) {
        chatForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;

            // Render User Bubble
            appendMessage(message, "user");
            chatInput.value = "";

            // Render Temporary Typing Indicator
            const typingIndicator = appendTypingIndicator();

            try {
                const response = await fetch("/api/chatbot", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();

                typingIndicator.remove();
                appendBotResponse(data);
            } catch (err) {
                typingIndicator.remove();
                appendMessage("Sorry, I encountered an issue connecting to the XAI FAQ engine. Please try again.", "bot");
            }
        });
    }

    async function loadSuggestions() {
        try {
            const res = await fetch("/api/chatbot/suggestions");
            const data = await res.json();
            if (data && data.suggestions) {
                renderSuggestionChips(data.suggestions);
            }
        } catch (e) {
            // Silently ignore suggestion load error
        }
    }

    function renderSuggestionChips(queries) {
        if (!chatSuggestions) return;
        chatSuggestions.innerHTML = "";
        queries.forEach(q => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "suggestion-chip";
            chip.textContent = q;
            chip.addEventListener("click", () => {
                chatInput.value = q;
                chatForm.dispatchEvent(new Event("submit"));
            });
            chatSuggestions.appendChild(chip);
        });
    }

    function appendMessage(text, sender) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}-bubble animate-fade-in`;
        bubble.textContent = text;
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return bubble;
    }

    function appendBotResponse(data) {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble bot-bubble animate-fade-in";

        let html = "";
        if (data.topic) {
            html += `<div style="font-size:0.75rem; font-weight:700; color:var(--color-accent); margin-bottom:0.25rem;">[${escapeHtml(data.topic)}]</div>`;
        }
        html += `<div>${escapeHtml(data.answer).replace(/\n/g, "<br>")}</div>`;
        bubble.innerHTML = html;
        chatMessages.appendChild(bubble);

        // Update suggestions if follow-ups provided
        if (data.follow_ups && data.follow_ups.length > 0) {
            renderSuggestionChips(data.follow_ups);
        }

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendTypingIndicator() {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble bot-bubble animate-fade-in";
        bubble.innerHTML = '<span class="spinner" style="width:14px; height:14px; margin-right:6px;"></span> Searching knowledge base...';
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return bubble;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
}
