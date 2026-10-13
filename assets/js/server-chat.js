/* ==========================================================================
   server-chat.js — the chat interface when Flask is serving the pages
   --------------------------------------------------------------------------
   This is the same conversation UI as the prototype, with one difference that
   matters: the answer no longer comes from JavaScript.

   The typed question is posted to /student/ask, and the Python backend runs
   the real pipeline before replying:

       question
         -> NLP preprocessing (tokenise, stopword removal, Porter stemming)
         -> TF-IDF vectorisation
         -> Naive Bayes  -> category
         -> cosine similarity -> closest stored question
         -> answer read from the MySQL questions table

   Nothing about matching or answering happens in this file. It sends the
   question, shows the typing indicator while the backend thinks, and renders
   whatever comes back.
   ========================================================================== */

(function () {
  "use strict";

  const thread = document.getElementById("chatThread");
  const scroll = document.getElementById("chatScroll");
  const welcome = document.getElementById("chatWelcome");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const chipRow = document.getElementById("chipRow");
  const suggest = document.getElementById("chatSuggest");

  let busy = false;

  /* ======================================================== rendering */

  function now() {
    return UI.formatTime(new Date().toISOString());
  }

  function scrollToEnd() {
    requestAnimationFrame(function () {
      scroll.scrollTop = scroll.scrollHeight;
    });
  }

  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg msg-user";
    row.innerHTML =
      '<div class="msg-body">' +
        '<div class="bubble">' + UI.escapeHtml(text) + "</div>" +
        '<div class="msg-meta"><span>' + now() + "</span></div>" +
      "</div>";
    thread.appendChild(row);
    scrollToEnd();
  }

  function addBotMessage(text, category) {
    const row = document.createElement("div");
    row.className = "msg msg-bot";
    row.innerHTML =
      '<div class="msg-avatar">' + UI.icon("bot") + "</div>" +
      '<div class="msg-body">' +
        '<div class="bubble">' + UI.escapeHtml(text) + "</div>" +
        '<div class="msg-meta">' +
          "<span>" + now() + "</span>" +
          (category
            ? '<span class="msg-source">' + UI.icon("tag", 12) +
              UI.escapeHtml(category) + "</span>"
            : "") +
          '<button class="msg-speak-btn" type="button" title="Listen to this answer" aria-label="Listen to answer">' +
            '<span class="btn-icon">' + UI.icon("volume", 12) + "</span>" +
            '<span class="btn-text">Listen</span>' +
          "</button>" +
        "</div>" +
      "</div>";
    thread.appendChild(row);
    scrollToEnd();

    const speakBtn = row.querySelector(".msg-speak-btn");
    if (speakBtn) {
      speakBtn.addEventListener("click", function () {
        speak(text, speakBtn);
      });
    }

    if (voiceEnabled) {
      speak(text, speakBtn);
    }
  }

  function showTyping() {
    const row = document.createElement("div");
    row.className = "msg msg-bot";
    row.id = "typingRow";
    row.innerHTML =
      '<div class="msg-avatar">' + UI.icon("bot") + "</div>" +
      '<div class="msg-body">' +
        '<div class="typing" aria-label="The chatbot is thinking">' +
          "<i></i><i></i><i></i>" +
        "</div>" +
      "</div>";
    thread.appendChild(row);
    scrollToEnd();
  }

  function hideTyping() {
    const row = document.getElementById("typingRow");
    if (row) row.remove();
  }

  /* ========================================================== voice assistant */

  const micBtn = document.getElementById("micBtn");
  const voiceMuteToggle = document.getElementById("voiceMuteToggle");
  const voiceMuteIcon = document.getElementById("voiceMuteIcon");
  const voiceMuteLabel = document.getElementById("voiceMuteLabel");
  const voiceStatusHint = document.getElementById("voiceStatusHint");
  const voiceStatusText = document.getElementById("voiceStatusText");

  let voiceEnabled = localStorage.getItem("sea_chatbot_voice_enabled") !== "false";
  let currentSpeakingBtn = null;

  function updateVoiceToggleUI() {
    if (!voiceMuteToggle) return;
    if (voiceEnabled) {
      voiceMuteToggle.classList.add("active");
      voiceMuteToggle.classList.remove("muted");
      if (voiceMuteIcon) voiceMuteIcon.innerHTML = UI.icon("volume", 13);
      if (voiceMuteLabel) voiceMuteLabel.textContent = "Voice answers: ON";
      voiceMuteToggle.title = "Click to mute automatic voice readout";
    } else {
      voiceMuteToggle.classList.remove("active");
      voiceMuteToggle.classList.add("muted");
      if (voiceMuteIcon) voiceMuteIcon.innerHTML = UI.icon("volumeX", 13);
      if (voiceMuteLabel) voiceMuteLabel.textContent = "Voice answers: OFF";
      voiceMuteToggle.title = "Click to enable automatic voice readout";
    }
  }

  if (voiceMuteToggle) {
    voiceMuteToggle.addEventListener("click", function () {
      voiceEnabled = !voiceEnabled;
      localStorage.setItem("sea_chatbot_voice_enabled", voiceEnabled ? "true" : "false");
      updateVoiceToggleUI();
      if (!voiceEnabled) {
        stopSpeaking();
      }
    });
  }

  function stopSpeaking() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    if (currentSpeakingBtn) {
      currentSpeakingBtn.classList.remove("speaking");
      const icon = currentSpeakingBtn.querySelector(".btn-icon");
      const text = currentSpeakingBtn.querySelector(".btn-text");
      if (icon) icon.innerHTML = UI.icon("volume", 12);
      if (text) text.textContent = "Listen";
      currentSpeakingBtn = null;
    }
  }

  function cleanForSpeech(raw) {
    return String(raw || "")
      .replace(/<[^>]*>/g, "")
      .replace(/[*_#`~[\]()]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function speak(text, btn) {
    if (!("speechSynthesis" in window)) return;
    const clean = cleanForSpeech(text);
    if (!clean) return;

    if (currentSpeakingBtn === btn && window.speechSynthesis.speaking) {
      stopSpeaking();
      return;
    }

    stopSpeaking();

    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    const voices = window.speechSynthesis.getVoices();
    const preferredVoice =
      voices.find(function (v) {
        return (
          v.lang.startsWith("en") &&
          (v.name.includes("India") ||
           v.name.includes("Natural") ||
           v.name.includes("Google") ||
           v.name.includes("Samantha"))
        );
      }) ||
      voices.find(function (v) {
        return v.lang.startsWith("en");
      });

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    if (btn) {
      currentSpeakingBtn = btn;
      btn.classList.add("speaking");
      const icon = btn.querySelector(".btn-icon");
      const label = btn.querySelector(".btn-text");
      if (icon) icon.innerHTML = UI.icon("volumeX", 12);
      if (label) label.textContent = "Stop";
    }

    utterance.onend = function () {
      stopSpeaking();
    };

    utterance.onerror = function () {
      stopSpeaking();
    };

    window.speechSynthesis.speak(utterance);
  }

  if ("speechSynthesis" in window && window.speechSynthesis.onvoiceschanged !== undefined) {
    window.speechSynthesis.onvoiceschanged = function () {
      window.speechSynthesis.getVoices();
    };
  }

  /* ---------------- Speech Recognition (Voice Input) ---------------- */
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isListening = false;

  function setListeningState(listening, statusMsg) {
    isListening = listening;
    if (micBtn) {
      if (listening) {
        micBtn.classList.add("listening");
        micBtn.title = "Listening... Click to stop";
      } else {
        micBtn.classList.remove("listening");
        micBtn.title = "Speak your question (Voice Input)";
      }
    }
    if (voiceStatusHint && voiceStatusText) {
      if (listening) {
        voiceStatusHint.classList.add("listening");
        voiceStatusText.innerHTML =
          UI.icon("mic", 14) + " " + (statusMsg || "Listening... Speak your question now");
      } else {
        voiceStatusHint.classList.remove("listening");
        voiceStatusText.innerHTML = "";
      }
    }
  }

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-IN";

    recognition.onstart = function () {
      stopSpeaking();
      setListeningState(true, "Listening... Speak your question");
    };

    recognition.onresult = function (event) {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          final += event.results[i][0].transcript;
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      const text = final || interim;
      if (text) {
        input.value = text;
        autoGrow();
        updateSendState();
      }
    };

    recognition.onerror = function (event) {
      console.warn("[speech recognition]", event.error);
      setListeningState(false);
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        UI.toast(
          "Microphone access was denied. Please allow microphone permissions in your browser.",
          "error"
        );
      } else if (event.error !== "no-speech") {
        UI.toast("Voice input error: " + event.error, "error");
      }
    };

    recognition.onend = function () {
      setListeningState(false);
      const query = input.value.trim();
      if (query && !busy) {
        ask(query);
      }
    };
  }

  if (micBtn) {
    micBtn.addEventListener("click", function () {
      if (!SpeechRecognition) {
        UI.toast(
          "Voice recognition is not supported in this browser. Please use Chrome, Edge, or Safari.",
          "info"
        );
        return;
      }
      if (isListening) {
        recognition.stop();
      } else {
        try {
          recognition.start();
        } catch (err) {
          console.warn("[speech start]", err);
        }
      }
    });
  }

  /* ========================================================== asking */

  function ask(text) {
    const question = String(text || "").trim();
    if (!question || busy) return;

    busy = true;
    stopSpeaking();
    if (welcome) welcome.classList.add("hidden");
    addUserMessage(question);
    input.value = "";
    autoGrow();
    updateSendState();
    removeChip(question);
    showTyping();

    fetch("/student/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
      credentials: "include",
    })
      .then(function (response) {
        if (response.status === 401 || response.redirected) {
          window.location.href = "/student/login";
          throw new Error("session expired");
        }
        if (!response.ok) throw new Error("server returned " + response.status);
        return response.json();
      })
      .then(function (result) {
        hideTyping();
        addBotMessage(result.answer, result.category);
      })
      .catch(function (error) {
        hideTyping();
        if (error.message === "session expired") return;
        addBotMessage(
          "I could not reach the college server just now. Please check that " +
          "the application is running and try again.",
          null
        );
        console.error("[chat]", error);
      })
      .finally(function () {
        busy = false;
        input.focus();
      });
  }

  /* ======================================================= composer */

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 132) + "px";
  }

  function updateSendState() {
    sendBtn.disabled = !input.value.trim();
  }

  input.addEventListener("input", function () {
    autoGrow();
    updateSendState();
  });

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask(input.value);
    }
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    ask(input.value);
  });

  /* ========================================================== chips */

  function removeChip(text) {
    if (!chipRow) return;
    Array.prototype.slice.call(chipRow.children).forEach(function (chip) {
      if (chip.dataset.question === text) chip.remove();
    });
    if (!chipRow.children.length && suggest) suggest.classList.add("hidden");
  }

  if (chipRow) {
    chipRow.addEventListener("click", function (event) {
      const chip = event.target.closest(".chip");
      if (chip) ask(chip.dataset.question);
    });
  }

  /* ----------------------------------------------------------- boot */

  /* Give the chips their icon, matching the prototype's styling */
  if (chipRow) {
    Array.prototype.slice.call(chipRow.children).forEach(function (chip) {
      chip.insertAdjacentHTML("afterbegin", UI.icon("messageSquare", 13));
    });
  }

  updateSendState();
  updateVoiceToggleUI();
  input.focus();
})();
